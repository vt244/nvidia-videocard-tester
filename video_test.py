#!/usr/bin/env python3
"""Discover CUDA devices, build native kernels, and run isolated GPU diagnostics."""
import argparse
import concurrent.futures
import csv
import ctypes
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess as sp
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
PROFILES = [
    # Conservative driver API thresholds; no dependency on minor compatibility.
    (12040, 'nvidia/cuda:12.4.1-devel-ubuntu22.04', {50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90}),
    (12090, 'nvidia/cuda:12.9.1-devel-ubuntu22.04', {50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 103, 120}),
    (12080, 'nvidia/cuda:12.8.1-devel-ubuntu22.04', {50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 120}),
    (13000, 'nvidia/cuda:13.0.2-devel-ubuntu22.04', {75, 80, 86, 87, 89, 90, 100, 103, 110, 120, 121}),
    (11080, 'nvidia/cuda:11.8.0-devel-ubuntu22.04', {35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90}),
    (10020, 'nvcr.io/nvidia/cuda:10.2-devel-ubuntu18.04', {30, 32, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75}),
    (8000, 'nvcr.io/nvidia/cuda:8.0-devel-ubuntu16.04', {20, 21, 30, 32, 35, 37, 50, 52, 53, 60, 61, 62}),
]


def command(args, **kwargs):
    return sp.run(args, check=True, text=True, **kwargs)


def output(args):
    return command(args, stdout=sp.PIPE, stderr=sp.PIPE).stdout.strip()


def pci_key(value):
    domain, bus, devfunc = value.lower().split(':')
    return (int(domain, 16), int(bus, 16), devfunc)


def inventory():
    fields = 'index,uuid,pci.bus_id,name,memory.total'
    rows = list(csv.reader(output(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits']).splitlines()))
    if not rows:
        raise RuntimeError('NVIDIA GPU не обнаружены')
    lib = ctypes.CDLL('libcuda.so.1')
    def cuda(name, *args):
        rc = getattr(lib, name)(*args)
        if rc:
            raise RuntimeError(f'{name}: CUDA error {rc}; проверьте поддержку карт драйвером и доступ к устройствам')
    cuda('cuInit', 0)
    version = ctypes.c_int()
    cuda('cuDriverGetVersion', ctypes.byref(version))
    count = ctypes.c_int()
    cuda('cuDeviceGetCount', ctypes.byref(count))
    capabilities = {}
    for ordinal in range(count.value):
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        cuda('cuDeviceGet', ctypes.byref(device), ordinal)
        cuda('cuDeviceGetAttribute', ctypes.byref(major), 75, device)
        cuda('cuDeviceGetAttribute', ctypes.byref(minor), 76, device)
        bus = ctypes.create_string_buffer(32)
        cuda('cuDeviceGetPCIBusId', bus, len(bus), device)
        capabilities[pci_key(bus.value.decode())] = major.value * 10 + minor.value
    devices = []
    for row in rows:
        index, uuid, pci, name, memory = [part.strip() for part in row]
        if pci_key(pci) not in capabilities:
            raise RuntimeError(f'{name} ({uuid}): отсутствует среди CUDA devices; возможен MIG или несовместимый драйвер')
        devices.append(dict(index=int(index), uuid=uuid, pci=pci, name=name,
                            memory_mib=int(memory), arch=capabilities[pci_key(pci)], driver_api=version.value))
    return devices


def select_profile(arch, driver_api, override=None):
    if override:
        # nvcc and the actual CUDA launch verify custom/future architectures.
        if not re.fullmatch(r'[A-Za-z0-9._/:@-]+', override):
            raise ValueError('Invalid --cuda-image')
        return override
    for minimum, image, supported in PROFILES:
        if arch in supported and driver_api >= minimum:
            return image
    raise ValueError(f'Нет совместимого профиля для sm_{arch}, CUDA driver API {driver_api}. '
                     'Задайте --cuda-image с devel-образом, поддерживающим GPU и драйвер. '
                     'CUDA 1.x и MIG не поддерживаются этим набором утилит.')


def make_plan(devices, args):
    selected = devices
    if args.gpus != 'all':
        tokens = args.gpus.split(',')
        selected = [d for d in devices if str(d['index']) in tokens or d['uuid'] in tokens]
        found = {str(d['index']) for d in selected} | {d['uuid'] for d in selected}
        if any(t not in found for t in tokens):
            raise ValueError('--gpus: неизвестный индекс или UUID')
    for device in selected:
        device['cuda_image'] = select_profile(device['arch'], device['driver_api'], args.cuda_image)
        # Small GPUs need smaller matrices; compare needs at least two results.
        device['matrix_size'] = 1024 if device['memory_mib'] < 1024 else 2048 if device['memory_mib'] < 4096 else 8192
        cuda_tag = device['cuda_image'].split(':', 1)[-1].split('-devel')[0]
        # Include base image identity so custom images cannot reuse an incompatible build.
        import hashlib
        digest = hashlib.sha256(device['cuda_image'].encode()).hexdigest()[:8]
        device['image'] = f"video-test:cuda{cuda_tag}-sm{device['arch']}-m{device['matrix_size']}-{digest}"
    if not selected:
        raise ValueError('Не выбрана ни одна GPU')
    return selected


def build_images(plan, rebuild=False):
    built = set()
    for d in plan:
        if d['image'] in built:
            continue
        built.add(d['image'])
        exists = sp.run(['docker', 'image', 'inspect', d['image']], stdout=sp.DEVNULL, stderr=sp.DEVNULL).returncode == 0
        if exists and not rebuild:
            continue
        command(['docker', 'build', '--build-arg', 'CUDA_IMAGE=' + d['cuda_image'],
                 '--build-arg', f"CUDA_ARCHS={d['arch']}", '--build-arg', f"MATRIX_SIZE={d['matrix_size']}",
                 '-t', d['image'], str(ROOT)])
        # A CPU-only check: useful while another diagnostic still owns the GPUs.
        command(['docker', 'run', '--rm', '--network', 'none', d['image'], '--help'])


def assert_idle(plan):
    active = output(['docker', 'ps', '--filter', 'label=video-test.managed=true', '--format', '{{.Names}}'])
    legacy = output(['docker', 'ps', '--filter', 'name=^/video-test$', '--format', '{{.Names}}'])
    if active or legacy:
        raise RuntimeError('Уже выполняется диагностика: ' + ', '.join((active + '\n' + legacy).split()))
    # CUDA/compute processes on selected devices; a later race is still possible.
    busy = output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    selected = {d['uuid'] for d in plan}
    for row in csv.reader(busy.splitlines()):
        if row and row[0].strip() in selected:
            raise RuntimeError('GPU занята вычислениями: ' + ','.join(row))
    # Also refuse a selected GPU with substantial existing allocations (e.g. graphics).
    used = output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'])
    for row in csv.reader(used.splitlines()):
        if row[0].strip() in selected and (not row[1].strip().isdigit() or int(row[1]) > 128):
            raise RuntimeError('GPU занята или объём занятой памяти неизвестен: ' + ','.join(row))


def parse_telemetry(text, expected, limit):
    rows = list(csv.reader(text.strip().splitlines()))
    seen = set()
    for row in rows:
        if len(row) != 7:
            raise ValueError('Неполная строка телеметрии')
        uuid = row[2].strip()
        if uuid not in expected:
            continue
        seen.add(uuid)
        temperature = row[3].strip()
        if temperature.isdigit() and int(temperature) >= limit:
            raise ValueError(f'{uuid}: температура {temperature} >= {limit} °C')
        if not temperature.isdigit():
            raise ValueError(f'{uuid}: температура недоступна')
    if seen != expected:
        raise ValueError('Не все выбранные GPU доступны в телеметрии')


def run_diagnostics(plan, args):
    # A local lock stops two simultaneous launches before either starts a container.
    import fcntl
    lock = open(ROOT / '.run.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Другой сценарий уже выполняет диагностику')
    assert_idle(plan)
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    run = ROOT / 'results' / stamp
    run.mkdir(parents=True)
    latest = ROOT / 'results' / 'latest'
    temp_link = latest.with_name('.latest-' + stamp)
    temp_link.symlink_to(run.name)
    temp_link.replace(latest)
    (run / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    (run / 'before.txt').write_text(output(['nvidia-smi', '-q']) + '\n')
    print(f'Журналы: {run}', flush=True)
    containers = []
    abort = threading.Event()
    finished = threading.Event()
    abort_reason = []
    def stop_all(reason):
        if not abort.is_set():
            abort_reason.append(reason)
            (run / 'abort.txt').write_text(reason + '\n')
            abort.set()
        # Lock-free repeated docker stop is harmless; never touch unrelated containers.
        for name in list(containers):
            sp.run(['docker', 'stop', '-t', '10', name], stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=25)
    def interrupted(signum, frame):
        abort_reason.append(f'Сигнал {signum}')
        abort.set()
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    def monitor():
        try:
            with (run / 'telemetry.csv').open('a') as log:
                while not finished.is_set():
                    data = sp.run(['nvidia-smi', '--query-gpu=timestamp,index,uuid,temperature.gpu,utilization.gpu,memory.used,power.draw',
                                   '--format=csv,noheader,nounits'], check=True, text=True, stdout=sp.PIPE, stderr=sp.PIPE, timeout=15).stdout
                    log.write(data)
                    log.flush()
                    parse_telemetry(data, {d['uuid'] for d in plan}, args.temperature_limit)
                    if abort.is_set():
                        raise RuntimeError('; '.join(abort_reason) or 'Остановка')
                    finished.wait(5)
        except Exception as exc:
            stop_all(str(exc))
    monitor_thread = None
    try:
        for d in plan:
            if abort.is_set():
                raise RuntimeError('Запуск прерван')
            name = f"video-test-{stamp}-gpu{d['index']}"
            directory = run / f"gpu-{d['index']}"
            directory.mkdir()
            command(['docker', 'run', '-d', '--name', name, '--label', 'video-test.managed=true',
                     '--gpus', 'device=' + d['uuid'], '--network', 'none', '--stop-timeout', '10',
                     '-e', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility',
                     '-e', f'MEM_PASSES={args.passes}', '-e', f'MEM_ITERATIONS={args.iterations}',
                     '-e', f'MEM_TIMEOUT={args.memory_timeout}', '-e', f'BURN_SECONDS={args.seconds}',
                     '-e', f'BURN_MEMORY={args.burn_memory}%', '-e', f'SMOKE={int(args.smoke)}',
                     '-v', f'{directory}:/results', d['image']])
            containers.append(name)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        # Wait in parallel so a failure on any card stops its peers promptly.
        def wait_one(name):
            rc = int(output(['docker', 'wait', name]))
            directory = run / f"gpu-{next(d['index'] for d in plan if name.endswith('-gpu' + str(d['index'])))}"
            (directory / 'container.exit').write_text(str(rc) + '\n')
            logs = sp.run(['docker', 'logs', name], text=True, stdout=sp.PIPE, stderr=sp.STDOUT)
            (directory / 'container.log').write_text(logs.stdout)
            status = directory / 'status.txt'
            if rc or not status.exists() or status.read_text().strip() != 'PASSED':
                stop_all(f'{name}: тест завершился с ошибкой или не полностью, exit={rc}')
            return rc
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(containers)) as pool:
            futures = [pool.submit(wait_one, name) for name in containers]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        finished.set()
        monitor_thread.join(timeout=20)
        if abort.is_set():
            raise RuntimeError('; '.join(abort_reason))
        (run / 'status.txt').write_text('PASSED\n')
        print('Все выбранные GPU: PASSED', flush=True)
    except BaseException as exc:
        stop_all(str(exc))
        (run / 'status.txt').write_text('FAILED_OR_ABORTED\n')
        raise
    finally:
        finished.set()
        if monitor_thread:
            monitor_thread.join(timeout=20)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        result = sp.run(['nvidia-smi', '-q'], text=True, stdout=sp.PIPE, stderr=sp.STDOUT, timeout=20)
        (run / 'after.txt').write_text(result.stdout)
        (run / 'end.txt').write_text(dt.datetime.now().astimezone().isoformat() + '\n')
        lock.close()


def positive(value):
    val = int(value)
    if val <= 0:
        raise argparse.ArgumentTypeError('нужно положительное целое число')
    return val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='показать GPU и подходящие образы, без сборки и нагрузки')
    mode.add_argument('--build-only', action='store_true', help='собрать образы, без нагрузки')
    parser.add_argument('--gpus', default='all', help='all, индексы через запятую или GPU UUID')
    parser.add_argument('--cuda-image', default=os.environ.get('CUDA_IMAGE'), help='свой CUDA devel image вместо автоматического выбора')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--seconds', type=positive, default=900)
    parser.add_argument('--passes', type=positive, default=1)
    parser.add_argument('--iterations', type=positive, default=100)
    parser.add_argument('--memory-timeout', type=positive, default=1800)
    parser.add_argument('--temperature-limit', type=positive, default=85)
    parser.add_argument('--burn-memory', type=positive, default=90, help='процент свободной памяти для gpu-burn (1–95)')
    parser.add_argument('--smoke', action='store_true', help='короткая проверка запуска: 64 МиБ memtest и 5 секунд gpu-burn')
    args = parser.parse_args()
    if not 1 <= args.burn_memory <= 95 or not 1 <= args.temperature_limit <= 95:
        parser.error('--burn-memory и --temperature-limit должны быть от 1 до 95')
    if args.smoke:
        args.seconds = 5
    plan = make_plan(inventory(), args)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.plan:
        return
    build_images(plan, args.rebuild)
    if not args.build_only:
        run_diagnostics(plan, args)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError, sp.SubprocessError) as exc:
        print('ОШИБКА: ' + str(exc), file=sys.stderr, flush=True)
        sys.exit(1)
