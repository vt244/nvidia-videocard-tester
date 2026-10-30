import argparse
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import video_test as v


class CompatibilityTests(unittest.TestCase):
    def test_profiles_across_generations(self):
        examples = [(20, 8000, ':8.0-'), (21, 8000, ':8.0-'),
                    (30, 10020, ':10.2-'), (35, 11080, ':11.8.0-'),
                    (52, 12040, ':12.4.1-'), (61, 13030, ':12.4.1-'),
                    (75, 13030, ':12.4.1-'), (86, 12040, ':12.4.1-'),
                    (89, 11080, ':11.8.0-'), (90, 12040, ':12.4.1-'),
                    (100, 12090, ':12.9.1-'), (120, 13030, ':12.9.1-'),
                    (120, 12080, ':12.8.1-'), (121, 13030, ':13.0.2-')]
        for arch, driver, expected in examples:
            with self.subTest(arch=arch):
                self.assertIn(expected, v.select_profile(arch, driver))

    def test_old_driver_uses_older_toolkit(self):
        self.assertIn(':10.2-', v.select_profile(75, 11040))

    def test_unknown_or_unsupported_is_not_silently_accepted(self):
        for arch, driver in [(13, 13030), (120, 12040), (999, 13030), (89, 10020)]:
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                v.select_profile(arch, driver)

    def test_custom_future_toolkit(self):
        self.assertEqual(v.select_profile(121, 13030, 'nvidia/cuda:13.0.2-devel-ubuntu22.04'),
                         'nvidia/cuda:13.0.2-devel-ubuntu22.04')
        with self.assertRaises(ValueError):
            v.select_profile(75, 13030, 'image; shell')

    def test_mixed_gpus_and_small_memory(self):
        devices = [dict(index=0, uuid='GPU-a', arch=75, driver_api=13030, memory_mib=20480),
                   dict(index=1, uuid='GPU-b', arch=52, driver_api=13030, memory_mib=512),
                   dict(index=2, uuid='GPU-c', arch=120, driver_api=13030, memory_mib=32768)]
        plan = v.make_plan(devices, argparse.Namespace(gpus='all', cuda_image=None))
        self.assertEqual([d['matrix_size'] for d in plan], [8192, 1024, 8192])
        self.assertEqual(len({d['image'] for d in plan}), 3)
        selected = v.make_plan(devices, argparse.Namespace(gpus='GPU-b,2', cuda_image=None))
        self.assertEqual([d['index'] for d in selected], [1, 2])
        with self.assertRaises(ValueError):
            v.make_plan(devices, argparse.Namespace(gpus='99', cuda_image=None))

    def test_pci_domain_width(self):
        self.assertEqual(v.pci_key('00000000:81:00.0'), v.pci_key('0000:81:00.0'))


class TelemetryTests(unittest.TestCase):
    def telemetry(self, temperatures):
        return '\n'.join(f'now, {i}, GPU-{i}, {t}, 100, 1000, 100' for i, t in enumerate(temperatures))

    def test_arbitrary_gpu_count(self):
        v.parse_telemetry(self.telemetry([50] * 8), {f'GPU-{i}' for i in range(8)}, 85)

    def test_overheat_first_of_many(self):
        with self.assertRaises(ValueError):
            v.parse_telemetry(self.telemetry([85, 50, 50]), {'GPU-0', 'GPU-1', 'GPU-2'}, 85)

    def test_missing_gpu_and_missing_temperature(self):
        with self.assertRaises(ValueError):
            v.parse_telemetry(self.telemetry([50]), {'GPU-0', 'GPU-1'}, 85)
        with self.assertRaises(ValueError):
            v.parse_telemetry(self.telemetry(['N/A']), {'GPU-0'}, 85)

    def test_unselected_gpu_is_ignored(self):
        v.parse_telemetry(self.telemetry([50, 90]), {'GPU-0'}, 85)


if __name__ == '__main__':
    unittest.main()
