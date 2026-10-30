"""CPU-only integration checks: emulate tools that return zero despite failures."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

IMAGE = os.environ.get('VIDEO_TEST_IMAGE')


@unittest.skipUnless(IMAGE, 'set VIDEO_TEST_IMAGE for Docker integration checks')
class ContainerTests(unittest.TestCase):
    def run_case(self, memory_text, burn_text, memory_ok=True):
        with tempfile.TemporaryDirectory(prefix='video-test-check-') as tmp:
            root = Path(tmp)
            results = root / 'results'
            results.mkdir()
            mem = root / 'memory'
            burn = root / 'burn'
            mem.write_text('#!/bin/sh\ncat <<\'LOG\'\n' + memory_text + '\nLOG\nexit 0\n')
            burn.write_text('#!/bin/sh\ncat <<\'LOG\'\n' + burn_text + '\nLOG\nexit 0\n')
            mem.chmod(0o755)
            burn.chmod(0o755)
            proc = subprocess.run(['docker', 'run', '--rm', '--network', 'none',
                                   '-v', f'{mem}:/opt/cuda_memtest/cuda_memtest:ro',
                                   '-v', f'{burn}:/opt/gpu-burn/gpu_burn:ro',
                                   '-v', f'{results}:/results', IMAGE], capture_output=True, text=True)
            status = (results / 'status.txt').read_text().strip()
            burn_ran = (results / 'gpu-burn.log').exists()
            return proc.returncode, status, burn_ran

    def test_clean_completion(self):
        self.assertEqual(self.run_case('main thread: Program exits', 'errors: 0\nGPU 0: OK'),
                         (0, 'PASSED', True))

    def test_faulty_gpu_even_with_zero_exit(self):
        rc, status, burn_ran = self.run_case('main thread: Program exits', 'errors: 1\nGPU 0: FAULTY')
        self.assertNotEqual(rc, 0)
        self.assertEqual(status, 'FAILED')
        self.assertTrue(burn_ran)

    def test_memory_error_prevents_burn_even_with_zero_exit(self):
        rc, status, burn_ran = self.run_case('ERROR: mismatch\nmain thread: Program exits', 'GPU 0: OK')
        self.assertNotEqual(rc, 0)
        self.assertEqual(status, 'FAILED')
        self.assertFalse(burn_ran)

    def test_incomplete_memory_check_is_failure(self):
        rc, status, burn_ran = self.run_case('Test0 started', 'GPU 0: OK')
        self.assertNotEqual(rc, 0)
        self.assertEqual(status, 'FAILED')
        self.assertFalse(burn_ran)

    def test_missing_burn_summary_is_failure(self):
        rc, status, burn_ran = self.run_case('main thread: Program exits', 'errors: 0')
        self.assertNotEqual(rc, 0)
        self.assertEqual(status, 'FAILED')
        self.assertTrue(burn_ran)
