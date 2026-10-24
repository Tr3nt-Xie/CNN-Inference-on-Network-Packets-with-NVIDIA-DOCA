"""Input contract tests; do not require PyTorch, TensorRT or a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

PART5 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART5))
from frame_io import FRAME_BYTES, compare_pixels, load_frames


class FrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pixels = np.random.default_rng(542).integers(0, 256, (3, 224, 224, 3), dtype=np.uint8)
        self.original = self.root / 'original.npy'
        np.save(self.original, self.pixels)

    def load(self, path, ids=None):
        return load_frames(path, ids, expected_frames=3)

    def test_raw_matches_original_exactly(self):
        path = self.root / 'received.bin'
        path.write_bytes(self.pixels.tobytes())
        self.assertTrue(np.all(compare_pixels(self.load(self.original), self.load(path)) == 0))

    def test_shuffled_completion_order_is_aligned_by_explicit_ids(self):
        order = np.array([2, 0, 1])
        path = self.root / 'received.npz'
        np.savez(path, frames=self.pixels[order], frame_ids=order)
        self.assertTrue(np.all(compare_pixels(self.load(self.original), self.load(path)) == 0))
        raw = self.root / 'shuffled.bin'
        raw.write_bytes(self.pixels[order].tobytes())
        ids = self.root / 'ids.json'
        ids.write_text(json.dumps(order.tolist()))
        self.assertTrue(np.all(compare_pixels(self.load(self.original), self.load(raw, ids)) == 0))
        self.assertTrue(np.all(compare_pixels(self.load(self.original), self.load(raw)) > 0))

    def test_single_corrupted_byte_cannot_pass(self):
        corrupt = self.pixels.copy()
        corrupt[1, 40, 40, 2] ^= 1
        path = self.root / 'corrupt.npy'
        np.save(path, corrupt)
        self.assertEqual(compare_pixels(self.load(self.original), self.load(path)).tolist(), [0, 1, 0])

    def test_truncated_and_missing_frames_rejected(self):
        path = self.root / 'bad.bin'
        for content in (b'', self.pixels.tobytes()[:-1], self.pixels[:2].tobytes()):
            path.write_bytes(content)
            with self.assertRaises(ValueError):
                self.load(path)

    def test_header_prefixed_file_rejected(self):
        path = self.root / 'header.bin'
        path.write_bytes(b'FRAM' + self.pixels.tobytes())
        with self.assertRaisesRegex(ValueError, 'no headers'):
            self.load(path)

    def test_dtype_and_layout_are_not_silently_converted(self):
        path = self.root / 'bad.npy'
        for array in (self.pixels.astype(np.float32), self.pixels.transpose(0, 3, 1, 2)):
            np.save(path, array)
            with self.assertRaisesRegex(ValueError, 'uint8'):
                self.load(path)

    def test_duplicate_and_invalid_ids_rejected(self):
        path = self.root / 'ids.npz'
        for ids in ([0, 0, 2], [0, -1, 2], [0.0, 1.0, 2.0], [0, 1, 2**32]):
            np.savez(path, frames=self.pixels, frame_ids=np.array(ids))
            with self.assertRaises(ValueError):
                self.load(path)

    def test_missing_frame_id_rejected_even_when_count_matches(self):
        path = self.root / 'different_ids.npz'
        np.savez(path, frames=self.pixels, frame_ids=np.array([0, 1, 3]))
        with self.assertRaisesRegex(ValueError, 'missing=\\[2\\], extra=\\[3\\]'):
            compare_pixels(self.load(self.original), self.load(path))

    def test_numbered_png_directory_preserves_frame_ids(self):
        from PIL import Image
        directory = self.root / 'images'
        directory.mkdir()
        for i, frame_id in enumerate([11, 2, 8]):
            Image.fromarray(self.pixels[i]).save(directory / f'frame_{frame_id}.png')
        result = self.load(directory)
        self.assertEqual(result.ids.tolist(), [2, 8, 11])
        np.testing.assert_array_equal(result.pixels, self.pixels[[1, 2, 0]])

    def test_cli_validation_pass_failure_and_no_gpu_claim(self):
        path = self.root / 'received.bin'
        for corrupt in (False, True):
            pixels = self.pixels.copy()
            if corrupt:
                pixels[0, 0, 0, 0] ^= 1
            path.write_bytes(pixels.tobytes())
            out = self.root / str(corrupt)
            result = subprocess.run([sys.executable, str(PART5 / 'classify.py'),
                                     '--original', str(self.original), '--received', f'ordered={path}',
                                     '--expected-frames', '3', '--batch-size', '3', '--copy-batches', '1', '3',
                                     '--validate-only', '--out', str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, int(corrupt), result.stdout + result.stderr)
            self.assertIn('NOT RUN', result.stdout)
            evidence = json.loads((out / 'input_validation.json').read_text())
            self.assertFalse(evidence['gpu_classification_run'])
            self.assertFalse((out / 'results.json').exists())
            self.assertFalse((out / 'report.md').exists())


if __name__ == '__main__':
    unittest.main()
