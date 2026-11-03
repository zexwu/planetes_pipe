"""extraction.py / kernels.py: sparse spectral extraction."""


import unittest

import numpy as np

from planetes_pipe.kernels import extract_sparse_spectra


class ExtractionTests(unittest.TestCase):
    def test_sparse_extractor_skips_invalid_sentinels(self) -> None:
        cube = np.ones((1, 2, 3))
        cube[0, -1, -1] = 1000.0
        ys = np.array([[0, -1]])
        xs = np.array([[0, -1]])

        batch = extract_sparse_spectra(cube, ys, xs)

        self.assertEqual(batch[0, 0, 0], 1.0)
        self.assertEqual(batch.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
