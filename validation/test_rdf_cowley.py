"""Observable numerical invariants for the imported user-wide analysis helper."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "skills/lammps/scripts/rdf_cowley.py"
SPEC = importlib.util.spec_from_file_location("rdf_cowley", MODULE)
rdf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rdf)


class AnalysisChecks(unittest.TestCase):
    def test_original_and_added_self_checks(self):
        rdf._self_check()

    def test_directed_counts_reciprocity_and_local_weighted_identity(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        positions = np.array([0., .5, 1.1, 1.8, 2.2, 2.8])
        neighbors = abs(positions[:, None] - positions[None, :]) < 1.01
        np.fill_diagonal(neighbors, False)
        local = np.stack([neighbors[:, labels == b].sum(1) for b in range(3)], axis=1)
        counts = np.bincount(labels)
        coordination = np.stack([local[labels == a].mean(0) for a in range(3)])
        pair_totals = counts[:, None] * coordination
        np.testing.assert_allclose(pair_totals, pair_totals.T)
        degrees = local.sum(1)
        for finite in (False, True):
            baseline = ((counts[None, :] - np.eye(3)) / (counts.sum() - 1)
                        if finite else np.broadcast_to(counts / counts.sum(), (3, 3)))
            alpha = rdf.cowley(coordination, counts, finite)
            np.testing.assert_allclose((alpha * baseline).sum(1), 0, atol=1e-14)
            for a in range(3):
                selected = labels == a
                local_alpha = 1 - local[selected] / degrees[selected, None] / baseline[a]
                np.testing.assert_allclose(
                    np.average(local_alpha, axis=0, weights=degrees[selected]), alpha[a], atol=1e-14)

    def test_batched_cowley_and_missing_present_channel(self):
        z = np.array([[[2., np.nan], [1., 3.]], [[2., 2.], [2., 2.]]])
        actual = rdf.cowley(z, [5, 5])
        self.assertTrue(np.isnan(actual[0, 0]).all())
        np.testing.assert_allclose(actual[0, 1], [.5, -.5])
        np.testing.assert_allclose(actual[1], 0)
        self.assertTrue(np.isnan(z[0, 0, 1]))

    def test_absent_species_exact_zero_without_mutation(self):
        z = np.array([[6., np.nan], [np.nan, np.nan]])
        actual = rdf.cowley(z, [100, 0], True)
        self.assertEqual(actual[0, 0], 0)
        self.assertTrue(np.isnan(actual[1]).all())
        self.assertTrue(np.isnan(actual[:, 1]).all())
        self.assertTrue(np.isnan(z[0, 1]))
        with self.assertRaises(ValueError):
            rdf.cowley([[6., 1.], [0., 0.]], [100, 0])

    def test_cumulative_edges_and_vector_channels(self):
        edges = np.array([.5, 1., 2.])
        cumulative = np.array([[1., 3., 7.], [2., 6., 14.]])
        for radius, expected in [(0, [0, 0]), (.25, [.5, 1]),
                                 (1, [3, 6]), (1.5, [5, 10]), (2, [7, 14])]:
            np.testing.assert_allclose(rdf.cumulative_at_radius(edges, cumulative, radius), expected)
        for bad in ([1., np.nan, 3.], [1., np.inf, 3.], [-1., 2., 3.]):
            with self.assertRaises(ValueError):
                rdf.cumulative_at_radius(edges, bad, 1.5)
        for bad_radius in (-.01, 2.01, np.nan):
            with self.assertRaises(ValueError):
                rdf.cumulative_at_radius(edges, cumulative, bad_radius)

    def test_polynomial_minimum_on_nonuniform_grid_and_invalid_bracket(self):
        r = np.linspace(2.5, 4.5, 301) ** 1.01
        true = 3.731
        g = .4 + .8 * (r - true) ** 2 + .2 * (r - true) ** 4
        result = rdf.continuous_minimum(r, g, (3.4, 4.1), half_window=.2)
        self.assertAlmostEqual(result['radius'], true, places=10)
        self.assertGreater(result['curvature'], 0)
        for kwargs in ({'half_window': 0}, {'degree': 1}):
            with self.assertRaises(ValueError):
                rdf.continuous_minimum(r, g, (3.4, 4.1), **kwargs)
        with self.assertRaises(ValueError):
            rdf.continuous_minimum(r, r, (3.4, 4.1))

    def test_regularization_linear_nullspace_and_penalty_order(self):
        x = np.array([0., .1, .3, .7, 1.])
        truth = 3.5 + .2 * x
        radii = truth + np.array([-.02, 0, .02])[:, None]
        a = rdf.regularize_minima(x, radii, penalties=[0, .1, 1., 10.])
        b = rdf.regularize_minima(x, radii, penalties=[10., 0, 1., .1])
        np.testing.assert_allclose(a['radius'], truth, atol=1e-10)
        np.testing.assert_allclose(a['map'] @ radii.mean(0), a['radius'])
        np.testing.assert_allclose(a['radius'], b['radius'])
        self.assertEqual(a['lambda'], b['lambda'])
        with self.assertRaises(ValueError):
            rdf.regularize_minima(x, radii, penalties=[-1])

    def test_affine_background_batched_sideband_fit(self):
        x = np.arange(5.)
        y = np.stack([2 + 3*x, 5 - 2*x])
        y[:, 2] += [2, 4]
        fit, residual, coefficients = rdf.affine_background(x, y, [True, True, False, True, True])
        np.testing.assert_allclose(coefficients, [[2, 3], [5, -2]], atol=1e-13)
        np.testing.assert_allclose(residual[:, 2], [2, 4], atol=1e-13)
        np.testing.assert_allclose(fit + residual, y)
        with self.assertRaises(ValueError):
            rdf.affine_background([1, 1, 1], [2, 2, 2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
