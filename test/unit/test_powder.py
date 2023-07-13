import unittest
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from powdiffrac import Powder
from powdiffrac.simulation import peak_shapes


class TestPowder(unittest.TestCase):
    def setUp(self) -> None:
        self.calc = XRDCalculator()
        return super().setUp()

    def test_1_pymatgen(self):
        struct = Structure.from_file("data/inputs/9007064.cif")
        self.assertEqual(struct.lattice.a, 5.322)

    def test_2_calc(self):
        struct = Structure.from_file("data/inputs/9007064.cif")
        pat = self.calc.get_pattern(struct)
        self.assertAlmostEqual(pat.x[0], 29.061, 2)
        self.assertAlmostEqual(pat.y[0], 86.318, 2)

    def test_3_xrd_load(self):
        xrd = Powder.from_cif("data/inputs/9007064.cif")
        pat = self.calc.get_pattern(xrd.struct)
        self.assertEqual(pat.x[0], xrd.angles[0])

    def test_4_xrd_vary(self):
        xrd = Powder.from_cif(
            "data/inputs/9007064.cif",
            two_theta=(10, 80),
            vary_strain=True,
            vary_texture=True,
            seed=2023,
        )
        self.assertAlmostEqual(xrd.intensities[0], 79.162, 2)

    def test_5_shapes(self):
        fwhm = 0.3
        kernel_gaus = peak_shapes.get_gaussian_kernel(fwhm, step_size=0.01)
        kernel_lor = peak_shapes.get_lorentzian_kernel(fwhm, step_size=0.01)
        kernel_pv = peak_shapes.get_pseudo_voigt_kernel(fwhm, step_size=0.01, eta=0.6)
        kernel_voigt = peak_shapes.get_voigt_kernel(fwhm, step_size=0.01)
        self.assertAlmostEqual(np.median(kernel_gaus), 0.0039024, 6)
        self.assertAlmostEqual(np.median(kernel_lor), 0.0002170, 6)
        self.assertAlmostEqual(np.median(kernel_pv), 0.0001302, 6)
        self.assertAlmostEqual(np.median(kernel_voigt), 0.0214710, 6)

    def test_6_sim_signal(self):
        xrd = Powder.from_cif(
            "data/inputs/9007064.cif",
            two_theta=(10, 80),
            vary_strain=True,
            vary_texture=True,
            vary_domain=True,
            max_domain_size=50,
            peak_shape="pseudo-voigt",
            seed=2023,
        )
        compare = np.loadtxt("data/expected_outputs/simulated_signal.txt")
        sim = xrd.get_signal()
        np.testing.assert_array_almost_equal(sim, compare)
        return
