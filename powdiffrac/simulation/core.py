import numpy as np
from numpy.random import default_rng
from scipy.ndimage import convolve1d
import warnings
from typing import Type

from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.diffraction.core import AbstractDiffractionPatternCalculator

import powdiffrac.simulation.peak_shapes as peaks


class Powder(object):
    """
    Class for simulation of diffraction scans from pymatgen Structures
    """

    def __init__(
        self,
        struct: Type[Structure],
        two_theta: tuple = (10, 80),
        step_size: float = 0.01,
        split_datapoints_intensities: bool = False,
        max_strain: float = 0.04,
        min_domain_size: float = 10,
        max_domain_size: float = 100,
        max_texture: float = 0.6,
        domain_distribution: str = "uniform",
        peak_shape: str = "gaussian",
        vary_strain: bool = False,
        vary_texture: bool = False,
        vary_domain: bool = False,
        seed: int = None,
    ):
        """
        Args:
            struct (pymatgen Structure instance): structure baseline
            two_theta (tuple): tuple of two_theta values for min and max
            step_size (float): step size of measurement
            split_datapoints_intensities (bool): if False then simulated angles
                are rounded to the closest datapoint. if True then simulated
                intensities are split proportianally on neighboring datapoints
            max_strain (float): maximum allowed change in the magnitude
                of the strain tensor components
            min_domain_size (float): smallest domain size (in nm) to be sampled,
                leading to the broadest peaks
            max_domain_size (float): largest domain size (in nm) to be sampled,
                leading to the most narrow peaks
            max_texture (float): maximum strength of texture applied.
                For example, max_texture=0.6 implies peaks will be
                scaled by as much as +/- 60% of their original
                intensities.
            domain_distribution (string, choice): either uniform or gamma
            peak_shape (string): peak shape to convolve, defaults to gaussian
            vary_strain (bool): include varied lattices
            vary_texture (bool): include varied preferred orientations
            vary_domain (bool): include varied crystallite sizes
            seed (int): Optional, seed for numpy random generator
        """
        if isinstance(struct, Structure):
            self.struct = struct
        else:
            raise ValueError("Expecting pymatgen Structure, received something else")
        self.two_theta = two_theta
        self.step_size = step_size
        self.split_intensities = split_datapoints_intensities
        self.vary_strain = vary_strain
        self.vary_texture = vary_texture
        self.vary_domain = vary_domain

        # defaults
        self.max_strain = max_strain
        self.min_domain_size = min_domain_size
        self.max_domain_size = max_domain_size
        self.max_texture = max_texture
        self.default_domain_size = 100
        self.domain_distribution = domain_distribution
        self._domain_size = self.default_domain_size
        # Default radiation is X-rays with CuKa wavelenght
        self.radiation = "CuKa"
        self._calculator = XRDCalculator(self.radiation)
        self._scaling_factors = [1.0]
        self.peak_shape = peak_shape
        self._rng = default_rng(seed)
        # ensure accessing properties does not fail after init
        if vary_strain or vary_texture:
            self.roll_variances()

    @classmethod
    def from_cif(
        cls,
        filename: str,
        two_theta: tuple = (10, 80),
        step_size: float = 0.01,
        split_datapoints_intensities: bool = False,
        max_strain: float = 0.04,
        min_domain_size: float = 10,
        max_domain_size: float = 100,
        max_texture: float = 0.6,
        domain_distribution: str = "uniform",
        peak_shape: str = "gaussian",
        vary_strain: bool = False,
        vary_texture: bool = False,
        vary_domain: bool = False,
        seed: int = None,
    ):
        """
        Generate Powder instance from cif file
        Args:
            struct (pymatgen Structure instance): structure baseline
            two_theta (tuple): tuple of two_theta values for min and max
            step_size (float): step size of measurement
            split_datapoints_intensities (bool): if False then simulated angles
                are rounded to the closest datapoint. if True then simulated
                intensities are split proportianally on neighboring datapoints
            max_strain (float): maximum allowed change in the magnitude
                of the strain tensor components
            min_domain_size (float): smallest domain size (in nm) to be sampled,
                leading to the broadest peaks
            max_domain_size (float): largest domain size (in nm) to be sampled,
                leading to the most narrow peaks
            max_texture (float): maximum strength of texture applied.
                For example, max_texture=0.6 implies peaks will be
                scaled by as much as +/- 60% of their original
                intensities.
            domain_distribution (string, choice): either uniform or gamma
            peak_shape (string): peak shape to convolve, defaults to gaussian
            vary_strain (bool): include varied lattices
            vary_texture (bool): include varied preferred orientations
            vary_domain (bool): include varied crystallite sizes
            seed (int): Optional, seed for numpy random generator
        """
        arguments = locals()
        arguments.pop("filename", None)
        arguments.pop("cls", None)
        struct = Structure.from_file(filename)
        return cls(struct, **arguments)

    # Diffraction Calculator to transform Structure to peaks
    @property
    def calculator(self):
        return self._calculator

    def calc_checker(self, diffrac_calc):
        if type(diffrac_calc) is list:
            for calc in diffrac_calc:
                self.calc_checker(calc)
        else:
            if not isinstance(diffrac_calc, AbstractDiffractionPatternCalculator):
                raise ValueError(
                    "Input not Instance of pymatgen Diffraction Calculator!"
                )
        return

    @calculator.setter
    def calculator(self, diffraction_calculator):
        self.calc_checker(diffraction_calculator)
        self._calculator = diffraction_calculator

    @property
    def scaling_factors(self):
        return self._scaling_factors

    @scaling_factors.setter
    def scaling_factors(self, factors):
        if type(self.calculator) is list:
            assert len(factors) == len(self.calculator)
            self._scaling_factors = factors

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, seed):
        self._rng = default_rng(seed)

    # Do we need this?
    @property
    def conv_struct(self):
        sga = SpacegroupAnalyzer(self.struct)
        return sga.get_conventional_standard_structure()

    # space group of Structure
    @property
    def sg(self):
        return self.struct.get_space_group_info()[1]

    @property
    def sg_class(self):
        sg = self.sg
        if sg in list(range(195, 231)):
            return "cubic"
        elif sg in list(range(16, 76)):
            return "orthorhombic"
        elif sg in list(range(3, 16)):
            return "monoclinic"
        elif sg in list(range(1, 3)):
            return "triclinic"
        elif sg in list(range(76, 195)):
            if sg in list(range(75, 83)) + list(range(143, 149)) + list(
                range(168, 175)
            ):
                return "low-sym hexagonal/tetragonal"
            else:
                return "high-sym hexagonal/tetragonal"

    # Lattice and related matrix
    @property
    def lattice(self):
        return self.struct.lattice

    @property
    def matrix(self):
        return self.struct.lattice.matrix

    # vary strain tensor to modify lattice
    @property
    def strain_tensor(self):
        return self._strain_tensor

    @strain_tensor.setter
    def strain_tensor(self, tensor):
        warnings.warn("Not intended to use directly! Only for debugging purposes!")
        self._strain_tensor = tensor

    # set new randomized strain tensor
    def vary_strain_tensor(self):
        max_strain = self.max_strain
        s11, s22, s33 = self.rng.uniform(1 - max_strain, 1 + max_strain, 3)
        s12, s13, s21, s23, s31, s32 = self.rng.uniform(
            0 - max_strain, 0 + max_strain, 6
        )

        sg_class = self.sg_class

        if sg_class in [
            "cubic",
            "orthorhombic",
            "monoclinic",
            "high-sym hexagonal/tetragonal",
        ]:
            v1 = [s11, 0, 0]
        elif sg_class == "low-sym hexagonal/tetragonal":
            v1 = [s11, s12, 0]
        elif sg_class == "triclinic":
            v1 = [s11, s12, s13]

        if sg_class in ["cubic", "high-sym hexagonal/tetragonal"]:
            v2 = [0, s11, 0]
        elif sg_class == "orthorhombic":
            v2 = [0, s22, 0]
        elif sg_class == "monoclinic":
            v2 = [0, s22, s23]
        elif sg_class == "low-sym hexagonal/tetragonal":
            v2 = [-s12, s22, 0]
        elif sg_class == "triclinic":
            v2 = [s21, s22, s23]

        if sg_class == "cubic":
            v3 = [0, 0, s11]
        elif sg_class == "high-sym hexagonal/tetragonal":
            v3 = [0, 0, s33]
        elif sg_class == "orthorhombic":
            v3 = [0, 0, s33]
        elif sg_class == "monoclinic":
            v3 = [0, s23, s33]
        elif sg_class == "low-sym hexagonal/tetragonal":
            v3 = [0, 0, s33]
        elif sg_class == "triclinic":
            v3 = [s31, s32, s33]
        self._strain_tensor = np.array([v1, v2, v3])

    # use calculated strain_tensor to modify lattice
    @property
    def strained_matrix(self):
        return np.matmul(self.matrix, self.strain_tensor)

    @property
    def strained_lattice(self):
        return Lattice(self.strained_matrix)

    # new structure with modified lattice
    @property
    def strained_struct(self):
        new_struct = self.struct.copy()
        new_struct.lattice = self.strained_lattice
        return new_struct

    # use original or strained lattice, depending on vary_strain
    @property
    def pattern(self):
        if self.vary_strain:
            struct = self.strained_struct
        else:
            struct = self.struct
        if type(self.calculator) is list:
            pat = self.calculator[0].get_pattern(struct, two_theta_range=self.two_theta)
            for i, calc in enumerate(self.calculator[1:]):
                temp = calc.get_pattern(struct, two_theta_range=self.two_theta)
                pat.x = np.hstack([pat.x, temp.x])
                pat.y = np.hstack([pat.y, temp.y * self.scaling_factors[i + 1]])
        else:
            pat = self.calculator.get_pattern(struct, two_theta_range=self.two_theta)
        return pat

    @property
    def angles(self):
        return self.pattern.x

    @property
    def hkl_list(self):
        return [np.array(v[0]["hkl"]) for v in self.pattern.hkls]

    # change preferred orientation values to modify intensities
    @property
    def preferred_orientation(self):
        return self._preferred_orientation

    @preferred_orientation.setter
    def preferred_orientation(self, po_array):
        warnings.warn("Not intended to use directly! Only for debugging purposes!")
        self._preferred_orientation = po_array

    # set new randomized preferred orientation plane
    def vary_preferred_orientation(self):
        is_hex = len(self.hkl_list[0]) == 4
        # Four Miller indicies in hexagonal systems
        # Strangely the is_hexagonal function sometimes reports
        # hex. structures but we only get 3dim hkl planes
        # from the Diffraction Calculator and vice versa
        # if self.struct.lattice.is_hexagonal() == True:
        if is_hex:
            check = 0.0
            while check == 0.0:
                preferred_orientation = self.rng.choice(2, 4)
                check = np.dot(
                    preferred_orientation, preferred_orientation
                )  # Make sure we don't have 0-vector

        # Three indicies are used otherwise
        else:
            check = 0.0
            while check == 0.0:
                preferred_orientation = self.rng.choice(2, 3)
                check = np.dot(preferred_orientation, preferred_orientation)
        self._preferred_orientation = preferred_orientation

    def map_interval(self, v):
        """
        Maps a value (v) from the interval [0, 1] to
            a new interval [1 - max_texture, 1]
        """

        bound = 1.0 - self.max_texture
        return bound + (((1.0 - bound) / (1.0 - 0.0)) * (v - 0.0))

    # calculate modified intensities depending on preferred orientation
    @property
    def textured_intensities(self):
        preferred_orientation = self._preferred_orientation

        texture_factors = []
        for hkl in self.hkl_list:
            norm_1 = np.sqrt(np.dot(hkl, hkl))
            norm_2 = np.sqrt(np.dot(preferred_orientation, preferred_orientation))
            total_norm = norm_1 * norm_2
            factor = np.abs(np.dot(hkl, preferred_orientation) / total_norm)
            factor = self.map_interval(factor)
            texture_factors.append(factor)

        if type(self.calculator) is list:
            texture_factors = texture_factors * len(self.calculator)

        scaled_intensities = self.pattern.y * np.array(texture_factors)

        return scaled_intensities

    @property
    def intensities(self):
        if self.vary_texture:
            return self.textured_intensities
        else:
            return self.pattern.y

    # vary domain size to modify peak broadening
    @property
    def domain_size(self):
        return self._domain_size

    @domain_size.setter
    def domain_size(self, size):
        warnings.warn("Not intended to use directly! Only for debugging purposes!")
        self._domain_size = size

    # set new randomized domain size
    def vary_domain_size(self):
        if self.domain_distribution == "uniform":
            self._domain_size = self.rng.uniform(
                self.min_domain_size, self.max_domain_size
            )
        elif self.domain_distribution == "gamma":
            temp = self.rng.gamma(1.25, 10)
            self._domain_size = min(temp + self.min_domain_size, self.max_domain_size)
        else:
            warnings.warn(
                f"domain size distribution {self.domain_distribution} not recognized!"
            )

    # set new variances
    def roll_variances(self):
        if self.vary_strain:
            self.vary_strain_tensor()
        if self.vary_texture:
            self.vary_preferred_orientation()
        if self.vary_domain:
            self.vary_domain_size()
        return

    # output current values for lattice, intensities and domain size
    def output_variables(self):
        output_list = []
        if self.vary_strain:
            output_list.append(self.strained_lattice)
        else:
            output_list.append(self.lattice)
        if self.vary_texture:
            output_list.append(self.preferred_orientation)
            output_list.append(self.textured_intensities)
        else:
            output_list.append(np.array([None, None, None]))
            output_list.append(self.intensities)
        output_list.append(self.domain_size)
        return output_list

    # print current variable values
    def print_variables(self):
        print(self.output_variables())

    def calc_fwhm(self, two_theta, tau, wavelength):
        """
        calculate full width half maximum based on angle (two theta) and domain size (tau)
        Args:
            two_theta: angle in two theta space
            tau: domain size in nm
        Returns:
            fwhm for gaussian kernel
        """
        # Calculate FWHM based on the Scherrer equation
        K = 0.9  # shape factor
        wavelength = wavelength * 0.1  # angstrom to nm
        theta = np.radians(two_theta / 2.0)  # Bragg angle in radians
        fwhm = (K * wavelength) / (np.cos(theta) * tau)  # in radians
        return np.degrees(fwhm)  # return value in degrees

    @property
    def datapoints(self) -> int:
        return int(((self.two_theta[1] - self.two_theta[0]) / self.step_size) + 1)

    @property
    def steps(self) -> np.ndarray:
        return np.linspace(self.two_theta[0], self.two_theta[1], self.datapoints)

    def get_signal(
        self, vary: bool = False, output_variables: bool = False, scaling: float = None
    ) -> np.ndarray:
        """
        get signal for current structure and variances
        Args:
            vary (bool): roll new values for variances before calculations
            output_variables (bool): output lattice, intensities and domain size
            scaling (int): if not None, maximum value of simulated signal is set to (scaling)
        Returns:
            signal (1D np.array): calculated signal
            output_variables (list, optional): lattice, intensities and domain size
        """
        if vary:
            self.roll_variances()
        angles, intensities = self.angles, self.intensities

        signals = np.zeros([len(angles), self.steps.shape[0]])
        for i, ang in enumerate(angles):
            if not self.split_intensities:
                # map angle to closest datapoint step
                idx = np.argmin(np.abs(ang - self.steps))
                signals[i, idx] = intensities[i]
            else:
                # get the datapoints closest to the simulated angles
                diffs = np.argsort(np.abs(ang - self.steps))
                # angles in steps increasing so lower datapoint has lower index
                lower, upper = sorted(diffs[:2])
                # calculate ratio to split intensity
                ratio = (self.steps[upper] - ang) / self.step_size
                # ratio is "lever" -> apply inversely
                signals[i, lower] = ratio * intensities[i]
                signals[i, upper] = (1 - ratio) * intensities[i]
        # calculate respective wavelengths per peak
        distinct_wavelengths = (
            len(self.calculator) if type(self.calculator) == list else 1
        )
        # scenario: peak of higher wavelength cut, so we round up values
        # if uneven value encountered
        distinct_peaks = np.ceil((len(angles) / distinct_wavelengths))
        if type(self.calculator) == list:
            wavelengths = np.array([calc.wavelength for calc in self.calculator])
        else:
            wavelengths = np.array([self.calculator.wavelength])
        wavelengths = np.repeat(wavelengths, distinct_peaks)

        if self.domain_size != 0:
            domain_size = self.domain_size
            # convolve every row with unique kernel
            # iterate over rows; not vectorizable, changing kernel for every row
            for i in range(signals.shape[0]):
                row = signals[i, :]
                ang = angles[i]
                fwhm = self.calc_fwhm(ang, domain_size, wavelengths[i])
                if self.peak_shape == "lorentzian" or self.peak_shape == "cauchy":
                    kernel = peaks.get_lorentzian_kernel(fwhm, self.step_size)
                elif (
                    self.peak_shape == "pseudo_voigt"
                    or self.peak_shape == "pseudo-voigt"
                ):
                    eta = self.rng.random() if vary else 0.5
                    kernel = peaks.get_pseudo_voigt_kernel(fwhm, self.step_size, eta)
                elif self.peak_shape == "voigt":
                    kernel = peaks.get_voigt_kernel(fwhm, self.step_size)
                else:
                    kernel = peaks.get_gaussian_kernel(fwhm, self.step_size)
                signals[i] = convolve1d(row, kernel, mode="constant")

        # combine signals
        signal = np.sum(signals, axis=0)
        if scaling is not None:
            signal = signal / np.max(signal) * scaling
        if output_variables:
            return signal, self.output_variables()
        else:
            return signal
