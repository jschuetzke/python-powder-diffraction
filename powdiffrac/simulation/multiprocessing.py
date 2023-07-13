import os
import numpy as np
import warnings
from typing import Tuple, List
from pymatgen.core import Structure
from powdiffrac.simulation import Powder


# function to be called by multiprocessing pool
def generate_signals_process(
    i: int,
    files: List[str],
    directory: str,
    two_theta: Tuple[float],
    step_size: float,
    strain: float,
    texture: float,
    domain_sizes: Tuple[float],
    var_strain: bool,
    var_texture: bool,
    var_domain: bool,
    peak_shape: str = "gaussian",
    domain_distribution: str = "gamma",
    n: int = 50,
) -> Tuple[np.ndarray, int]:
    """
    Multiprocessing function to generate varied signals for a structure from cif

    Args:
        i (int): index in file list (relevant for multiprocessing)
        files (List[str]): list with all cif files
        directory (str): path to cif files
        two_theta (Tuple[float]): 2Theta Min and Max
        step_size (float): step size of signals
        strain (float): maximum strain value
        texture (float): maximum texture value
        domain_sizes (Tuple[float]): Min and Max size of crystallites/grains
        var_strain (bool): vary strain for structure
        var_texture (bool): vary texture for structure
        var_domain (bool): vary domain sizes for structure
        peak_shape (str, optional): choice [gaussian, lorentzian, pseudo-voigt, voigt].
        domain_distribution (str, optional): Distribution of random domain sizes. Defaults to "gamma".
        n (int, optional): Number of varied signals. Defaults to 50.

    Returns:
        Tuple[np.ndarray, int]: _description_
    """

    filename = files[i]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        struct = Structure.from_file(os.path.join(directory, filename))
    xrd = Powder(
        struct,
        two_theta=two_theta,
        step_size=step_size,
        max_strain=strain,
        max_texture=texture,
        domain_distribution=domain_distribution,
        min_domain_size=domain_sizes[0],
        max_domain_size=domain_sizes[1],
        peak_shape=peak_shape,
        vary_strain=var_strain,
        vary_texture=var_texture,
        vary_domain=var_domain,
    )
    signals = np.zeros([n, xrd.datapoints])
    for j in range(n):
        signals[j] = xrd.get_signal(vary=True)
    return signals, i
