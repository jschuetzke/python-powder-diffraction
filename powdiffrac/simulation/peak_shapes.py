import numpy as np
from scipy.ndimage._filters import _gaussian_kernel1d
from scipy.special import voigt_profile


def get_lorentzian_kernel(fwhm: float, step_size: float) -> np.ndarray:
    """
    Calculates a Lorentzian kernel for convolving peak shapes

    Args:
        fwhm (float): full width half maximum of peak shape IN DEGREES (2-THETA)
        step_size (float): step size of scan

    Returns:
        np.ndarray: kernel
    """
    # get radius
    sd = fwhm * 20 // step_size
    # assert uneven kernel length (maximum in mid)
    if not sd % 2:
        sd += 1
    t = (np.arange(sd) - int(sd / 2)) * step_size
    kernel = 1 / (1 + 4 * (t / fwhm) ** 2)
    return kernel / np.sum(kernel)


def get_gaussian_kernel(fwhm: float, step_size: float) -> np.ndarray:
    """
    Calculates a Gaussian kernel for convolving peak shapes

    Args:
        fwhm (float): full width half maximum of peak shape IN DEGREES (2-THETA)
        step_size (float): step size of scan

    Returns:
        np.ndarray: kernel
    """
    # Convert FWHM to std deviation of gaussian
    sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * fwhm / step_size

    sd = float(sigma)
    truncate = 4.0
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    return _gaussian_kernel1d(sigma, 0, lw)


def get_pseudo_voigt_kernel(
    fwhm: float, step_size: float, eta: float = 0.5
) -> np.ndarray:
    """
    Calculates a Pseudo Voigt kernel for convolving peak shapes

    Args:
        fwhm (float): full width half maximum of peak shape IN DEGREES (2-THETA)
        step_size (float): step size of scan
        eta (float): mixing factor of Lorentzian and Gaussian kernel

    Returns:
        np.ndarray: kernel
    """
    k_l = get_lorentzian_kernel(fwhm, step_size)
    k_g = get_gaussian_kernel(fwhm, step_size)
    # pad Gaussian kernel for matching length
    k_g_pad = np.pad(k_g, (k_l.size - k_g.size) // 2)
    return eta * k_l + (1 - eta) * k_g_pad


def get_voigt_kernel(fwhm: float, step_size: float) -> np.ndarray:
    """
    Calculates a Voigt kernel for convolving peak shapes

    Args:
        fwhm (float): full width half maximum of peak shape IN DEGREES (2-THETA)
        step_size (float): step size of scan

    Returns:
        np.ndarray: kernel
    """
    # get radius
    sd = fwhm * 20 // step_size
    # assert uneven kernel length (maximum in mid)
    if not sd % 2:
        sd += 1
    t = (np.arange(sd) - int(sd / 2)) * step_size
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm / 2
    return voigt_profile(t, sigma, gamma)
