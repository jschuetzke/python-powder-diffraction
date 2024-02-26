import numpy as np
from numpy.random import default_rng


def generate_noise(
    input_scans,
    twotheta_min=10.0,
    twotheta_max=80.0,
    seed=None,
    air_scattering="skip",
    background_ratio=0.1,
    noise_lvl=None,
    noise_min=0.005,
    noise_max=0.015,
    poisson=True,
):
    """
    function to simulate noise for a given (batch of) simulated xrd signal(s)
    as described in https://doi.org/10.1107/S2052252521002402

    Parameters
    ----------
    input_scans : 1D or 2D numpy array
        Input array to add noise to.
        Either 1D or 2D (in format [n_scans, datapoints])
    twotheta_min : float, optional
        Start 2Theta angle of scan. The default is 10.0.
    twotheta_max : TYPE, optional
        End 2Theta angle of scan. The default is 80.0.
    seed : int, optional
        input random seed for debugging purposes. The default is None.
    airscatt_val : string, optional
        One over x value for air scattering. "random" to include airscattering.
        The default is "skip".
    background_ratio : float, optional
        Amplitude of background in relation to maximum intensity in signal.
        The default is 0.1 (max 10% of highest intensity).
    noise_lvl : float, optional
        Amplitude of noise in relation to maximum intensity in signal.
        The default is None (randomized).
    noise_min : float, optional
        Minimum amplitude of noise in relation to maximum intensity in signal
        if drawn randomly. The default is .005.
    noise_max : float, optional
        Maximum amplitude of noise in relation to maximum intensity in signal
        if drawn randomly. The default is .015.
    poisson : bool, optional
        Include Poisson noise that is dependent on signal intensity. The default is True.

    Returns
    -------
    noisy_scans : 1D or 2D numpy array
        returns signals with added noise. 1D or 2D output depending on input

    """
    dim = input_scans.ndim
    # copy to avoid pointer confusion
    if dim == 1:  # [datapoints,]
        scans = np.expand_dims(input_scans, 0).copy()
    elif dim == 2:  # [1,datapoints]
        scans = input_scans.copy()
    elif dim == 3:  # [scans,datapoints,1] but should not be the case
        scans = input_scans[:, :, 0].copy()
    else:
        raise ValueError("Dimensions unknown, check input scan!")

    datapoints = scans.shape[1]
    if seed is None:
        rng = default_rng()
    else:
        rng = default_rng(int(seed))
    # we split the artificial noise generation into three parts
    # air scattering, background (chebyshev-polynomial)
    # and gaussian noise

    # air scattering part
    steps = np.linspace(twotheta_min, twotheta_max, datapoints)

    if air_scattering == "random":
        # observation: air scattering begins between 0.02 and 0.035 of scan max
        airscatt_min = 0.1
        airscatt_max = 1.5

        airscatt_val = rng.uniform(
            airscatt_min, airscatt_max, (scans.shape[0])
        ) * np.max(scans, axis=1)
        onx = airscatt_val[:, None] / (steps)
    else:
        onx = np.zeros_like(scans)

    # background function using a chebyshev polynomial
        
    if (background_ratio == 0) or (background_ratio is None):
        cheb = np.zeros_like(scans)
    else:
        coef_max = background_ratio
        coef_min = -background_ratio
        cheb = np.zeros_like(scans)
        for i in range(cheb.shape[0]):
            polynom_order = rng.integers(2, 5)
            ccoefs = rng.uniform(
                coef_min / polynom_order, coef_max / polynom_order, (polynom_order)
            )
            c = np.polynomial.chebyshev.Chebyshev(ccoefs)
            cheb[i, :] = c.linspace(datapoints)[1]
        # correction to avoid negative background
        negative = np.min(cheb, axis=1) < 0
        cheb = cheb + (np.abs(np.min(cheb, axis=1)) * negative)[:, None]
        # scale according to scans
        cheb = cheb * np.max(scans, axis=1)[:, None]
    bkg = np.add(onx, cheb)

    # gaussian noise -> dependent on signal
    # we want to avoid extreme outliers so we clip the noise
    # with 1/3 and -3/+3 we achieve a nice normal distribution for values -1 to 1
    gaus = 1 / 3 * np.clip(rng.normal(0, 1, scans.shape), -3, 3)
    # next we shift the noise from -1 and 1 to 0 and 1
    gaus = (gaus * 0.5) + 0.5

    # same for Poisson noise
    if poisson:
        pois = 1 / 3 * np.clip(rng.normal(0, 1, scans.shape), -3, 3)
        pois = (pois * 0.5) + 0.5

    # scale noise according to max intensity
    if type(noise_lvl) is str or noise_lvl is None:  # assume some form of random
        noise_lvl = rng.uniform(noise_min, noise_max, scans.shape[0])
    elif type(noise_lvl) is int:
        noise_lvl = np.repeat(np.array([noise_lvl]), scans.shape[0], axis=0) 
    gaus = gaus * (noise_lvl * np.max(scans, axis=1))[:, None]
    noisy_scan = np.add(scans, gaus)
    
    if poisson:
        pois = np.sqrt(scans) * noise_lvl[:, None] * pois
        noisy_scan += pois

    # add background
    noisy_scan += bkg
    if dim == 1:
        noisy_scan = noisy_scan[0, :]
    return noisy_scan
