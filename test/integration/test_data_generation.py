import numpy as np
from powdiffrac.simulation import generate_signals_process, generate_noise
from powdiffrac.processing import scale_min_max
from functools import partial
from multiprocessing import Pool


def test_simulation_pipeline():
    directory = "data/inputs/"
    two_theta = (10, 80)
    step_size = 0.01

    files = ["9007064.cif"]
    datapoints = int(((two_theta[1] - two_theta[0]) / step_size) + 1)

    generate_signals_wrapper = partial(
        generate_signals_process,
        files=files,
        directory=directory,
        two_theta=(10, 80),
        step_size=0.01,
        strain=0.04,
        texture=0.6,
        domain_sizes=(10, 100),
        peak_shape="pseudo-voigt",
        var_strain=True,
        var_texture=True,
        var_domain=True,
        n=10,
    )

    x = np.zeros([10, datapoints])

    with Pool() as pool:
        for result in pool.imap_unordered(
            generate_signals_wrapper, list(range(len(files)))
        ):
            x = result[0]

    noise = generate_noise(x)
    assert np.min(noise) > 0.0
    min_arr = np.min(noise[0])
    max_arr = np.max(noise[0])
    np.testing.assert_array_equal(
        scale_min_max(noise)[0], (noise[0] - min_arr) / (max_arr - min_arr)
    )

    return
