import numpy as np
from powdiffrac.simulation import generate_noise


def test_noise():
    signal = np.loadtxt("data/expected_outputs/simulated_signal.txt")
    noise = generate_noise(
        signal,
        twotheta_min=10,
        twotheta_max=80,
        seed=2023,
        noise_min=0.03,
        noise_max=0.07,
    )
    compare = np.loadtxt("data/expected_outputs/signal_noise.txt")
    np.testing.assert_array_equal(noise, compare)
    return


if __name__ == "__main__":
    test_noise()
