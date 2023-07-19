import numpy as np
import numpy.typing as npt


def scale_min_max(
    ndarray: npt.ArrayLike,
    *,
    output_max: bool = False,
) -> np.ndarray:
    """
    Scale X-ray Powder Diffraction Patterns according to their minimum and maximum count/intensity.
    Relevant to preprocess patterns as input for Neural Networks.

    Args:
        ndarray (np.typing.ArrayLike): Patterns to scale
        output_values (bool, optional): Include Defaults to 0.2.

    Returns:
        np.ndarray: _description_
    """
    x = ndarray.copy()
    if x.ndim == 1:
        min_arr = np.min(x, axis=0)
        max_arr = np.max(x, axis=0)
    else:
        min_arr = np.min(x, axis=1, keepdims=True)
        max_arr = np.max(x, axis=1, keepdims=True)
    if output_max:
        return ((x - min_arr) / (max_arr - min_arr)), np.hstack([min_arr, max_arr])
    return (x - min_arr) / (max_arr - min_arr)
