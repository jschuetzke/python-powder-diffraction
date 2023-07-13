import os
import argparse
import ast
import numpy as np
from powdiffrac.simulation import generate_signals_process
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm


# basic function to use tuples as CLI arguments and convert to tuple args in Python
# proposed by ChatGPT, no copyright infrigement intended
def tuple_type(arg_string):
    try:
        # Use ast.literal_eval to safely parse the string as a tuple
        value = ast.literal_eval(arg_string)
        if not isinstance(value, tuple):
            raise argparse.ArgumentTypeError("Value must be a tuple")
        assert len(value) == 2
        assert value[0] < value[1]
        return value
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid tuple format")
    except AssertionError:
        raise argparse.ArgumentTypeError("Provide only two values in form (Min,Max)")


def simulate_signals(
    directory: str,
    # arguments for XRDStructure
    two_theta: tuple,
    step_size: float,
    strain: float,
    texture: float,
    domain_sizes: tuple,
    peak_shape: str,
    var_strain: bool,
    var_texture: bool,
    var_domain: bool,
    # arguments for script
    n_train: int,
    n_val: int,
):
    files = sorted([f for f in os.listdir(directory) if f.endswith("cif")])
    datapoints = int(((two_theta[1] - two_theta[0]) / step_size) + 1)

    def generate_save_signals(n, name=""):
        x = np.zeros([len(files), n, datapoints])
        y = np.repeat(np.arange(len(files)), n)

        generate_signals_wrapper = partial(
            generate_signals_process,
            files=files,
            directory=directory,
            two_theta=two_theta,
            step_size=step_size,
            strain=strain,
            texture=texture,
            domain_sizes=domain_sizes,
            peak_shape=peak_shape,
            var_strain=var_strain,
            var_texture=var_texture,
            var_domain=var_domain,
            n=n,
        )

        with Pool() as pool:
            for result in tqdm(
                pool.imap_unordered(generate_signals_wrapper, list(range(len(files)))),
                total=len(files),
            ):
                x[result[1]] = result[0]
        np.save(os.path.join(directory, f"x_{name}.npy"), x)
        np.save(os.path.join(directory, f"y_{name}.npy"), y)
        # overwrite to unlock space
        x = None
        y = None
        return

    generate_save_signals(n_train, name="train")
    generate_save_signals(n_val, name="val")
    return


def main():
    parser = argparse.ArgumentParser(description="generate xrd signals")
    parser.add_argument(
        "directory", type=str, nargs="?", help="full path to cif folder"
    )
    parser.add_argument(
        "--theta_range",
        type=tuple_type,
        default="(10,80)",
        dest="two_theta",
        help="2Theta range for all scans",
    )
    parser.add_argument(
        "--step_scan",
        type=float,
        default=0.01,
        dest="step_size",
        help="step size  (Delta2Theta) for all scans",
    )
    parser.add_argument(
        "--strain", type=float, default=0.01, help="maximum amount of strain on lattice"
    )
    parser.add_argument(
        "--texture",
        type=float,
        default=0.4,
        help="maximum amount of texture in the pattern",
    )
    parser.add_argument(
        "--domain_sizes",
        type=tuple_type,
        default="(10,100)",
        dest="domain_sizes",
        help="Min and Max Domain Size for all scans",
    )
    parser.add_argument(
        "--peak_shape",
        type=str,
        default="gaussian",
        help="shape of peaks in simulated signal",
    )
    parser.add_argument(
        "--no_var_strain",
        action="store_false",
        dest="var_strain",
        help="varies peak positions by default",
    )
    parser.add_argument(
        "--no_var_domain",
        action="store_false",
        dest="var_domain",
        help="varies peak shapes by default",
    )
    parser.add_argument(
        "--no_var_texture",
        action="store_false",
        dest="var_texture",
        help="varies peak intensities by default",
    )

    parser.add_argument(
        "--n_train",
        type=int,
        default=50,
        help="number of variations per phase for training data",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=10,
        help="number of variations per phase for training data",
    )

    args = parser.parse_args()
    simulate_signals(**vars(args))
    return


if __name__ == "__main__":
    main()
