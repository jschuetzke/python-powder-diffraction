![Supported Versions](https://img.shields.io/badge/Python-3.9_|_3.10_|_3.11-blue)
![pytest status](https://github.com/jschuetzke/python-powder-diffraction/actions/workflows/python-package.yml/badge.svg)
![Version](https://img.shields.io/github/v/release/jschuetzke/python-powder-diffraction)
![License](https://img.shields.io/github/license/jschuetzke/python-powder-diffraction)

# Python Powder Diffraction Simulation Tools

Comprehensive Python package to generate powder diffraction patterns from cif files. The package builds on [ _pymatgen_](https://pymatgen.org/) and adds functionality to similate powder patterns with a defined 2θ range and step size. While pymatgen calculates theoretical peak positions and relative intensities for a given wavelength and structure, this package provides peak shape, noise and background simulation.

The basic functionality of this package was written by [njszym](https://github.com/njszym) for the [XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer) project and later extended by [jschuetzke](https://github.com/jschuetzke). The intended use is to generate many variations/perturbations of structures from a database (e.g. ICSD), so that Neural Networks can learn realistic variation in XRPD patterns. For a given structure, strain is added to manipulate the lattice, resulting in shifted peak positions. Furthermore, intensity changes due to preferred orientation (texture) of the grains the powder are simulated. Finally, the peak shapes are derived from the desired size of the crystallites in the simulated specimen.

The package is intended for generating large data sets of diffraction patterns to train neural networks with. Thus, `multiprocessing` is utilized to efficiently generate the simulated signals.

## Installation

install the package from github repo
```bash 
pip install git+https://github.com/jschuetzke/python-powder-diffraction
```
afterwards, the modules and scripts can be imported 
```python 
import powdiffrac
from powdiffrac.simulation import generate_noise
```

_alternatively_, clone this repository and install through `setup.py` 

## Usage

The main functionality of this package revolves around the `Powder` class. Construct an instance by either providing a pymatgen Structure or from a cif file:

```python 
from powdiffrac import Powder

# either
powder = Powder(pymatgen.Structure)
# or
powder = Powder.from_cif("structure.cif")
```

In order to generate variations for a given structure, the Powder class expects different arguments when calling when of the constructor methods:

```python 
powder = Powder(
  pymatgen.Structure,
  # arguments concerning the scans
  two_theta: tuple = (10,80), # 2θ range (Min, Max)
  step_size: float = 0.01, # step size/width of scans
  # arguments controlling the variations
  max_strain: float = 0.04, # maximum strain on the lattice -> default: 4%
  max_texture: float = 0.6, # maximum texture -> default 60% of grains oriented
  min_domain_size: float = 10, # minimum grain/crystallite size 10nm
  max_domain_size: float = 100, # maximum grain/crystallite size 100nm
  peak_shape: str = "gaussian" # desired peak shapes
  # arguments to switch on/off the variations
  vary_strain: bool = False,
  vary_texture: bool = False,
  vary_domain: bool = False,
)
```

The Powder instance returns a simulated pattern when calling the method `get_signal()`:

```python 
powder = Powder(...)
signal = powder.get_signal(vary=True)
```

The object contains the variations (strain, texture etc.) as properties, which are only re-rolled when intended. Thus, the method only manipulates the strain, texture and domain size if the argument `vary` is set _True_.

## Generation of large-scale data set

An example to generate a data set to train a neural network for the classification (identification) of powder diffraction signals is given in [`generate-varied-patterns`](https://github.com/jschuetzke/python-powder-diffraction/blob/main/scripts/generate-varied-patterns). Copy, modify and execute your version of this script so that it fits your use-case.
