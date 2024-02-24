from setuptools import setup, find_packages

# Install dependencies
setup(
    name="powder-diffraction",
    version="1.0.2",
    author="Jan Schuetzke, Nathan J. Szymanski",
    author_email="jan.schuetzke@kit.edu; nathan_szymanski@berkeley.edu",
    description="A package for the generation of synthetic powder diffraction scans from pymatgen Structures",
    url="https://github.com/jschuetzke/python-powder-diffraction",
    packages=find_packages(),
    python_requires=">=3.9",
    scripts=["scripts/generate_training_data.py"],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "scipy", "pymatgen", "tqdm"],
)
