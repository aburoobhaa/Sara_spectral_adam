from setuptools import setup, find_packages

setup(
    name="spectral_adam",
    version="0.1.0",
    description="Spectral Adam Optimizer for PyTorch",
    author="Saranyan M",
    url="https://github.com/saranyan18/spectral_adam",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy",
    ],
    python_requires=">=3.7",
)
