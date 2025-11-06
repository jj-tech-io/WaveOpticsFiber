from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bem_solver",
    version="1.0.0",
    author="Based on work by Mengqi (Mandy) Xia et al.",
    description="BEM solver for 3D electromagnetic scattering from infinite cylinders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mandyxmq/WaveOpticsFiber",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "examples": ["matplotlib>=3.3.0"],
    },
)
