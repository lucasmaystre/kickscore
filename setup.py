from setuptools import setup
from os import path


HERE = path.abspath(path.dirname(__file__))


def readme():
    with open(path.join(HERE, "README.rst")) as f:
        return f.read()


setup(
    name="kickscore",
    version="0.1.4",
    author="Lucas Maystre",
    author_email="lucas@maystre.ch",
    description="A dynamic skill rating system.",
    long_description=readme(),
    url="https://github.com/lucasmaystre/kickscore",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
    ],
    keywords="elo ranking skill score rating strength game comparison match",
    packages=[
        "kickscore",
        "kickscore.fitter",
        "kickscore.kernel",
        "kickscore.observation",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "numba"
    ],
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest",
    ],
    include_package_data=True,
    zip_safe=False,
)
