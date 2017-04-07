from setuptools import setup
from os import path


HERE = path.abspath(path.dirname(__file__))


def readme():
    with open(path.join(HERE, "README.md")) as f:
        return f.read()


setup(
    name="kickscore",
    version="0.1.0",
    author="Lucas Maystre",
    author_email="lucas@maystre.ch",
    description="A dynamic skill rating system.",
    long_description=readme(),
    url="http://lucas.maystre.ch/",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
    ],
    keywords="elo ranking skill score rating strength game",
    packages=["kickscore"],
    install_requires=[
        "numpy",
        "scipy",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    include_package_data=True,
    zip_safe=False,
)
