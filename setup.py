# setup.py
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "angle_calculator_cpp",
        ["angle_calculator_cpp.cpp"],
        extra_compile_args=[
            '-O3',              # Maximum optimization
            '-march=native',    # Optimize for RPi 4 CPU
            '-DNDEBUG',         # Disable debug assertions
            '-ffast-math',      # Fast math operations
        ],
    ),
]

setup(
    name="angle_calculator_cpp",
    version="1.0.0",
    author="Your Name",
    description="Fast C++ angle calculator for pose estimation",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)