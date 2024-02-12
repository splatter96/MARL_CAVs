from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["highway_env/road/lane.pyx",
                             "highway_env/road/road.py",
                             "highway_env/vehicle/kinematics.py",
                             "highway_env/utils.pyx"],
                            annotate=True)
)

