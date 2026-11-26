from glob import glob
import os
from setuptools import find_packages, setup

package_name = "khonshu_planner"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "scipy"],
    zip_safe=True,
    maintainer="Ivan Barasa",
    maintainer_email="ivan@khonshu.local",
    description="Khonshu FS-AI path planning package (v0.1 baseline).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "planner_node = khonshu_planner.nodes.planner_node:main",
            "pure_pursuit_sim = khonshu_planner.nodes.pure_pursuit_sim:main",
        ],
    },
)
