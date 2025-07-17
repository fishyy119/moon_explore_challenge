from typing import List, Tuple

from setuptools import find_packages, setup

package_name = "path_planner"

data_files: List[Tuple[str, List[str]]] = []
data_files.append(("share/ament_index/resource_index/packages", ["resource/" + package_name]))
# data_files.append(("share/" + package_name + "/resource", ["resource/map_passable.npy"]))
# TODO：添加rs_table7x7.npy
data_files.append(("share/" + package_name, ["package.xml"]))


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yyy",
    maintainer_email="1830979240@qq.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "path_planner_node = path_planner.path_planner_node:main",
            "hpath_viz_node = path_planner.hpath_viz_node:main",
        ],
    },
)
