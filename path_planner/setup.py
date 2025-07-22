import glob
from typing import List, Tuple

from setuptools import find_packages, setup

package_name = "path_planner"

data_files: List[Tuple[str, List[str]]] = []
data_files.append(("share/ament_index/resource_index/packages", ["resource/" + package_name]))

rs_table_files = glob.glob("resource/rs_table*")
data_files.append(("share/" + package_name + "/resource", rs_table_files + ["resource/map_raw.txt"]))
data_files.append(("share/" + package_name, ["package.xml"]))


setup(
    name=package_name,
    version="0.20250722.1",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yyy",
    maintainer_email="1830979240@qq.com",
    description="规划主体包，包括基于混合A*算法的路径规划与探索规划",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "path_planner_node = path_planner.path_planner_node:main",
            "hpath_viz_node = path_planner.hpath_viz_node:main",
            "test_planner_node = path_planner.test_planner_node:main",
        ],
    },
)
