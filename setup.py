from setuptools import find_packages, setup

package_name = "moon_explore_challenge"

data_files = []
data_files.append(("share/ament_index/resource_index/packages", ["resource/" + package_name]))
data_files.append(("share/" + package_name + "/resource", [""]))
data_files.append(("share/" + package_name, ["package.xml"]))


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=[
        "setuptools",
        "scikit-image",
    ],
    zip_safe=True,
    maintainer="yyy",
    maintainer_email="1830979240@qq.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        # "console_scripts": [
        #     "explore_controller = moon_explore.explore_controller:main",
        # ],
    },
)
