from setuptools import find_packages, setup

package_name = "sightline_planner"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Saurabh Khimesra",
    maintainer_email="learningkhimesra@gmail.com",
    description="The robot's own planner: the overhead camera's view as a grid of off-limits directions, a guard that checks the arm against it every 10 ms, and the task that chooses which screw to drive. Imports nothing from the simulation.",
    license="MIT",
)
