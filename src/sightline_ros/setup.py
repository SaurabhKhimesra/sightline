from setuptools import find_packages, setup

package_name = "sightline_ros"

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
    description="The ROS 2 layer: the cell and the planner as nodes in lockstep on sim time, the topic contract between them, and the live replay of a run into Gazebo Sim and rviz.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "cell_node = sightline_ros.cell_node:main",
            "planner_node = sightline_ros.planner_node:main",
            "replay_live = sightline_ros.replay_live:main",
            "xwin = sightline_ros.xwin:main"
        ],
    },
)
