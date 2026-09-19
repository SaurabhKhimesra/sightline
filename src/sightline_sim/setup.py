from setuptools import find_packages, setup

package_name = "sightline_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={'sightline_sim.data': ['*.yaml']},
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Saurabh Khimesra",
    maintainer_email="learningkhimesra@gmail.com",
    description="The cell in MuJoCo: the station, the worker and his cycle, the cameras with their noise, the judge that scores the two rules from the simulator's own state, the validation gates, and the export to Gazebo Sim.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "gate1_stills = sightline_sim.gates.gate1_stills:main",
            "gate1_numbers = sightline_sim.gates.gate1_numbers:main",
            "gate2_cycle = sightline_sim.gates.gate2_cycle:main",
            "gate3_b0 = sightline_sim.gates.gate3_b0:main",
            "gate4_perception = sightline_sim.gates.gate4_perception:main",
            "gate4_hole_views = sightline_sim.gates.gate4_hole_views:main",
            "gate5_planner = sightline_sim.gates.gate5_planner:main",
            "eyes_options = sightline_sim.gates.eyes_options:main",
            "eyes_explain = sightline_sim.gates.eyes_explain:main",
            "eyes_optimise = sightline_sim.gates.eyes_optimise:main",
            "blind_zone = sightline_sim.gates.blind_zone:main",
            "make_replay_world = sightline_sim.gazebo.make_replay_world:main",
            "mjcf_to_sdf = sightline_sim.gazebo.mjcf_to_sdf:main",
            "replay_gazebo = sightline_sim.gazebo.replay_gazebo:main",
            "frame_grabber = sightline_sim.gazebo.frame_grabber:main",
            "assemble_video = sightline_sim.gazebo.assemble_video:main",
            "caption_video = sightline_sim.gazebo.caption_video:main"
        ],
    },
)
