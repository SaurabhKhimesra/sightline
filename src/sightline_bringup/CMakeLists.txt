cmake_minimum_required(VERSION 3.10)
project(sightline_bringup NONE)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config urdf DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/record_windows.sh scripts/record_bag_rviz.sh DESTINATION lib/${PROJECT_NAME})

ament_package()
