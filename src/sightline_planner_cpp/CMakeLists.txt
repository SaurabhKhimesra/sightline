cmake_minimum_required(VERSION 3.16)
project(sightline_planner_cpp CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(MuJoCo REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/calibration.cpp
  src/qp.cpp
  src/kin.cpp
  src/viewgrid.cpp
  src/motion.cpp
  src/guard.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen MuJoCo::mujoco)

# libmujoco.so lives in the wheel, not on the loader's path
set_target_properties(${PROJECT_NAME} PROPERTIES
  INSTALL_RPATH "${MUJOCO_LIBRARY_DIR}"
  INSTALL_RPATH_USE_LINK_PATH TRUE)

install(DIRECTORY include/ DESTINATION include)
install(FILES cmake/FindMuJoCo.cmake DESTINATION share/${PROJECT_NAME}/cmake)
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)
ament_package(CONFIG_EXTRAS "cmake/sightline_planner_cpp-extras.cmake")
