# Finds the MuJoCo C library and its headers.
#
# There is no system package for MuJoCo here. It arrives with the Python wheel,
# which ships the C headers under include/mujoco and libmujoco.so beside them, so
# this looks there as well as in a normal install. Point MUJOCO_DIR at either one,
# by cache variable or environment variable, to say which copy to use.
#
# Sets MuJoCo_FOUND, MUJOCO_INCLUDE_DIR, MUJOCO_LIBRARY and the imported target
# MuJoCo::mujoco.

if(NOT MUJOCO_DIR AND DEFINED ENV{MUJOCO_DIR})
  set(MUJOCO_DIR "$ENV{MUJOCO_DIR}")
endif()

# No hint: ask a Python that has mujoco installed where its package sits.
if(NOT MUJOCO_DIR)
  set(_mujoco_pythons "python3")
  if(DEFINED ENV{VIRTUAL_ENV})
    list(PREPEND _mujoco_pythons "$ENV{VIRTUAL_ENV}/bin/python")
  endif()
  if(DEFINED ENV{SIGHTLINE_PYTHON})
    list(PREPEND _mujoco_pythons "$ENV{SIGHTLINE_PYTHON}")
  endif()
  foreach(_py IN LISTS _mujoco_pythons)
    execute_process(
      COMMAND "${_py}" -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))"
      OUTPUT_VARIABLE _mujoco_guess OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET RESULT_VARIABLE _mujoco_rc)
    if(_mujoco_rc EQUAL 0 AND EXISTS "${_mujoco_guess}")
      set(MUJOCO_DIR "${_mujoco_guess}")
      break()
    endif()
  endforeach()
endif()

find_path(MUJOCO_INCLUDE_DIR
  NAMES mujoco/mujoco.h
  HINTS "${MUJOCO_DIR}/include" "${MUJOCO_DIR}"
  PATHS /usr/local/include /usr/include)

find_library(MUJOCO_LIBRARY
  NAMES mujoco
  HINTS "${MUJOCO_DIR}/lib" "${MUJOCO_DIR}"
  PATHS /usr/local/lib /usr/lib)

# The wheel ships libmujoco.so.3.13.0 with no unversioned symlink, which
# find_library will not match, so take the versioned file itself.
if(NOT MUJOCO_LIBRARY)
  file(GLOB _mujoco_versioned
    "${MUJOCO_DIR}/libmujoco.so.*"
    "${MUJOCO_DIR}/lib/libmujoco.so.*")
  list(SORT _mujoco_versioned)
  list(REVERSE _mujoco_versioned)
  if(_mujoco_versioned)
    list(GET _mujoco_versioned 0 _mujoco_pick)
    set(MUJOCO_LIBRARY "${_mujoco_pick}" CACHE FILEPATH "MuJoCo library" FORCE)
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MuJoCo
  REQUIRED_VARS MUJOCO_LIBRARY MUJOCO_INCLUDE_DIR
  FAIL_MESSAGE "MuJoCo not found. Set MUJOCO_DIR to the directory holding include/mujoco and libmujoco.so")

if(MuJoCo_FOUND AND NOT TARGET MuJoCo::mujoco)
  add_library(MuJoCo::mujoco UNKNOWN IMPORTED)
  set_target_properties(MuJoCo::mujoco PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}")
  get_filename_component(MUJOCO_LIBRARY_DIR "${MUJOCO_LIBRARY}" DIRECTORY)
endif()

mark_as_advanced(MUJOCO_INCLUDE_DIR MUJOCO_LIBRARY)
