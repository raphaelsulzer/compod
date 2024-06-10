cmake_minimum_required(VERSION 3.14.0)
project(COMPOSE LANGUAGES CXX)

######################################################
###################### SETTINGS ######################
######################################################
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

######################################################
###################### EXTERNAL ######################
######################################################
#include(${CGAL_USE_FILE})
## issues with eigen? Do this: https://stackoverflow.com/a/25537967

# if(NOT SKBUILD)
    # manually set the conda env when generating with cmake
    set(CONDA_ENV "compocity2")
    set(Python_ROOT_DIR "/opt/miniconda3/envs/${CONDA_ENV}/bin")
    set(Python_FIND_STRATEGY "LOCATION")
    set(CONDA_PREFIX "/opt/miniconda3/envs/${CONDA_ENV}")
# endif()
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
include_directories(${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

set(CMAKE_PREFIX_PATH ${CONDA_PREFIX})

# Run `nanobind.cmake_dir()` from Python to detect where nanobind is installed
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

find_package(CGAL REQUIRED HINTS $CONDA_PREFIX/include/CGAL)
include(${CGAL_USE_FILE})
find_package(Boost REQUIRED COMPONENTS system filesystem thread serialization program_options)
find_package(Eigen3 REQUIRED PATHS $CONDA_PREFIX/include/eigen3)
find_package(yaml-cpp REQUIRED)
### for spdlog, found no better solution than to change its code with this: https://github.com/gabime/spdlog/issues/2767#issuecomment-1604460152
find_package(spdlog REQUIRED)
find_package(xtl REQUIRED) # for xtensor
find_package(ZLIB REQUIRED) # for xtensor
find_package(xtensor REQUIRED)
find_package(xtensor-io REQUIRED)

######################################################
###################### INTERNAL ######################
######################################################
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CONDA_PREFIX}/include")


##########################################
############# python bindings ############
##########################################
nanobind_add_module(libPYPDSE "${CMAKE_CURRENT_SOURCE_DIR}/src/pypdse.cpp")
target_include_directories(libPYPDSE PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(libPYPDSE PUBLIC "${CONDA_PREFIX}/include")
target_link_libraries(libPYPDSE PUBLIC ${Boost_LIBRARIES})
target_link_libraries(libPYPDSE PUBLIC CGAL::CGAL)
target_link_libraries(libPYPDSE PUBLIC spdlog::spdlog)
### including this breaks the whole package, because it adds /usr/local/includes to the compiler includes, which in turn
### leads to CGAL being included from /usr/local/include/CGAL instead of the miniconda CGAL
# target_link_libraries(libPYPDSE PUBLIC ZLIB::ZLIB)
# target_link_libraries(libPYPDSE PUBLIC xtensor xtensor-io)
install(TARGETS libPYPDSE LIBRARY DESTINATION pycompose)

nanobind_add_module(libPYPDL "${CMAKE_CURRENT_SOURCE_DIR}/src/pypdl.cpp")
target_include_directories(libPYPDL PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(libPYPDL PUBLIC "${CONDA_PREFIX}/include")
target_link_libraries(libPYPDSE PUBLIC ${Boost_LIBRARIES})
target_link_libraries(libPYPDL PUBLIC CGAL::CGAL)
target_link_libraries(libPYPDL PUBLIC spdlog::spdlog)
install(TARGETS libPYPDL LIBRARY DESTINATION pycompose)
