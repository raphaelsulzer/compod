cmake_minimum_required(VERSION 3.14.0)
project(COMPOSE LANGUAGES CXX)

######################################################
###################### SETTINGS ######################
######################################################
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

######################################################
###################### EXTERNAL ######################
######################################################
find_package(CGAL REQUIRED)
include(${CGAL_USE_FILE})
## issues with eigen? Do this: https://stackoverflow.com/a/25537967

if(NOT SKBUILD)
    # manually set the conda env when generating with cmake
    set(CONDA_ENV "ksr3")
    set(Python_ROOT_DIR "/opt/miniconda3/envs/${CONDA_ENV}/bin")
endif()
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
include_directories(${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS system filesystem thread serialization program_options)

# Run `nanobind.cmake_dir()` from Python to detect where nanobind is installed
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

find_package(Eigen3 REQUIRED PATHS $CONDA_PREFIX/include/eigen3)
find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)
find_package(xtl REQUIRED) # for xtensor
find_package(ZLIB REQUIRED) # for xtensor
find_package(xtensor REQUIRED)
find_package(xtensor-io REQUIRED)

######################################################
###################### INTERNAL ######################
######################################################
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")


###########################################
############ libSurfaceSimplifier ############
###########################################
add_library(SurfaceSimplifier STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/pdse.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mesh.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cdt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/plane.cpp"
)
target_link_libraries(SurfaceSimplifier ${Boost_LIBRARIES})
target_link_libraries(SurfaceSimplifier ${CGAL_LIBRARIES})
target_link_libraries(SurfaceSimplifier ZLIB::ZLIB xtensor xtensor-io)
target_link_libraries(SurfaceSimplifier spdlog::spdlog)

if(NOT SKBUILD)
    add_executable(SurfaceSimplifierExe "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    target_link_libraries(SurfaceSimplifierExe SurfaceSimplifier)
    target_link_libraries(SurfaceSimplifierExe ${Boost_LIBRARIES})
    target_link_libraries(SurfaceSimplifierExe ${CGAL_LIBRARIES})
    target_link_libraries(SurfaceSimplifierExe ZLIB::ZLIB xtensor xtensor-io)
    target_link_libraries(SurfaceSimplifierExe spdlog::spdlog)
    target_link_libraries(SurfaceSimplifierExe)
endif()

##########################################
############# python bindings ############
##########################################
nanobind_add_module(libPYPDSE "${CMAKE_CURRENT_SOURCE_DIR}/src/pypdse.cpp")
if(SKBUILD)
    include_directories("${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(libPYPDSE PUBLIC SurfaceSimplifier spdlog::spdlog)
    install(TARGETS libPYPDSE LIBRARY DESTINATION pycompose)
endif()

nanobind_add_module(libPYPDL "${CMAKE_CURRENT_SOURCE_DIR}/src/pypdl.cpp")
if(SKBUILD)
    include_directories("${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(libPYPDL PUBLIC ${CGAL_LIBRARIES} spdlog::spdlog)
    install(TARGETS libPYPDL LIBRARY DESTINATION pycompose)
endif()
