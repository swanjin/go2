cmake_minimum_required(VERSION 3.5)
project(helloworld_python_wrapper)

SET(CMAKE_CXX_STANDARD 17)

# check arch and os
message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
    set(ARCH x86_64)
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
    set(ARCH aarch64)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/${ARCH})
file(GLOB_RECURSE sources "cpp/*.cpp")

set(EXTRA_LIBS libunitree_sdk2.a)
# set(EXTRA_LIBS -pthread rt m libunitree_sdk2.a)
# python
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../lib/python/${ARCH}) # .so .dylib

add_subdirectory(third-party/pybind11)
include_directories(/usr/local/include/ddscxx /usr/local/include/iceoryx/v2.0.2)

pybind11_add_module(robot_interface python_interface.cpp)
target_link_libraries(robot_interface PRIVATE ${EXTRA_LIBS} libunitree_sdk2.a unitree_sdk2 ddsc ddscxx rt pthread)

