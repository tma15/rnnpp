cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

link_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(rnnpp_test main.cpp)

target_link_libraries(rnnpp_test rnnpp)
