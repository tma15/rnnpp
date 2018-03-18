cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

enable_testing()

link_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${GTEST_PATH}/lib)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${GTEST_PATH}/include)

add_executable(rnnpp_tensor_test main.cc test_tensor.cc)
add_test(NAME rnnpp_tensor_test COMMAND rnnpp_tensor_test)

add_executable(rnnpp_node_test main.cc test_node.cc)
add_test(NAME rnnpp_node_test COMMAND rnnpp_node_test)

add_executable(rnnpp_dim_test main.cc test_dim.cc)
add_test(NAME rnnpp_dim_test COMMAND rnnpp_dim_test)

target_link_libraries(rnnpp_tensor_test rnnpp gtest gtest_main pthread)
target_link_libraries(rnnpp_node_test rnnpp gtest gtest_main pthread)
target_link_libraries(rnnpp_dim_test rnnpp gtest gtest_main pthread)
