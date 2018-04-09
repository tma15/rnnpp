cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

link_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(train_xor xor/train_xor.cc)
target_link_libraries(train_xor rnnpp)

add_executable(train_xor_batch xor/train_xor_batch.cc)
target_link_libraries(train_xor_batch rnnpp)

add_executable(train_embed embed/train_embed.cc)
target_link_libraries(train_embed rnnpp)
