cmake_minimum_required(VERSION 3.1)

add_subdirectory(src)
add_subdirectory(examples)

message("EIGEN_PATH:" ${EIGEN_PATH})
message("GTEST_PATH:" ${GTEST_PATH}) 

include_directories(${EIGEN_PATH})

if(GTEST_PATH) 
	message("ENABLE TEST")
	add_subdirectory(tests)
	enable_testing()
endif()
