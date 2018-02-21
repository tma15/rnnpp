cmake_minimum_required(VERSION 3.1)

add_subdirectory(src)

message("EIGEN_PATH:" ${EIGEN_PATH})
message("GTEST_PATH:" ${GTEST_PATH}) 

include_directories(${EIGEN_PATH})

if(GTEST_PATH) 
	message("ENABLE TEST")
	enable_testing()
	add_subdirectory(tests)
endif()
