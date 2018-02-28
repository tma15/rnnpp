cmake_minimum_required(VERSION 3.1)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN_PATH})

set(CMAKE_CXX_STANDARD 11)

add_library(rnnpp SHARED
	dim.h
	expr.h expr.cc
	error.h
	graph.h
	tensor.h
	node.h node.cc
	rnnpp.h rnnpp.cc
	)
