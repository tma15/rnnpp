cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

add_library(rnnpp SHARED
	dim.h dim.cc
	expr.h expr.cc
	error.h
	gradcheck.h gradcheck.cc
	graph.h
	optimizer.h optimizer.cc
	parameter.h parameter.cc
	tensor.h tensor.cc
	node.h node.cc
	rnnpp.h rnnpp.cc
	)
