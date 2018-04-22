cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

enable_testing()

link_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${GTEST_PATH}/lib)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${GTEST_PATH}/include)

foreach(TESTNAME expr dim graph node tensor parameter)
	add_executable(rnnpp_${TESTNAME}_test main.cc ${TESTNAME}_test.cc)
	add_test(NAME rnnpp_${TESTNAME}_test COMMAND rnnpp_${TESTNAME}_test)
	target_link_libraries(rnnpp_${TESTNAME}_test rnnpp gtest gtest_main pthread)
endforeach(TESTNAME)
