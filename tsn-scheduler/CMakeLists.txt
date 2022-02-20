cmake_minimum_required(VERSION 3.14)

project(tsn_scheduler)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

file(GLOB autopilot_sources src/*.cc src/gridworld/*.cc src/utility/*.cc)
set(LIB_SRC_FILES ${autopilot_sources})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -std=c++11 -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-reorder -Wno-sign-compare -Wno-missing-braces")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")

# runtime library -> tsn_scheduler.build
add_library(tsn_scheduler SHARED ${LIB_SRC_FILES})