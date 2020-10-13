cmake_minimum_required(VERSION 3.16.3)
project(NewPr)
find_package(OpenCV REQUIRED)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(NewPr main.cpp)
target_link_libraries(NewPr ${OpenCV_LIBS})