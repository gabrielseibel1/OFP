cmake_minimum_required(VERSION 3.12)
project(OFP)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
add_executable(OFP main.cpp)
target_link_libraries(OFP ${OpenCV_LIBS})