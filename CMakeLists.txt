cmake_minimum_required(VERSION 3.12)
project(OFP)

set(CMAKE_CXX_STANDARD 17)

#opengm
include_directories(opengm/include)
add_executable(OFP src/main.cpp
        src/grid_potts_example.h
        src/grid_potts_example.cpp
        opengm/include/opengm/graphicalmodel/graphicalmodel.hxx
        opengm/include/opengm/graphicalmodel/space/simplediscretespace.hxx
        opengm/include/opengm/functions/potts.hxx
        opengm/include/opengm/operations/adder.hxx
        opengm/include/opengm/inference/messagepassing/messagepassing.hxx
        opengm/include/opengm/inference/gibbs.hxx
)

#opencv
find_package(OpenCV REQUIRED)
target_link_libraries(OFP ${OpenCV_LIBS})