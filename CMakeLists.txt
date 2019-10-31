cmake_minimum_required(VERSION 2.8.3)
project(test)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

set(OpenCV_DIR /usr/local/opencv3/share/OpenCV)

find_package(OpenCV REQUIRED)
include_directories("/usr/include/eigen3")

find_package(G2O REQUIRED)
find_package( CSparse REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${G2O_INCLUDE_DIRS})
include_directories(${CSPARSE_INCLUDE_DIR})


add_executable(lidtocam lidtocam.cpp)
target_link_libraries(lidtocam ${OpenCV_LIBS} ${CSPARSE_LIBRARY} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)
