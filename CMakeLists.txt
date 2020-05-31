cmake_minimum_required(VERSION 2.8)

project(lab6)
find_package(OpenCV REQUIRED)

set(SOURCE_FILES src/main.cpp src/objectDetection.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})
