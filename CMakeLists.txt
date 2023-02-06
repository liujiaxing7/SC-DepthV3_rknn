cmake_minimum_required(VERSION 3.10)
project(SC_DepthV3_rknn)

set(CMAKE_CXX_STANDARD 14)

# opencv
find_package(OpenCV REQUIRED)

# npu
include_directories(/usr/include/)

add_executable(SC_DepthV3_rknn main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )
target_link_libraries(${PROJECT_NAME}  /usr/lib/librknn_api.so)
target_link_libraries(${PROJECT_NAME}  /usr/lib/librknnrt.so)