cmake_minimum_required(VERSION 2.8.3)
project(jetson_semantic_map)

find_package(OpenMP REQUIRED)

add_definitions(-DPROJECT_NAME="${PROJECT_NAME}")
set(CMAKE_BUILD_TYPE RelWithDebInfo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMPILER_FLAGS_DEB_REL} -g -O2 -Wall -Wno-unused-function -Wno-reorder -Wfatal-errors -march=native ${OpenMP_C_FLAGS} ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPILER_FLAGS_DEB_REL} -g -O2 -Wall -Wno-unused-function -Wno-reorder -Wfatal-errors -march=native ${OpenMP_CXX_FLAGS} ")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  cv_bridge
  tf2
  tf2_ros
  tf2_eigen
  message_filters
  std_msgs
  geometry_msgs
  sensor_msgs
  image_transport
  rosbag
  message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
   FILES
   MappOccIdx.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs
)

find_package(OpenCV REQUIRED)
message("OpenCV version: ${OpenCV_VERSION}")

catkin_package(
  INCLUDE_DIRS
  LIBRARIES
  CATKIN_DEPENDS roscpp roslib message_runtime
)

include_directories(
  include
  ${Eigen_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${OPENCV_INCLUDE_DIRS}
)

## Declare a cpp executable
add_executable(${PROJECT_NAME}_node src/${PROJECT_NAME}_node.cpp src/cloud_fusion.cpp src/SemanticFusor.cpp src/color_maps.cpp)

add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES} ${OpenCV_LIBS}
)
