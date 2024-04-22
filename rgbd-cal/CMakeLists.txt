cmake_minimum_required(VERSION 3.1)
project(pelical)
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    message_filters
    cv_bridge
    message_generation
    gluestick_detection
    )
    
find_package(OpenCV REQUIRED)
find_package(PoseLib REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus)
find_package(Eigen3 REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})
include_directories(${SOPHUS_INCLUDE_DIRS})
include_directories(/usr/local/include/sophus)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_message_files(
  FILES
  CustomFloatArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)
catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp cv_bridge message_filters message_runtime
    )

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    ./include)


add_library(calib ./src/cam_pair.cpp ./src/line_feature.cpp ./src/line_optimize.cpp  ./src/pnp_ral_func.cpp ./src/util_func.cpp)
target_link_libraries( calib ${OpenCV_LIBS} ${Eigen_LIBRARIES} ${CERES_LIBRARIES} Eigen3::Eigen PoseLib::PoseLib ${Sophus_LIBRARIES})

add_executable(calibration_node ./src/main.cpp)
target_link_libraries(calibration_node ${catkin_LIBRARIES} calib)
