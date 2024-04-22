#pragma once

#include <iostream>
#include <cmath>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <eigen3/Eigen/Dense>

class line_feature {
    public:
     line_feature(const std::pair<Eigen::Vector2d, Eigen::Vector2d> & ref_2d_point_pair,const std::pair<Eigen::Vector2d, Eigen::Vector2d> & target_2d_point_pair 
                  ,const std::pair<Eigen::Vector3d, Eigen::Vector3d> & ref_3d_pair, const std::pair<Eigen::Vector3d, Eigen::Vector3d> & target_3d_pair, int id, int case_);
     void multiply_K(const Eigen::Matrix3d & K);

     std::pair<Eigen::Vector2d, Eigen::Vector2d> target_img_2d_pair_;
     std::pair<Eigen::Vector2d, Eigen::Vector2d> ref_img_2d_pair_;
     std::pair<Eigen::Vector3d, Eigen::Vector3d> target_3d_pair_;
     std::pair<Eigen::Vector3d, Eigen::Vector3d> ref_3d_pair_;

     Eigen::Matrix<double,6,1> plucker_ref; // plucker_refcoordinate from ref_frame
     Eigen::Matrix<double,6,1> plucker_target;
     Eigen::Vector3d K_mutliplied_coord;
     Eigen::Vector3d line_coord; // 2D line_coordinate from target image
     int img_id = 0;
     int case_ = 0;
};