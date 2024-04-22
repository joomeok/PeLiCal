#pragma once 

#include <iostream>
#include <cmath>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "line_feature.hpp"
#include "util_func.hpp"
#include "pnp_ral_func.hpp"
#include "line_optimize.hpp"

#include <PoseLib/misc/re3q3.h>
#include <sophus/so3.hpp>
#include <random>
#include <algorithm>
#include <set>
class camera_pair {
    public:
     void set_intrinsics(const Eigen::Matrix<double,3,3> & intrinsic);
     void add_images(const cv::Mat & ref_rgb_img, const cv::Mat& target_rgb_img, const cv::Mat & ref_depth_img, const cv::Mat& target_depth_img);
     void add_line_features(std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> ref_p_pairs, std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> target_p_pairs , cv::Mat ref_depth_img_, cv::Mat target_depth_img_);
     bool perform_PnL();
     double check_rotation(const line_feature & line);
     bool check_translation(const Eigen::Matrix3d & R, std::vector<line_feature> & result_full,std::vector<line_feature> & result_pnl );
     double compute_so3_distance(std::vector<line_feature> lines_full,std::vector<line_feature> lines_pnl );

     cv::Mat current_ref_rgb;
     cv::Mat current_target_rgb;
     std::vector<line_feature> lines_opt;
     std::vector<line_feature> lines_full;
     std::vector<line_feature> lines_pnl;     
    //  Eigen::Matrix<double,3,3> M = Eigen::Matrix3d::Zero();
     int image_num = 0;
     std::vector<int> used_img_id;
     double so3_distance = 1e4;
     Eigen::Matrix3d R;
     Eigen::Vector3d t;
     Eigen::Matrix<double,3,4> P;
     Eigen::Matrix<double,3,3> target_K_ = Eigen::Matrix3d::Zero();
     Eigen::Matrix<double,3,3> source_K_ = Eigen::Matrix3d::Zero();
     double eps;
     int voting_thresh;
     int minimum_line_num;
     double convergency_cost;
};