#pragma once 

#include <eigen3/Eigen/Core>
#include <vector>
#include "line_feature.hpp"
#include "util_func.hpp"

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> construct_A_B (const std::vector<line_feature>& lines_full,const std::vector<line_feature>& lines_pnl);
Eigen::Matrix<double,3,10> compute_C(Eigen::MatrixXd K, int n);
Eigen::Matrix3d restore_R(double s1, double s2 ,double s3, const Eigen::MatrixXd & A, const Eigen::MatrixXd & B);
Eigen::Vector3d restore_t(double s1, double s2 ,double s3, const Eigen::MatrixXd & A, const Eigen::MatrixXd & B);