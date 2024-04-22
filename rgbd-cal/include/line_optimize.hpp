#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "ceres/manifold.h"
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include "util_func.hpp"
#include "line_feature.hpp"
#include "ceres/loss_function.h"

class LineReprojectionFunctor {
    public:
        LineReprojectionFunctor(const std::pair<Eigen::Vector2d,Eigen::Vector2d> & point_pair, const Eigen::Matrix3d& intrinsic, const Eigen::Matrix<double,6,1>& plucker_ref,
const std::pair<Eigen::Vector3d, Eigen::Vector3d> &ref_3d_pair);
    
    template <typename T>
    bool operator()(const T* const rotvec, const T* const translation, T* residual) const;
    std::pair<Eigen::Vector2d,Eigen::Vector2d> point_pair_;
    Eigen::Matrix3d K_l = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double,6,1> plucker_targ_;
    Eigen::Matrix<double,6,1> plucker_ref_;
    std::pair<Eigen::Vector3d, Eigen::Vector3d> ref_3d_pair_;
    std::pair<Eigen::Vector3d, Eigen::Vector3d> targ_3d_pair_;

};

class FullPointLineFunctor {
    public : 
        FullPointLineFunctor(const Eigen::Matrix<double,6,1>& plucker_ref, const Eigen::Matrix<double,6,1>& plucker_targ,
const std::pair<Eigen::Vector3d, Eigen::Vector3d> &ref_3d_pair, const std::pair<Eigen::Vector3d, Eigen::Vector3d> &targ_3d_pair);

    template <typename T>
    bool operator()(const T* const rotvec, const T* const translation, T* residual) const;
    Eigen::Matrix<double,6,1> plucker_targ_;
    Eigen::Matrix<double,6,1> plucker_ref_;
    std::pair<Eigen::Vector3d, Eigen::Vector3d> ref_3d_pair_;
    std::pair<Eigen::Vector3d, Eigen::Vector3d> targ_3d_pair_;
};


// main function
double optimize_line_reprojection(const std::vector<line_feature> & lines, const Eigen::Matrix3d & intrinsic, 
const Eigen::Matrix3d & R_init,  const Eigen::Vector3d & t_init, Eigen::Matrix3d & R_opt, Eigen::Vector3d & t_opt);

// The cost functor for the least squares problem
class LineCost {

    public:
        LineCost(const Eigen::Vector3d& p);

    template <typename T>
    bool operator()(const T* const line_params, T* residual) const;
    Eigen::Vector3d point;
};

double fit_line(const std::vector<Eigen::Vector3d> & inliers, const Eigen::Vector3d & a, const Eigen::Vector3d & b, Eigen::Vector3d & a_opt, Eigen::Vector3d& b_opt);
bool ransac_optimize(const std::vector<Eigen::Vector3d>& points, int num_iterations, double distance_threshold
,std::pair<Eigen::Vector3d, Eigen::Vector3d> & result);