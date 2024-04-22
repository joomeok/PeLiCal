#include "line_feature.hpp"

line_feature::line_feature(const std::pair<Eigen::Vector2d, Eigen::Vector2d> & ref_2d_point_pair,const std::pair<Eigen::Vector2d, Eigen::Vector2d> & target_2d_point_pair 
            ,const std::pair<Eigen::Vector3d, Eigen::Vector3d> & ref_3d_pair, const std::pair<Eigen::Vector3d, Eigen::Vector3d> & target_3d_pair, int id, int case_){
    ref_img_2d_pair_ = ref_2d_point_pair;
    target_img_2d_pair_ = target_2d_point_pair;
    ref_3d_pair_ = ref_3d_pair;
    target_3d_pair_ = target_3d_pair;
    img_id = id;
    this->case_ = case_;
    // X축이 오른쪽 가로방향 
    plucker_ref.block(0,0,3,1) = (ref_3d_pair_.second - ref_3d_pair_.first) / (ref_3d_pair_.second - ref_3d_pair_.first).norm(); // d
    plucker_ref.block(3,0,3,1) = ref_3d_pair_.first.cross(ref_3d_pair_.second) / (ref_3d_pair_.second - ref_3d_pair_.first).norm() ;// m
    plucker_ref *= 10;
    if(plucker_ref(0,0) < 0 ) plucker_ref *= -1;
    
    // full 3D
    if(case_ == 0 ){
        plucker_target.block(0,0,3,1) = (target_3d_pair_.second - target_3d_pair_.first)/ (target_3d_pair_.second - target_3d_pair_.first).norm(); // d
        plucker_target.block(3,0,3,1) = target_3d_pair_.first.cross(target_3d_pair_.second)/(target_3d_pair_.second - target_3d_pair_.first).norm(); // m
        plucker_target *= 10;
        if(plucker_target(0,0) < 0 ) plucker_target *= -1;
    }


    double a = (target_img_2d_pair_.second(1,0) - target_img_2d_pair_.first(1,0)) / 2000.0;
    double b = (target_img_2d_pair_.first(0,0) - target_img_2d_pair_.second(0,0)) / 2000.0;
    double c = (target_img_2d_pair_.second(0,0)*target_img_2d_pair_.first(1,0) - target_img_2d_pair_.first(0,0)*target_img_2d_pair_.second(1,0))/2000.0;
    line_coord << a, b, c;
}

void line_feature::multiply_K(const Eigen::Matrix3d & K){
    K_mutliplied_coord = K.transpose() * line_coord;
}