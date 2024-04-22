#pragma once 

#include <Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <Eigen/SVD>

std::pair<Eigen::Vector3d, Eigen::Vector3d> randomSampleLine(const std::vector<Eigen::Vector3d>& points);
std::vector<Eigen::Vector3d> randomSamplePlane(const std::vector<Eigen::Vector3d>& points);
std::pair<Eigen::Vector3d, Eigen::Vector3d> fitLine(const Eigen::Vector3d& first, const Eigen::Vector3d& second);

double calculateDistance(const std::pair<Eigen::Vector3d, Eigen::Vector3d>& line, const Eigen::Vector3d& point);
double point_to_line_d(const Eigen::Vector3d &d, const Eigen::Vector3d &A, const Eigen::Vector3d &P);
bool ransac(const std::vector<Eigen::Vector3d>& points, int num_iterations, double distance_threshold
,std::pair<Eigen::Vector3d, Eigen::Vector3d> & result);

bool ransac(const std::vector<Eigen::Vector3d>& points, int num_iterations, double distance_threshold
,Eigen::Matrix<double,4,1> & result, std::vector<Eigen::Vector3d>& pts);


double dot(Eigen::MatrixXd v1, Eigen::MatrixXd v2);
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v);
double distance_SO3(const Eigen::MatrixXd & M);
Eigen::Vector3d computeEquidistantPoint(const Eigen::Vector3d& P1, const Eigen::Vector3d& d1,
                                        const Eigen::Vector3d& P2, const Eigen::Vector3d& d2);
void combine(const std::vector<int>& data, int start, std::vector<int>& current, int depth, 
             std::vector<std::tuple<int, int, int>>& result);
std::vector<std::tuple<int, int, int>> generateTriplets(const std::vector<int>& data);