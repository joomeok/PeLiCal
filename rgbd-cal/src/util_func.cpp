#include "util_func.hpp"

std::pair<Eigen::Vector3d, Eigen::Vector3d> randomSampleLine(const std::vector<Eigen::Vector3d>& points) {
    // Create a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    // Uniform distribution over the indices of the points
    std::uniform_int_distribution<> distrib(0, points.size() - 1);
    
    // Sample two distinct indices
    int i, j;
    do {
        i = distrib(gen);
        j = distrib(gen);
    } while (i == j);
    // Return the corresponding points
    return std::make_pair(points[i], points[j]);
}


// Fit a line to two points
std::pair<Eigen::Vector3d, Eigen::Vector3d> fitLine(const Eigen::Vector3d& first, const Eigen::Vector3d& second) {
    return std::make_pair(first,second);
}



// Calculate the distance from a point to a line
double calculateDistance(const std::pair<Eigen::Vector3d, Eigen::Vector3d>& line, const Eigen::Vector3d& point) {
    Eigen::Vector3d d = line.first - line.second;

    double t = (point - line.first).dot(d) / (d(0,0)*d(0,0) + d(1,0)*d(1,0) + d(2,0)*d(2,0));
    double distance = pow(point(0,0) - line.first(0,0) - d(0,0) *t, 2) + pow(point(1,0) - line.first(1,0) - d(1,0) *t, 2) 
    + pow(point(2,0) - line.first(2,0) - d(2,0) *t, 2);
    return distance;
}



double point_to_line_d(const Eigen::Vector3d &d, const Eigen::Vector3d &A, const Eigen::Vector3d &P){
    Eigen::Vector3d AtoP = P-A;
    Eigen::Vector3d V = AtoP.cross(d);
    double distance = V.norm() / d.norm();
    return distance;
}

// return best line by ransac
bool ransac(const std::vector<Eigen::Vector3d>& points, int num_iterations, double distance_threshold
,std::pair<Eigen::Vector3d, Eigen::Vector3d> & result) {
    if(points.size() < 10) return false;
    std::pair<Eigen::Vector3d, Eigen::Vector3d> best_line;
    int max_inliers = 0;
    for (int i = 0; i < num_iterations; ++i) {
        auto [first, second] = randomSampleLine(points);
        std::pair<Eigen::Vector3d, Eigen::Vector3d> line = fitLine(first, second);
        int num_inliers = 0;
        for (const Eigen::Vector3d& point : points) {
            // std::cout << "Line dist : " << calculateDistance(line,point) << std::endl;
            if (calculateDistance(line, point) < distance_threshold) {
                ++num_inliers;
            }
        }

        if (num_inliers > max_inliers) {
            max_inliers = num_inliers;
            best_line = line;
        }
    }

    double inlier_ratio = (double) max_inliers / (double) points.size();
    // std::cout << "Inlier ratio : " << inlier_ratio  << std::endl;
    if(inlier_ratio < 0.6) {
        return false;}
    else{
        result = best_line; 
        return true;
    } 
}


double dot(Eigen::MatrixXd v1, Eigen::MatrixXd v2){
    int n = v1.rows();
    double result = 0;
    for(int i = 0; i < n; i++){
        result += v1(i,0) * v2(i,0);
    }
    return result;
}

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d m;
    m << 0, -v(2), v(1),
         v(2), 0, -v(0),
        -v(1), v(0), 0;
    return m;
}

double distance_SO3(const Eigen::MatrixXd & M){
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Extracting the components
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd S = svd.singularValues();
    
    // Create the Sigma matrix
    Eigen::MatrixXd Sigma = Eigen::MatrixXd::Zero(3,3);
    for (int i = 0; i < S.size(); ++i) {
        Sigma(i, i) = S(i);
    }

    Eigen::MatrixXd SigmaPrime = Eigen::MatrixXd::Identity(3, 3);
    SigmaPrime(2,2) = (U*V.transpose()).determinant();
    // std::cout << "Angles with only 3D : " << projected_SO3.eulerAngles(0,1,2) * (180/M_PI) << std::endl;
    return (Sigma - SigmaPrime).norm();
}



Eigen::Vector3d computeEquidistantPoint(const Eigen::Vector3d& P1, const Eigen::Vector3d& d1,
                                        const Eigen::Vector3d& P2, const Eigen::Vector3d& d2) {
    Eigen::Vector3d w = P1 - P2;

    Eigen::Vector3d cross_d1_d2 = d1.cross(d2);
    double denominator = cross_d1_d2.squaredNorm();

    // Almost parallel check
    if (denominator < 1e-10) {
        std::cerr << "Lines are almost parallel!" << std::endl; 
        return Eigen::Vector3d::Zero();
    }

    double s = (d2.cross(w)).dot(cross_d1_d2) / denominator;
    double t = (d1.cross(w)).dot(cross_d1_d2) / denominator;

    Eigen::Vector3d Q1 = P1 + s * d1;
    Eigen::Vector3d Q2 = P2 + t * d2;

    return (Q1 + Q2) / 2;
}

void combine(const std::vector<int>& data, int start, std::vector<int>& current, int depth, 
             std::vector<std::tuple<int, int, int>>& result) {
    if (depth == 3) {
        result.emplace_back(current[0], current[1], current[2]);
        return;
    }
    
    for (std::size_t i = start; i < data.size(); ++i) {
        current.push_back(data[i]);
        combine(data, i + 1, current, depth + 1, result);
        current.pop_back();
    }
}

std::vector<std::tuple<int, int, int>> generateTriplets(const std::vector<int>& data) {
    std::vector<std::tuple<int, int, int>> result;
    std::vector<int> current;
    combine(data, 0, current, 0, result);
    return result;
}
