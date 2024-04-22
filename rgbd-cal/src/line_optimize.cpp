#include "line_optimize.hpp"


FullPointLineFunctor::FullPointLineFunctor(const Eigen::Matrix<double,6,1>& plucker_ref, const Eigen::Matrix<double,6,1>& plucker_targ,
const std::pair<Eigen::Vector3d, Eigen::Vector3d> &ref_3d_pair, const std::pair<Eigen::Vector3d, Eigen::Vector3d> &targ_3d_pair)
:  plucker_ref_(plucker_ref), plucker_targ_(plucker_targ), ref_3d_pair_(ref_3d_pair), targ_3d_pair_(targ_3d_pair) {
}
    
template <typename T>
bool FullPointLineFunctor::operator()(const T* const rotvec, const T* const translation, T* residual) const {

    Eigen::Matrix<T, 3, 1> t(translation[0], translation[1], translation[2]);

    // Angle Axis version
    Eigen::Matrix<T, 3, 1> rotation_vector(rotvec[0], rotvec[1], rotvec[2]);
    T angle = rotation_vector.norm();
    Eigen::Matrix<T, 3, 1> axis = rotation_vector.normalized();
     Eigen::AngleAxis<T> angle_axis(angle, axis);

    // Convert angle-axis to rotation matrix
    Eigen::Matrix<T, 3, 3> R = angle_axis.toRotationMatrix();

    // Projection using camera intrinsics
    Eigen::Matrix<T, 3, 3> I = Eigen::Matrix<T, 3, 3>::Identity();
    Eigen::Matrix<T,3,1> d = (targ_3d_pair_.second.template cast<T>() - targ_3d_pair_.first.template cast<T>()).normalized();
    
    Eigen::Matrix<T, 3, 1> res1 = (I - (d * d.transpose()).eval()) * (R * ref_3d_pair_.first.template cast<T>() + t - targ_3d_pair_.first.template cast<T>());
    Eigen::Matrix<T, 3, 1> res2 = (I - (d * d.transpose()).eval()) * (R * ref_3d_pair_.second.template cast<T>()+ t - targ_3d_pair_.second.template cast<T>());



    residual[0] = res1[0];
    residual[1] = res1[1];
    residual[2] = res1[2];
    residual[3] = res2[0];
    residual[4] = res2[1];
    residual[5] = res2[2];
    return true;
}



LineReprojectionFunctor::LineReprojectionFunctor(const std::pair<Eigen::Vector2d,Eigen::Vector2d> & point_pair, const Eigen::Matrix3d& intrinsic, const Eigen::Matrix<double,6,1>& plucker_ref,
const std::pair<Eigen::Vector3d, Eigen::Vector3d> &ref_3d_pair)
: point_pair_(point_pair),  plucker_ref_(plucker_ref), ref_3d_pair_(ref_3d_pair){
    K_l << intrinsic(1,1), 0, 0, 0, intrinsic(0,0), 0 , -intrinsic(0,2) * intrinsic(1,1), -intrinsic(1,2) * intrinsic(0,0),  intrinsic(0,0) * intrinsic(1,1);
}
    
template <typename T>
bool LineReprojectionFunctor::operator()(const T* const rotvec, const T* const translation, T* residual) const {
   Eigen::Matrix<T, 3, 1> t(translation[0], translation[1], translation[2]);

    // Angle Axis version
    Eigen::Matrix<T, 3, 1> rotation_vector(rotvec[0], rotvec[1], rotvec[2]);
    T angle = rotation_vector.norm();
    Eigen::Matrix<T, 3, 1> axis = rotation_vector.normalized();
     Eigen::AngleAxis<T> angle_axis(angle, axis);

    // Convert angle-axis to rotation matrix
    Eigen::Matrix<T, 3, 3> R = angle_axis.toRotationMatrix();

    // Compute skew symmetric matrix for t
    Eigen::Matrix<T, 3, 3> t_skew;
    t_skew << T(0), -t(2), t(1),
                t(2), T(0), -t(0),
                -t(1), t(0), T(0);



    // Projection using camera intrinsics
    Eigen::Matrix<T, 3, 1> projectedLine = K_l.template cast<T>() * (R*plucker_ref_.tail(3).template cast <T>() + t_skew * R *plucker_ref_.head(3).template cast <T>() ) ;
    // Compute the distance between the projected line and the point
    T numerator1 = ceres::abs(projectedLine.dot(point_pair_.first.homogeneous().template cast<T>()));
    T numerator2 = ceres::abs(projectedLine.dot(point_pair_.second.homogeneous().template cast<T>()));

    residual[0] = numerator1 / projectedLine.head(2).norm();
    residual[1] = numerator2 / projectedLine.head(2).norm();
    return true;
}

// main function
double optimize_line_reprojection(const std::vector<line_feature> & lines_opt, const Eigen::Matrix3d & intrinsic, 
const Eigen::Matrix3d & R_init,  const Eigen::Vector3d & t_init, Eigen::Matrix3d & R_opt, Eigen::Vector3d & t_opt){
    
    // set initial pose
    Sophus::SE3d se3_transform(R_init, t_init);
    Eigen::Matrix<double, 6, 1> initial_pose = se3_transform.log();


    ceres::Problem problem;
    int line_num = lines_opt.size();

    problem.AddParameterBlock(initial_pose.data(), 3); // Rotation (angle-axis)
    problem.AddParameterBlock(initial_pose.data() + 3, 3); // Translation

    for (int i = 0; i < line_num; i++){
        if(lines_opt.at(i).case_ == 0){
            auto point_3d_ref = lines_opt.at(i).ref_3d_pair_;
            auto point_3d_targ = lines_opt.at(i).target_3d_pair_;
            auto plucker_ref_ = lines_opt.at(i).plucker_ref;
            auto plucker_targ_ = lines_opt.at(i).plucker_target;
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<FullPointLineFunctor, 6, 3,3>(
            new FullPointLineFunctor(plucker_ref_,plucker_targ_,point_3d_ref,point_3d_targ));        
            ceres::LossFunction* loss_function = new ceres::CauchyLoss(2.0);
 

            problem.AddResidualBlock(cost_function, loss_function, initial_pose.data(), initial_pose.data()+3);
        }
        // Do not use PnL Case
        else{
            // auto point_pair = lines_opt.at(i).target_img_2d_pair_;
            // auto point_3d_ref = lines_opt.at(i).ref_3d_pair_;
            // auto plucker_ref_ = lines_opt.at(i).plucker_ref;
            // ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<LineReprojectionFunctor, 2, 3,3>(
            // new LineReprojectionFunctor(point_pair, intrinsic, plucker_ref_,point_3d_ref));        
            // // ceres::CostFunction* cost_function =   new LinePointDistanceCostFunction(point_pair, intrinsic, plucker);
            // ceres::LossFunction* loss_function = new ceres::CauchyLoss(2.0);
            // problem.AddResidualBlock(cost_function, loss_function, initial_pose.data(), initial_pose.data()+3);
        }
    }



    
    // Configure solver

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    // Solve
    // Add bound
    problem.SetParameterLowerBound(initial_pose.data(), 0,-0.02); // For first element
    problem.SetParameterLowerBound(initial_pose.data(), 2,-0.02); // For last element
    problem.SetParameterUpperBound(initial_pose.data(), 0,0.02); // For first element
    problem.SetParameterUpperBound(initial_pose.data(), 2,0.02); // For last element
    ceres::Solve(options, &problem, &summary);


    Eigen::Matrix<double, 3, 1> rotation_vector(initial_pose[0], initial_pose[1], initial_pose[2]);
    double angle = rotation_vector.norm();
    Eigen::Matrix<double, 3, 1> axis = rotation_vector.normalized();
    Eigen::AngleAxisd angle_axis(angle, axis);

    // Convert angle-axis to rotation matrix
    R_opt = angle_axis.toRotationMatrix();
    // Rodrigues' rotation formula
    t_opt = initial_pose.tail<3>();
    return summary.final_cost;
}


// The cost functor for the least squares problem


LineCost::LineCost(const Eigen::Vector3d& p) : point(p) {}

template <typename T>
bool LineCost::operator()(const T* const line_params, T* residual) const {
    Eigen::Matrix<T, 3, 1> a(line_params[0], line_params[1], line_params[2]);
    Eigen::Matrix<T, 3, 1> b(line_params[3], line_params[4], line_params[5]);
    Eigen::Matrix<T, 3, 1> P = point.cast<T>();

    // Compute the component of the vector from A to P that is orthogonal to B
    Eigen::Matrix<T, 3, 1> AP_orthogonal = (P - a) - (P - a).dot(b) * b;
    
    // The residual is the norm of the orthogonal component (distance from P to the line)
    residual[0] = AP_orthogonal[0];
    residual[1] = AP_orthogonal[1];
    residual[2] = AP_orthogonal[2];
    return true;
}


double fit_line(const std::vector<Eigen::Vector3d> & inliers, Eigen::Vector3d & a, Eigen::Vector3d & b, Eigen::Vector3d & a_opt, Eigen::Vector3d& b_opt) {
    // Sample inlier points

    Eigen::MatrixXd line_params(6,1);
    line_params.block(0,0,3,1) = a;   // Initial values for a
    line_params.block(3,0,3,1) = b;   // Initial values for b
    ceres::Problem problem;

    for (const auto& point : inliers) {
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<LineCost,3,6>(new LineCost(point));
        problem.AddResidualBlock(cost_function, nullptr, line_params.data());   
        }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    a_opt = line_params.block(0,0,3,1);
    b_opt = line_params.block(3,0,3,1);
    
}

bool ransac_optimize(const std::vector<Eigen::Vector3d>& points, int num_iterations, double distance_threshold
,std::pair<Eigen::Vector3d, Eigen::Vector3d> & result) {

    if(points.size() < 10) return false;
    std::vector<Eigen::Vector3d> inliers;
    std::pair<Eigen::Vector3d, Eigen::Vector3d> best_line;
    int max_inliers = 0;
    for (int i = 0; i < num_iterations; ++i) {
        std::vector<Eigen::Vector3d> inliers_temp;
        auto [first, second] = randomSampleLine(points);
        std::pair<Eigen::Vector3d, Eigen::Vector3d> line = fitLine(first, second);
        int num_inliers = 0;
        for (const Eigen::Vector3d& point : points) {
            if (calculateDistance(line, point) < distance_threshold) {
                ++num_inliers;
                inliers_temp.push_back(point);
            }
        }

        if (num_inliers > max_inliers) {
            max_inliers = num_inliers;
            best_line = line;
            inliers = inliers_temp;
        }
    }

    
    double inlier_ratio = (double) max_inliers / (double) points.size();
    // 0.6 by default
    if(inlier_ratio < 0.8) {
        return false;}
    else{
        result = std::make_pair(best_line.first , best_line.second); 

        return true;
    } 
}