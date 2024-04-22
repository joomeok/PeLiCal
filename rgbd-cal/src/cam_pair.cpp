#include "cam_pair.hpp"

void camera_pair::add_images(const cv::Mat & ref_rgb_img, const cv::Mat& target_rgb_img,const cv::Mat & ref_depth_img, const cv::Mat& target_depth_img){
    current_ref_rgb = ref_rgb_img;
    current_target_rgb = target_rgb_img;
    image_num++;
}

void camera_pair::add_line_features(std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> ref_p_pairs, std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> target_p_pairs , cv::Mat ref_depth_img_, cv::Mat target_depth_img_){
    cv::Mat ref_normalizedDepth;
    cv::normalize(ref_depth_img_, ref_normalizedDepth, 0, 255, cv::NORM_MINMAX);
    cv::Mat ref_depthImage8bit;
    ref_normalizedDepth.convertTo(ref_depthImage8bit, CV_8U);

    cv::Mat target_normalizedDepth;
    cv::normalize(target_depth_img_, target_normalizedDepth, 0, 255, cv::NORM_MINMAX);
    cv::Mat target_depthImage8bit;
    target_normalizedDepth.convertTo(target_depthImage8bit, CV_8U);
    // Perform RANSAC of 3D line fitting at reference image

    for(int i = 0; i <(int) ref_p_pairs.size(); i++){
        // Iterate on 2D line of reference image & compute 3D coordinate
        cv::LineIterator iter_ref(ref_depthImage8bit, cv::Point(ref_p_pairs.at(i).first(0,0),ref_p_pairs.at(i).first(1,0))
        , cv::Point(ref_p_pairs.at(i).second(0,0),ref_p_pairs.at(i).second(1,0)));

        std::vector<Eigen::Vector3d> points_3d_ref;         

        for(int j = 0 ; j < iter_ref.count; ++iter_ref, ++j){
            auto depth = ref_depth_img_.at<unsigned short>(iter_ref.pos());
            if(depth > 0){
                Eigen::Vector3d uv; 
                uv << iter_ref.pos().x , iter_ref.pos().y, 1 ;
                points_3d_ref.push_back(depth * source_K_.inverse() * uv);
            }
        }

        // Iterate on 2D line of target image & compute 3D coordinate

        cv::LineIterator iter_target(target_depthImage8bit, cv::Point(target_p_pairs.at(i).first(0,0),target_p_pairs.at(i).first(1,0))
        , cv::Point(target_p_pairs.at(i).second(0,0),target_p_pairs.at(i).second(1,0)));


        std::vector<Eigen::Vector3d> points_3d_target;

        for(int j = 0 ; j < iter_target.count; ++iter_target, ++j){
            auto depth = target_depth_img_.at<unsigned short>(iter_target.pos());
            if(depth > 0){
                Eigen::Vector3d uv; 
                uv << iter_target.pos().x , iter_target.pos().y, 1 ;
                points_3d_target.push_back(depth * target_K_.inverse() * uv);
            }
        }

        std::pair<Eigen::Vector3d, Eigen::Vector3d> best_ref_line;
        std::pair<Eigen::Vector3d, Eigen::Vector3d> best_target_line;
        // Fit 3D line by RANSAC
        bool ransac_result_ref = ransac_optimize(points_3d_ref, 10, 3, best_ref_line);
        bool ransac_result_target = ransac_optimize(points_3d_target, 10, 3, best_target_line);
        // Add only one line
        if(i > 0) break;
        // PnL Case
        if(ransac_result_ref == true && ransac_result_target == false) {
            line_feature new_line = line_feature(ref_p_pairs.at(i), target_p_pairs.at(i), best_ref_line, best_target_line,image_num-1,1);
            int line_num = lines_full.size() + lines_pnl.size();
            Eigen::Vector3d d_r = new_line.plucker_ref.head(3);

            if(line_num<3){
                Eigen::MatrixXd V_r(3,line_num+1);
                bool skip = false; 
                for(int k=0; k < lines_full.size(); k++){
                        double angle = (180.0 / M_PI) * std::acos( d_r.dot(lines_full.at(k).plucker_ref.head(3)) / (d_r.norm() * lines_full.at(k).plucker_ref.head(3).norm()));
                        if( angle < 5 ){
                            skip = true;
                            break;
                        }
                }  
                for(int k=0; k < lines_pnl.size(); k++){
                        double angle = (180.0 / M_PI) * std::acos( d_r.dot(lines_pnl.at(k).plucker_ref.head(3)) / (d_r.norm() * lines_pnl.at(k).plucker_ref.head(3).norm()));
                        if( angle < 5 ){
                            skip = true;
                            break;
                        }
                }  

                if(skip == true) continue;

                for(int k=0; k < lines_full.size(); k++){
                    V_r.col(k) = lines_full.at(k).plucker_ref.head(3);
                }  

                for(int k=0; k < lines_pnl.size(); k++){
                    V_r.col(lines_full.size() + k) = lines_pnl.at(k).plucker_ref.head(3);
                }  
                V_r.col(lines_full.size() + lines_pnl.size()) = d_r;

                Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(V_r);

                if(lu_decomp.rank() == V_r.cols()) {
                    new_line.multiply_K(target_K_); 
                    lines_pnl.push_back(new_line);
                    lines_opt.push_back(new_line);
                    used_img_id.push_back(image_num-1);
                }
            }

            else{
                double d_to_so3 = check_rotation(new_line);
                if(d_to_so3 > 0){
                    so3_distance = d_to_so3;
                    new_line.multiply_K(target_K_); 
                    lines_pnl.push_back(new_line);
                    lines_opt.push_back(new_line);
                    used_img_id.push_back(image_num-1);
                    
                }
            }

        }
        // Add to Full 3D case
        else if(ransac_result_ref == true && ransac_result_target == true)  {
        
            line_feature new_line = line_feature(ref_p_pairs.at(i), target_p_pairs.at(i), best_ref_line, best_target_line,image_num-1,0);

            Eigen::Vector3d d_r = new_line.plucker_ref.head(3);
            Eigen::Vector3d d_t = new_line.plucker_target.head(3);
            int line_num = lines_full.size() + lines_pnl.size();
            
            // Abandon the line if direction is almost similar to any lines included in the system.
            // If a number of line is below 3, check the rank of V_r add it if the matrix is full rank.t
            if(line_num<3){
                Eigen::MatrixXd V_r(3,line_num+1);
                bool skip = false; 
                for(int k=0; k < lines_full.size(); k++){
                        double angle = (180.0 / M_PI) * std::acos( d_r.dot(lines_full.at(k).plucker_ref.head(3)) / (d_r.norm() * lines_full.at(k).plucker_ref.head(3).norm()));
                        if( angle < 5 ){
                            skip = true;
                            break;
                        }
                }  
                for(int k=0; k < lines_pnl.size(); k++){
                        double angle = (180.0 / M_PI) * std::acos( d_r.dot(lines_pnl.at(k).plucker_ref.head(3)) / (d_r.norm() * lines_pnl.at(k).plucker_ref.head(3).norm()));
                        if( angle < 5 ){
                            skip = true;
                            break;
                        }
                }  

                if(skip == true) continue;

                for(int k=0; k < lines_full.size(); k++){
                    V_r.col(k) = lines_full.at(k).plucker_ref.head(3);
                }  

                for(int k=0; k < lines_pnl.size(); k++){
                    V_r.col(lines_full.size() + k) = lines_pnl.at(k).plucker_ref.head(3);
                }  
                V_r.col(lines_full.size() + lines_pnl.size()) = d_r;

                Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(V_r);

                if(lu_decomp.rank() == V_r.cols()) {
                    new_line.multiply_K(target_K_); 
                    lines_full.push_back(new_line);
                    lines_opt.push_back(new_line);
                    used_img_id.push_back(image_num-1);
                }
            }
            // If a number of line is over 3, add the line if distance to SO(3) manifold becomes closer.
            else{
                double d_to_so3 = check_rotation(new_line);
                if(d_to_so3 > 0){
                    so3_distance = d_to_so3;
                    new_line.multiply_K(target_K_); 
                    lines_full.push_back(new_line);
                    lines_opt.push_back(new_line);
                    used_img_id.push_back(image_num-1);
                }
            }
        }
    }

}

bool camera_pair::perform_PnL(){
    auto [A, B] = construct_A_B(lines_full,lines_pnl);
    Eigen::MatrixXd K = A - B * (B.transpose() * B).inverse() * B.transpose() * A;
    auto solution = new Eigen::Matrix<double,3,8>;
    int K_size = 8* lines_full.size() + 2 * lines_pnl.size();

    int n_roots = poselib::re3q3::re3q3(compute_C(K, K_size),solution,true);

    double best_cost = 1e3;
    std::vector<line_feature> best_inliers;
    
    if(n_roots == 0) return false;
    

    for(int i = 0; i< n_roots ; i++){
        double s1 = (*solution)(0,i);
        double s2 = (*solution)(1,i);
        double s3 = (*solution)(2,i);
        auto R = restore_R(s1,s2,s3,A,B);
        auto t = restore_t(s1,s2,s3,A,B);
        Eigen::Matrix3d R_opt = Eigen::Matrix3d::Zero(); 
        Eigen::Vector3d t_opt = Eigen::Vector3d::Zero();

        double cost = optimize_line_reprojection(lines_opt, target_K_, R, t, R_opt,t_opt);
        Sophus::SO3d so3_transform(R_opt);
        Eigen::Quaterniond quaternion = so3_transform.unit_quaternion();
        Eigen::Vector3d euler_angles = (180 / M_PI) * quaternion.toRotationMatrix().eulerAngles(2, 1, 0);
    
        Eigen::AngleAxisd angleAxis(quaternion);
        double angle = angleAxis.angle();
        Eigen::Vector3d axis = angleAxis.axis();
        Eigen::Vector3d rotationVector = angle * axis;
        std::vector<line_feature> best_pnl, best_full;
        // Return true if solution t exists, compute rotation and translation again with inliers.
        bool t_converged = check_translation(R_opt, best_full,best_pnl);

        if(t_converged  && (lines_full.size()+lines_pnl.size()) > minimum_line_num && cost/lines_opt.size() < convergency_cost){
            double roll = euler_angles(0);
            double pitch = euler_angles(1);
            double yaw = euler_angles(2);
            std::cout << " -------------------------------------------" << std::endl;
            std::cout << "Rotation Vector: (rad)" << "\n";
            std::cout << rotationVector.transpose() << std::endl;
                // std::cout << roll << " " << pitch << " " << yaw << std::endl;
            std::cout << "\n";
            std::cout << "Translation optimized value (mm): " << "\n";
            std::cout << t_opt(0) << " " << t_opt(1) << " " << t_opt(2) << std::endl;
            std::cout << "\n";
            std::cout << "Solution converged! with cost : "  << cost/lines_opt.size() << std::endl;
            return true;
        } 
        else if(t_converged && cost/lines_opt.size() > convergency_cost){
            this->lines_full = best_full;
            this->lines_pnl = best_pnl;

            std::cout << "Solution converged! but cost is not small enough, cost : " << cost/lines_opt.size() <<std::endl;
            return false;
        }
    }
    
    delete solution;

    return false;
}

double camera_pair::check_rotation(const line_feature & line){
    int pnl_line_num = lines_pnl.size();
    int full_line_num = lines_full.size();
    std::cout << "PnL line num : " << pnl_line_num << " " << "full line num : " << full_line_num << std::endl;

    // If every line is pnl case and new line is full case, directly add it and return new distance 
    if(full_line_num == 0 && line.case_ == 0){
        Eigen::MatrixXd A_new(pnl_line_num + 3,9), B_new(pnl_line_num + 3,1);
        Eigen::MatrixXd a_full(3,9);
        Eigen::MatrixXd b_full(3,1);

        for(int i = 0; i < pnl_line_num; i++){
            Eigen::MatrixXd a_pnl(1,9);
            Eigen::Vector3d P_l = (P.transpose() * lines_pnl.at(i).line_coord).head(3);
            double d_1x = lines_pnl.at(i).plucker_ref(0,0);
            double d_1y = lines_pnl.at(i).plucker_ref(1,0);
            double d_1z = lines_pnl.at(i).plucker_ref(2,0);
            double a = P_l(0);
            double b = P_l(1);
            double c = P_l(2);
            a_pnl << a*d_1x, a*d_1y, a*d_1z, b*d_1x, b*d_1y, b*d_1z, c*d_1x,c*d_1y, c*d_1z; 
            A_new.row(i) = a_pnl;
            B_new.row(i).setZero();
        }

        double d_1x = line.plucker_ref(0,0);
        double d_1y = line.plucker_ref(1,0);
        double d_1z = line.plucker_ref(2,0);
        double d_2x = line.plucker_target(0,0);
        double d_2y = line.plucker_target(1,0);
        double d_2z = line.plucker_target(2,0);
        a_full << d_1x, d_1y, d_1z, 0,0,0,0,0,0,
                  0,0,0, d_1x, d_1y, d_1z,0,0,0,
                  0,0,0,0,0,0,d_1x, d_1y, d_1z;
        b_full << d_2x, d_2y, d_2z;
        A_new.block(pnl_line_num,0,3,9) = a_full;
        B_new.block(pnl_line_num,0,3,1) = b_full;
        Eigen::MatrixXd r_new = A_new.colPivHouseholderQr().solve(B_new);
        Eigen::Map<Eigen::MatrixXd> R_new(r_new.data(), 3, 3);
        return distance_SO3(R_new);
    }

    // If every line is pnl case and new line is also pnl case, reject it since we cannot compute distance to SO3
    else if(full_line_num ==0 && line.case_ == 1) return -1;

    // Finally, if there is at least one full case, we can compute distance to so3. 
    else{
        Eigen::MatrixXd A(3*full_line_num + pnl_line_num,9), B(3*full_line_num + pnl_line_num ,1);
        for(int i = 0; i< full_line_num; i++){
                Eigen::MatrixXd a_full(3,9);
                Eigen::MatrixXd b_full(3,1);
                double d_1x = lines_full.at(i).plucker_ref(0,0);
                double d_1y = lines_full.at(i).plucker_ref(1,0);
                double d_1z = lines_full.at(i).plucker_ref(2,0);
                double d_2x = lines_full.at(i).plucker_target(0,0);
                double d_2y = lines_full.at(i).plucker_target(1,0);
                double d_2z = lines_full.at(i).plucker_target(2,0);

                a_full << d_1x, d_1y, d_1z, 0,0,0,0,0,0,
                        0,0,0, d_1x, d_1y, d_1z,0,0,0,
                        0,0,0,0,0,0,d_1x, d_1y, d_1z;
                b_full << d_2x, d_2y, d_2z;
                A.block(3*i,0,3,9) = a_full;
                B.block(3*i,0,3,1) = b_full;
            }

            for(int i = 0; i < pnl_line_num; i++){
                Eigen::MatrixXd a_pnl(1,9);
                Eigen::Vector3d P_l = (P.transpose() * lines_pnl.at(i).line_coord).head(3);
                double d_1x = lines_pnl.at(i).plucker_ref(0,0);
                double d_1y = lines_pnl.at(i).plucker_ref(1,0);
                double d_1z = lines_pnl.at(i).plucker_ref(2,0);
                double a = P_l(0);
                double b = P_l(1);
                double c = P_l(2);
                a_pnl << a*d_1x, a*d_1y, a*d_1z, b*d_1x, b*d_1y, b*d_1z, c*d_1x,c*d_1y, c*d_1z; 
                A.row(3*full_line_num+i) = a_pnl;
                B.row(3*full_line_num+i).setZero();
            }
            Eigen::MatrixXd r = A.colPivHouseholderQr().solve(B);
            Eigen::Map<Eigen::MatrixXd> R_before(r.data(), 3, 3);
            double d_before = distance_SO3(R_before);
            double d_after = 0;

            if(line.case_ == 0){
                    Eigen::MatrixXd A_new(3*full_line_num + pnl_line_num + 3,9), B_new(3*full_line_num + pnl_line_num + 3,1);
                    // Fill A new
                    for(int i = 0; i< full_line_num; i++){
                        Eigen::MatrixXd a_full(3,9);
                        Eigen::MatrixXd b_full(3,1);
                        double d_1x = lines_full.at(i).plucker_ref(0,0);
                        double d_1y = lines_full.at(i).plucker_ref(1,0);
                        double d_1z = lines_full.at(i).plucker_ref(2,0);
                        double d_2x = lines_full.at(i).plucker_target(0,0);
                        double d_2y = lines_full.at(i).plucker_target(1,0);
                        double d_2z = lines_full.at(i).plucker_target(2,0);

                        a_full << d_1x, d_1y, d_1z, 0,0,0,0,0,0,
                                0,0,0, d_1x, d_1y, d_1z,0,0,0,
                                0,0,0,0,0,0,d_1x, d_1y, d_1z;
                        b_full << d_2x, d_2y, d_2z;
                        A_new.block(3*i,0,3,9) = a_full;
                        B_new.block(3*i,0,3,1) = b_full;
                }

                for(int i = 0; i < pnl_line_num; i++){
                    Eigen::MatrixXd a_pnl(1,9);
                    Eigen::Vector3d P_l = (P.transpose() * lines_pnl.at(i).line_coord).head(3);
                    double d_1x = lines_pnl.at(i).plucker_ref(0,0);
                    double d_1y = lines_pnl.at(i).plucker_ref(1,0);
                    double d_1z = lines_pnl.at(i).plucker_ref(2,0);
                    double a = P_l(0);
                    double b = P_l(1);
                    double c = P_l(2);
                    a_pnl << a*d_1x, a*d_1y, a*d_1z, b*d_1x, b*d_1y, b*d_1z, c*d_1x,c*d_1y, c*d_1z; 
                    A_new.row(3*full_line_num+i) = a_pnl;
                    B_new.row(3*full_line_num+i).setZero();
                }
                    Eigen::MatrixXd a_full(3,9);
                    Eigen::MatrixXd b_full(3,1);
                    double d_1x = line.plucker_ref(0,0);
                    double d_1y = line.plucker_ref(1,0);
                    double d_1z = line.plucker_ref(2,0);
                    double d_2x = line.plucker_target(0,0);
                    double d_2y = line.plucker_target(1,0);
                    double d_2z = line.plucker_target(2,0);
                    a_full << d_1x, d_1y, d_1z, 0,0,0,0,0,0,
                            0,0,0, d_1x, d_1y, d_1z,0,0,0,
                            0,0,0,0,0,0,d_1x, d_1y, d_1z;
                    b_full << d_2x, d_2y, d_2z;
                    A_new.block(3*full_line_num + pnl_line_num,0,3,9) = a_full;
                    B_new.block(3*full_line_num + pnl_line_num,0,3,1) = b_full;
                    Eigen::MatrixXd r_new = A_new.colPivHouseholderQr().solve(B_new);
                    Eigen::Map<Eigen::MatrixXd> R_new(r_new.data(), 3, 3);
                    d_after = distance_SO3(R_new);
                }
            else{
                Eigen::MatrixXd A_new(3*full_line_num + pnl_line_num + 1,9), B_new(3*full_line_num + pnl_line_num + 1,1);
                // Fill A new
                for(int i = 0; i< full_line_num; i++){
                    Eigen::MatrixXd a_full(3,9);
                    Eigen::MatrixXd b_full(3,1);
                    double d_1x = lines_full.at(i).plucker_ref(0,0);
                    double d_1y = lines_full.at(i).plucker_ref(1,0);
                    double d_1z = lines_full.at(i).plucker_ref(2,0);
                    double d_2x = lines_full.at(i).plucker_target(0,0);
                    double d_2y = lines_full.at(i).plucker_target(1,0);
                    double d_2z = lines_full.at(i).plucker_target(2,0);

                    a_full << d_1x, d_1y, d_1z, 0,0,0,0,0,0,
                            0,0,0, d_1x, d_1y, d_1z,0,0,0,
                            0,0,0,0,0,0,d_1x, d_1y, d_1z;
                    b_full << d_2x, d_2y, d_2z;
                    A_new.block(3*i,0,3,9) = a_full;
                    B_new.block(3*i,0,3,1) = b_full;
                }

            for(int i = 0; i < pnl_line_num; i++){
                    Eigen::MatrixXd a_pnl(1,9);
                    Eigen::Vector3d P_l = (P.transpose() * lines_pnl.at(i).line_coord).head(3);
                    double d_1x = lines_pnl.at(i).plucker_ref(0,0);
                    double d_1y = lines_pnl.at(i).plucker_ref(1,0);
                    double d_1z = lines_pnl.at(i).plucker_ref(2,0);
                    double a = P_l(0);
                    double b = P_l(1);
                    double c = P_l(2);
                    a_pnl << a*d_1x, a*d_1y, a*d_1z, b*d_1x, b*d_1y, b*d_1z, c*d_1x,c*d_1y, c*d_1z; 
                    A_new.row(3*full_line_num+i) = a_pnl;
                    B_new.row(3*full_line_num+i).setZero();
                }
                Eigen::MatrixXd a_pnl(1,9);
                Eigen::Vector3d P_l = (P.transpose() * line.line_coord).head(3);

                double d_1x = line.plucker_ref(0,0);
                double d_1y = line.plucker_ref(1,0);
                double d_1z = line.plucker_ref(2,0);
                double a = P_l(0);
                double b = P_l(1);
                double c = P_l(2);
                a_pnl << a*d_1x, a*d_1y, a*d_1z, b*d_1x, b*d_1y, b*d_1z, c*d_1x, c*d_1y, c*d_1z; 
                A_new.row(3*full_line_num + pnl_line_num) = a_pnl;
                B_new.row(3*full_line_num + pnl_line_num).setZero();
                Eigen::MatrixXd r_new;
                r_new = A_new.colPivHouseholderQr().solve(B_new);
                Eigen::Map<Eigen::MatrixXd> R_new(r_new.data(), 3, 3);
                d_after = distance_SO3(R_new);
            }
            if(d_after < d_before) return d_after;
            else if (d_after < 1.2) return d_before;
            else return -1;
    }

}


double camera_pair::compute_so3_distance(std::vector<line_feature> lines_full,std::vector<line_feature> lines_pnl ){
    int pnl_line_num = lines_pnl.size();
    int full_line_num = lines_full.size();

    Eigen::MatrixXd A(3*full_line_num + pnl_line_num,9), B(3*full_line_num + pnl_line_num,1);
    for(int i = 0; i< full_line_num; i++){
        Eigen::MatrixXd a_full(3,9);
        Eigen::MatrixXd b_full(3,1);
        double d_1x = lines_full.at(i).plucker_ref(0,0);
        double d_1y = lines_full.at(i).plucker_ref(1,0);
        double d_1z = lines_full.at(i).plucker_ref(2,0);
        double d_2x = lines_full.at(i).plucker_target(0,0);
        double d_2y = lines_full.at(i).plucker_target(1,0);
        double d_2z = lines_full.at(i).plucker_target(2,0);

        a_full << d_1x, d_1y, d_1z, 0,0,0,0,0,0,
                  0,0,0, d_1x, d_1y, d_1z,0,0,0,
                  0,0,0,0,0,0,d_1x, d_1y, d_1z;
        b_full << d_2x, d_2y, d_2z;
        A.block(3*i,0,3,9) = a_full;
        B.block(3*i,0,3,1) = b_full;
    }
    for(int i = 0; i < pnl_line_num; i++){
        Eigen::MatrixXd a_pnl(1,9);
        Eigen::Vector3d P_l = (P.transpose() * lines_pnl.at(i).line_coord).head(3);
        double d_1x = lines_pnl.at(i).plucker_ref(0,0);
        double d_1y = lines_pnl.at(i).plucker_ref(1,0);
        double d_1z = lines_pnl.at(i).plucker_ref(2,0);
        double a = P_l(0);
        double b = P_l(1);
        double c = P_l(2);
        a_pnl << a*d_1x, a*d_1y, a*d_1z, b*d_1x, b*d_1y, b*d_1z, c*d_1x,c*d_1y, c*d_1z; 
        A.row(3*full_line_num+i) = a_pnl;
        B.row(3*full_line_num+i).setZero();
    }
    Eigen::MatrixXd r = A.colPivHouseholderQr().solve(B);
    Eigen::Map<Eigen::MatrixXd> R_before(r.data(), 3, 3);
    double distance = distance_SO3(R_before);
    return distance;
}

bool camera_pair::check_translation(const Eigen::Matrix3d & R, std::vector<line_feature> & result_full, std::vector<line_feature> & result_pnl){
    int line_num = lines_full.size() + lines_pnl.size();
    std::vector<line_feature> mergedLines(lines_full.begin(), lines_full.end());
    mergedLines.insert(mergedLines.end(), lines_pnl.begin(), lines_pnl.end());

    std::vector<Eigen::Vector3d> EquiD_pts;
    std::vector<Eigen::Vector3d> center_pts;
    std::vector<std::pair<int,int>> EquiD_line_idxs;
    Eigen::Matrix3d K_l;
    K_l << target_K_(1,1), 0, 0, 0, target_K_(0,0), 0 , -target_K_(0,2) * target_K_(1,1), -target_K_(1,2) * target_K_(0,0),  target_K_(0,0) * target_K_ (1,1);
    // compute equi-distance points
    for (int i = 0; i < line_num - 1; ++i) {
        Eigen::Vector3d P1,d1;
        if(mergedLines.at(i).case_ == 0){
            Eigen::Vector3d a = R*mergedLines.at(i).plucker_ref.head(3);
            Eigen::Vector3d b = R* mergedLines.at(i).plucker_ref.tail(3) - mergedLines.at(i).plucker_target.tail(3);
            P1 = b.cross(a) / a.squaredNorm(); 
            d1 = a;
        }
        else{
            Eigen::Vector3d a1 = skewSymmetric(R *  mergedLines.at(i).plucker_ref.head(3)) * K_l.transpose() * mergedLines.at(i).target_img_2d_pair_.first.homogeneous();
            double b1 = mergedLines.at(i).target_img_2d_pair_.first.homogeneous().transpose() * K_l * R * mergedLines.at(i).plucker_ref.tail(3);
            b1 = b1 / a1.norm();
            a1.normalize();

            Eigen::Vector3d a2 = skewSymmetric(R *  mergedLines.at(i).plucker_ref.head(3)) * K_l.transpose() * mergedLines.at(i).target_img_2d_pair_.second.homogeneous();

            double b2 = mergedLines.at(i).target_img_2d_pair_.second.homogeneous().transpose() * K_l * R * mergedLines.at(i).plucker_ref.tail(3);
            b2 = b2 / a2.norm();
            a2.normalize();
            
            Eigen::Matrix<double,2,1> temp_b;
            Eigen::Matrix2d temp_a;
            temp_b << b1, b2;
            temp_a << a1(1), a1(2), a2(1), a2(2);

            if(temp_a.determinant() != 0){
                Eigen::Vector2d pt_on_x = temp_a.colPivHouseholderQr().solve(temp_b);
                P1 << 0, pt_on_x(0) , pt_on_x(1);
            }
            else{
                Eigen::Matrix2d temp_a2;
                temp_a2 << a1(0), a1(2), a2(0), a2(2);
                Eigen::Vector2d pt_on_y = temp_a2.colPivHouseholderQr().solve(temp_b);
                P1 << pt_on_y(0), 0, pt_on_y(1);
            }
            d1 = a1.cross(a2).normalized();
        }

        for (int j = i + 1; j < mergedLines.size(); ++j) {
            Eigen::Vector3d P2, d2;
            if(mergedLines.at(j).case_ == 0){
                Eigen::Vector3d a = R*mergedLines.at(j).plucker_ref.head(3);
                Eigen::Vector3d b = R* mergedLines.at(j).plucker_ref.tail(3) - mergedLines.at(j).plucker_target.tail(3);
                P2 = b.cross(a) / a.squaredNorm();
                d2 = a;
            }
            else if(mergedLines.at(j).case_ == 1){
                Eigen::Vector3d a1 = skewSymmetric(R *  mergedLines.at(j).plucker_ref.head(3)) * K_l.transpose() * mergedLines.at(j).target_img_2d_pair_.first.homogeneous();
                double b1 = mergedLines.at(j).target_img_2d_pair_.first.homogeneous().transpose() * K_l * R * mergedLines.at(j).plucker_ref.tail(3);
                
                b1 = b1 / a1.norm();
                a1.normalize();
                Eigen::Vector3d a2 = skewSymmetric(R *  mergedLines.at(j).plucker_ref.head(3)) * K_l.transpose() * mergedLines.at(j).target_img_2d_pair_.second.homogeneous();                
                double b2 = mergedLines.at(j).target_img_2d_pair_.second.homogeneous().transpose() * K_l * R * mergedLines.at(j).plucker_ref.tail(3);
                b2 = b2 / a2.norm();
                a2.normalize();
                
                Eigen::Matrix<double,2,1> temp_b;
                Eigen::Matrix2d temp_a;
                temp_b << b1, b2;
                temp_a << a1(1), a1(2), a2(1), a2(2);

                if(temp_a.determinant() != 0){
                    Eigen::Vector2d pt_on_x = temp_a.colPivHouseholderQr().solve(temp_b);
                    P2 << 0, pt_on_x(0), pt_on_x(1);
                }
                else{
                    Eigen::Matrix2d temp_a2;
                    temp_a2 << a1(0), a1(2), a2(0), a2(2);
                    Eigen::Vector2d pt_on_y = temp_a2.colPivHouseholderQr().solve(temp_b);

                    P2 << pt_on_y(0), 0 , pt_on_y(1);
                }
                d2 = a1.cross(a2).normalized();
            }

            EquiD_pts.push_back(computeEquidistantPoint(P1,d1,P2,d2));
            EquiD_line_idxs.push_back(std::make_pair(i,j));
        }
    }


    int best_num = 0;
    Eigen::Vector3d best_center;
    std::set<int> inliers;

    for(const auto & center : EquiD_pts){ 
        std::set<int> temp_inlier_idx;
        int temp_inlier_num = 0;
        double distance = 0;

        for (int k = 0; k< mergedLines.size(); k++){ 
            // auto equi_pt = EquiD_pts.at(k);
            if(mergedLines.at(k).case_ == 0){
                Eigen::Vector3d a = R*mergedLines.at(k).plucker_ref.head(3);
                Eigen::Vector3d b = R* mergedLines.at(k).plucker_ref.tail(3) - mergedLines.at(k).plucker_target.tail(3);
                Eigen::Vector3d temp_P = (b).cross(a) / a.squaredNorm();
                Eigen::Vector3d temp_d = a;
                distance = point_to_line_d(temp_d,temp_P,center);
            }
            else{
                Eigen::Vector3d a1 = skewSymmetric(R *  mergedLines.at(k).plucker_ref.head(3)) * K_l.transpose() * mergedLines.at(k).target_img_2d_pair_.first.homogeneous();

                double b1 = mergedLines.at(k).target_img_2d_pair_.first.homogeneous().transpose() * K_l * R * mergedLines.at(k).plucker_ref.tail(3);
                b1 = b1 / a1.norm();
                a1.normalize();
                Eigen::Vector3d a2 = skewSymmetric(R*  mergedLines.at(k).plucker_ref.head(3)) * K_l.transpose() * mergedLines.at(k).target_img_2d_pair_.second.homogeneous();

                double b2 = mergedLines.at(k).target_img_2d_pair_.second.homogeneous().transpose() * K_l * R * mergedLines.at(k).plucker_ref.tail(3);
                b2 = b2 / a2.norm();
                a2.normalize();
                Eigen::Matrix<double,2,1> temp_b;
                Eigen::Matrix2d temp_a;
                temp_b << b1, b2;
                temp_a << a1(1), a1(2), a2(1), a2(2);
                Eigen::Vector3d temp_P, temp_d;
                if(temp_a.determinant() != 0){
                    Eigen::Vector2d pt_on_x = temp_a.colPivHouseholderQr().solve(temp_b);
                    temp_P << 0, pt_on_x(0), pt_on_x(1);
                }
                else{
                    Eigen::Matrix2d temp_a2;
                    temp_a2 << a1(0), a1(2), a2(0), a2(2);
                    Eigen::Vector2d pt_on_y = temp_a2.colPivHouseholderQr().solve(temp_b);

                    temp_P << pt_on_y(0), 0 , pt_on_y(1);
                }
                temp_d = a1.cross(a2).normalized();
                distance = point_to_line_d(temp_d,temp_P,center);
            }

            if(distance < eps){
                temp_inlier_num++;
                temp_inlier_idx.insert(k);
            }
        }

        if(temp_inlier_num > best_num){
            best_num = temp_inlier_num;
            best_center = center;
            inliers = temp_inlier_idx;
        }
    }

    std::vector<line_feature> temp_lines;

    bool closer = false;
    double best_cost = 1e3;

    if(best_num > minimum_line_num){

        std::vector<std::pair<Eigen::Vector3d,Eigen::Vector3d>> lines_;
        for (int k = 0; k< mergedLines.size(); k++){ 
            
            if(mergedLines.at(k).case_ == 0){
                Eigen::Vector3d a = R*mergedLines.at(k).plucker_ref.head(3);
                Eigen::Vector3d b = R* mergedLines.at(k).plucker_ref.tail(3) - mergedLines.at(k).plucker_target.tail(3);
                Eigen::Vector3d temp_P = (b).cross(a) / a.squaredNorm();
                Eigen::Vector3d temp_d = a;
                lines_.push_back(std::make_pair(temp_d,temp_P));
            }
            else{
                Eigen::Vector3d a1 = skewSymmetric(R *  mergedLines.at(k).plucker_ref.head(3)) * K_l.transpose() * mergedLines.at(k).target_img_2d_pair_.first.homogeneous();


                double b1 = mergedLines.at(k).target_img_2d_pair_.first.homogeneous().transpose() * K_l * R * mergedLines.at(k).plucker_ref.tail(3);
                b1 = b1 / a1.norm();
                a1.normalize();
                Eigen::Vector3d a2 = skewSymmetric(R *  mergedLines.at(k).plucker_ref.head(3)) * K_l.transpose() * mergedLines.at(k).target_img_2d_pair_.second.homogeneous(); 
                double b2 = mergedLines.at(k).target_img_2d_pair_.second.homogeneous().transpose() * K_l * R * mergedLines.at(k).plucker_ref.tail(3);
                b2 = b2 / a2.norm();
                a2.normalize();
                Eigen::Matrix<double,2,1> temp_b;
                Eigen::Matrix2d temp_a;
                temp_b << b1, b2;
                temp_a << a1(1), a1(2), a2(1), a2(2);
                Eigen::Vector3d temp_P, temp_d;
                if(temp_a.determinant() != 0){
                    Eigen::Vector2d pt_on_x = temp_a.colPivHouseholderQr().solve(temp_b);
                    temp_P << 0, pt_on_x(0), pt_on_x(1);
                }
                else{
                    Eigen::Matrix2d temp_a2;
                    temp_a2 << a1(0), a1(2), a2(0), a2(2);
                    Eigen::Vector2d pt_on_y = temp_a2.colPivHouseholderQr().solve(temp_b);

                    temp_P << pt_on_y(0), 0 , pt_on_y(1);
                }
                temp_d = a1.cross(a2).normalized();
                lines_.push_back(std::make_pair(temp_d,temp_P));
            }   
        }
 
        std::vector<line_feature> temp_pnl, temp_full;
        for(const auto inlier_idx : inliers){
                if(mergedLines.at(inlier_idx).case_ == 0){
                    temp_full.push_back(mergedLines.at(inlier_idx));
                }
                else temp_pnl.push_back(mergedLines.at(inlier_idx));
        }

        so3_distance = compute_so3_distance(temp_full, temp_pnl);
        result_full = temp_full;
        result_pnl = temp_pnl;
        return true;
    }
    else return false;
}
