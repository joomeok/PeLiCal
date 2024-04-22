#include "pnp_ral_func.hpp"

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> construct_A_B (const std::vector<line_feature>& lines_full, const std::vector<line_feature>& lines_pnl){
  int n = lines_full.size();
  int m = lines_pnl.size();
  Eigen::MatrixXd A(8*n + 2*m, 10);
  Eigen::MatrixXd B(8*n + 2*m, 3);

for(int i = 0; i < n; i++){
        double d_x = lines_full.at(i).plucker_target(0,0);
        double d_y = lines_full.at(i).plucker_target(1,0);
        double d_z = lines_full.at(i).plucker_target(2,0);
        double m_x = lines_full.at(i).plucker_target(3,0);
        double m_y = lines_full.at(i).plucker_target(4,0);
        double m_z = lines_full.at(i).plucker_target(5,0);


        for(int j = 0; j < 2; j++){
            Eigen::Matrix <double,4,10> a_3d;
            Eigen::Matrix <double,4,3> b_3d;
            double P_x, P_y, P_z;

            if(j == 0){
                P_x = lines_full.at(i).ref_3d_pair_.first(0,0);
                P_y = lines_full.at(i).ref_3d_pair_.first(1,0);
                P_z = lines_full.at(i).ref_3d_pair_.first(2,0);
            }
            else {
                P_x = lines_full.at(i).ref_3d_pair_.second(0,0);
                P_y = lines_full.at(i).ref_3d_pair_.second(1,0);
                P_z = lines_full.at(i).ref_3d_pair_.second(2,0);
            }

            // 3D Constraint  
            a_3d << P_y*d_z-P_z*d_y+m_x, -P_y*d_z-P_z*d_y+m_x, P_y*d_z+P_z*d_y+m_x,
                -2*P_x*d_z, 2*P_x*d_y, 2*P_y*d_y-2*P_z*d_z, 2*P_y*d_y+2*P_z*d_z,
                -2*P_x*d_y, -2*P_x*d_z, -P_y*d_z +P_z*d_y+ m_x,

                P_x*d_z+P_z*d_x+m_y, -P_x*d_z+P_z*d_x+m_y, -P_x*d_z-P_z*d_x+m_y,
                2*P_y*d_z, -2*P_x+2*P_z*d_z, -2*P_y*d_x, -2*P_y*d_x,  2*P_x*d_x+2*P_z*d_z,
                -2*P_y*d_z, P_x*d_z-P_z*d_x+m_y,

                -P_x*d_y-P_y*d_x+m_z, P_x*d_y+P_y*d_x+m_z, P_x*d_y-P_y*d_x+m_z,
                2*P_x*d_x-2*P_y*d_y, -2*P_z*d_y, 2*P_z*d_x, -2*P_z*d_x, -2*P_z*d_y,
                2*P_x*d_x+2*P_y*d_y, -P_x*d_y+P_y*d_x+m_z,

                P_x*m_x-P_y*m_y-P_z*m_z, -P_x*m_x+P_y*m_y-P_z*m_z, -P_x*m_x-P_y*m_y+P_z*m_z,
                2*P_x*m_y+2*P_y*m_x, 2*P_x*m_z+2*P_z*m_x, 2*P_y*m_z+2*P_z*m_y,
                2*P_y*m_z-2*P_z*m_y, -2*P_x*m_z+2*P_z*m_x, 2*P_x*m_y-2*P_y*m_x, P_x*m_x+P_y*m_y+P_z*m_z;
            b_3d << 0, d_z, -d_y, -d_z, 0, d_x, d_y, -d_x, 0, -m_x, -m_y, -m_z;
            // All
            A.block(8*i + 4*j,0,4,10) = 100 * a_3d;
            B.block(8*i + 4*j,0,4,3) = 100 * b_3d;
        } 
    }

  for(int i = 0; i < m; i++){
        Eigen::Vector3d K_multiplied_line = lines_pnl.at(i).K_mutliplied_coord;
        double a = K_multiplied_line(0,0);
        double b = K_multiplied_line(1,0);
        double c = K_multiplied_line(2,0);
    for(int j = 0; j < 2; j++){
        Eigen::Matrix <double,1,10> a_2d;
        Eigen::Matrix <double,1,3> b_2d;
        double P_x, P_y, P_z;

        if(j == 0){
            P_x = lines_pnl.at(i).ref_3d_pair_.first(0,0);
            P_y = lines_pnl.at(i).ref_3d_pair_.first(1,0);
            P_z = lines_pnl.at(i).ref_3d_pair_.first(2,0);
        }
        else {
            P_x = lines_pnl.at(i).ref_3d_pair_.second(0,0);
            P_y = lines_pnl.at(i).ref_3d_pair_.second(1,0);
            P_z = lines_pnl.at(i).ref_3d_pair_.second(2,0);
        }
            // // 2D constraint
            a_2d << P_x*a - P_y*b - P_z*c,
                -P_x*a + P_y*b - P_z*c,
                -P_x*a - P_y*b + P_z*c,
                2*P_x*b + 2*P_y*a,
                2*P_x*c + 2*P_z*a,
                2*P_y*c + 2*P_z*b,
                2*P_y*c - 2*P_z*b,
                -2*P_x*c + 2*P_z*a,
                2*P_x*b - 2*P_y*a,
                P_x*a + P_y*b + P_z*c;
            b_2d << a, b ,c;

            A.row(8*n + 2*i +j) = a_2d;
            B.row(8*n + 2*i +j) = b_2d;
        } 
    }
    
   return std::make_pair(A,B);
}


Eigen::Matrix<double,3,10> compute_C(Eigen::MatrixXd K, int n){
    std::vector<Eigen::MatrixXd> k_n_vec, k_n_bar_vec;
    std::vector<int> r_order;
    int ith_idx = 0;
    for (int i=0; i < 9; i++){
        k_n_vec.push_back(K.col(i));
        if(K.col(ith_idx).squaredNorm() < K.col(i).squaredNorm()){
            ith_idx = i;
        }
    }
    r_order.push_back(ith_idx);

    int jth_idx = 0;
    Eigen::MatrixXd k_i = K.col(ith_idx);
    double jth_norm = 0;
    for (int j=0; j < 9; j++){
        Eigen::MatrixXd k_n = k_n_vec.at(j);
        Eigen::MatrixXd k_n_bar = k_n - (dot(k_i,k_n) / dot(k_i,k_i)) * k_i;
        k_n_bar_vec.push_back(k_n_bar);
        if(j == ith_idx) continue;
        
        if(jth_norm < k_n_bar.squaredNorm()){
            jth_idx = j;
            jth_norm = k_n_bar.squaredNorm();
        }
    }
    r_order.push_back(jth_idx);
    int kth_idx = 0;
    Eigen::MatrixXd k_j_bar = k_n_bar_vec.at(jth_idx);
    double kth_norm = 0;
    for (int k=0; k < 9; k++){
        if(k == ith_idx || k == jth_idx) continue;

        Eigen::MatrixXd k_n_bar = k_n_bar_vec.at(k);
        Eigen::MatrixXd k_n_tilde = k_n_bar - dot(k_j_bar,k_n_bar) / dot(k_j_bar,k_j_bar) * k_j_bar;

        if(kth_norm < k_n_tilde.squaredNorm()){
            kth_idx = k;
            kth_norm = k_n_tilde.squaredNorm();
        }
    }
    r_order.push_back(kth_idx);

    for(int i =0; i<10; i++){
        auto it = find(r_order.begin(), r_order.end(), i);
        if (it == r_order.end()) {
            r_order.push_back(i);
        }
    }

    Eigen::MatrixXd K_3(n,3);
    K_3.col(0) = K.col(ith_idx);
    K_3.col(1) = K.col(jth_idx);
    K_3.col(2) = K.col(kth_idx); 
    Eigen::MatrixXd K_7(n,7);

    for (int i = 3; i<10; i++){
        int n = r_order.at(i);
        K_7.col(i-3) = K.col(n);
    }

    Eigen::Matrix <double,3,10> C;
    C.block(0,0,3,3) = Eigen::Matrix3d::Identity();
    C.block(0,3,3,7) = (K_3.transpose() * K_3).inverse() * K_3.transpose() * K_7;
    Eigen::Matrix <double,3,10> ordered_C;
    // Reorder C in the order of s1^2, s2^2, s3^2, s1s2, s1s3, s2s3, 1
    for(int i = 0; i < 10; i++){
        ordered_C.col(r_order.at(i)) = C.col(i);
    }

    return ordered_C;
}

Eigen::Matrix3d restore_R(double s1, double s2 ,double s3, const Eigen::MatrixXd & A, const Eigen::MatrixXd & B){
    Eigen::Matrix3d R_bar;
    R_bar << 1 + pow(s1,2) - pow(s2,2) - pow(s3,2),
            -2 * s3 + 2 * s1 * s2,
            2 * s2 + 2 * s1 * s3,
            2 * s3 + 2 * s1 * s2,
            1 - pow(s1,2) + pow(s2,2) - pow(s3,2),
            -2 * s1 + 2 * s2 * s3,
            -2 * s2 + 2 * s1 *s3,
            2 * s1 + 2 * s2 * s3,
            1 - pow(s1,2) - pow(s2,2) + pow(s3,2);
    Eigen::Matrix3d R = R_bar / (1 + pow(s1,2) + pow(s2,2) + pow(s3,2));
    return R;
}

Eigen::Vector3d restore_t(double s1, double s2 ,double s3, const Eigen::MatrixXd & A, const Eigen::MatrixXd & B){

    Eigen::Matrix <double,10,1> r;
    r << pow(s1,2), pow(s2,2), pow(s3,2), s1*s2, s1*s3, s2*s3, s1, s2, s3, 1;
    Eigen::Vector3d tau = -(B.transpose() * B).inverse() * B.transpose() * A * r;
    Eigen::Vector3d t = tau / (1 + pow(s1,2) + pow(s2,2) + pow(s3,2));
    return t;
}