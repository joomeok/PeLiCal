#include <ros/ros.h>
#include "RGBD_cal.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "message_filter");
  ros::NodeHandle nh;
  // Create subscribers for the two topics
  std::vector<double> source_K_vec, target_K_vec;
  double eps, convergency_cost;
  int voting_line_num, minimum_line;
  Eigen::Matrix3d source_K, target_K;

  nh.getParam("target_intrinsic/data",target_K_vec);
  nh.getParam("source_intrinsic/data",source_K_vec);

  nh.getParam("voting_eps", eps);
  nh.getParam("voting_thresh", voting_line_num);
  nh.getParam("minimum_line_num", minimum_line);
  nh.getParam("convergency_cost", convergency_cost);

  calibration_node node(nh);

  source_K << source_K_vec[0], source_K_vec[1], source_K_vec[2], 
  source_K_vec[3], source_K_vec[4], source_K_vec[5], 
  source_K_vec[6], source_K_vec[7], source_K_vec[8]; 

  target_K << target_K_vec[0], target_K_vec[1], target_K_vec[2], 
  target_K_vec[3], target_K_vec[4], target_K_vec[5], 
  target_K_vec[6], target_K_vec[7], target_K_vec[8]; 
  
  node.setIntrinsic(source_K,target_K);
  node.setParam(eps,convergency_cost,minimum_line,voting_line_num);
  ros::Rate rate(10); // Rate object to control the loop frequency. Here it's 10Hz.

  while(ros::ok())
  {
    ros::spinOnce(); // Process callbacks.
    std::cout << node.hasConverged() << std::endl;
    if(node.hasConverged())
    { 
      ROS_INFO("Convergence reached! Shutting down node...");
      ros::shutdown();
    }

    rate.sleep(); // Sleep for the remainder of the loop cycle.
  }
  return 0;
}
