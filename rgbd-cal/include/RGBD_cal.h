#pragma once

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <gluestick_detection/CustomFloatArray.h>

#include <iostream>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include "cam_pair.hpp"


using namespace cv;
using namespace sensor_msgs;
using namespace message_filters;

typedef gluestick_detection::CustomFloatArray::ConstPtr FeaturePtr;

typedef sync_policies::ApproximateTime<gluestick_detection::CustomFloatArray,gluestick_detection::CustomFloatArray,gluestick_detection::CustomFloatArray,gluestick_detection::CustomFloatArray,sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

class calibration_node 
{
    public:
         calibration_node(ros::NodeHandle nh_) : cam1_line_sub(nh_,"/cam_1/lines",10),cam1_point_sub(nh_,"/cam_1/points",10), cam1_depth_sub(nh_,"/cam_1/depth",10),  
     cam2_line_sub(nh_,"/cam_2/lines",10),cam2_point_sub(nh_,"/cam_2/points",10),cam2_depth_sub(nh_,"/cam_2/depth",10),sync(MySyncPolicy(10),cam1_line_sub,cam2_line_sub,cam1_point_sub,cam2_point_sub,cam1_depth_sub,cam2_depth_sub)  {

        sync.registerCallback(boost::bind(&calibration_node::callback,this,_1,_2,_3,_4,_5,_6));
     }
	  
	  void setIntrinsic(const Eigen::Matrix3d & source_K, const Eigen::Matrix3d & target_K){
		this->system.source_K_ = source_K;
		this->system.target_K_ = target_K;
        Eigen::Matrix<double,3,4> temp;
    	temp.block(0,0,3,3) = Eigen::Matrix3d::Identity();
    	temp.block(0,3,3,1) = Eigen::Vector3d::Zero();
		this->system.P = target_K * temp;
	  }

	  void setParam(double eps, double cost, int minimum_line_num, int voting_line_num){
		this->system.eps = eps;
		this->system.voting_thresh = voting_line_num;
		this->system.minimum_line_num = minimum_line_num;
		this->system.convergency_cost = cost;
	  }

      void callback(const FeaturePtr& cam1_line_ptr, const FeaturePtr& cam2_line_ptr, const FeaturePtr& cam1_kp_ptr, const FeaturePtr& cam2_kp_ptr, const ImageConstPtr& cam1_depth_image, const ImageConstPtr& cam2_depth_image){
		cv_bridge::CvImagePtr cam1_depth_ptr = cv_bridge::toCvCopy(cam1_depth_image, sensor_msgs::image_encodings::TYPE_16UC1);
		cv_bridge::CvImagePtr cam2_depth_ptr = cv_bridge::toCvCopy(cam2_depth_image, sensor_msgs::image_encodings::TYPE_16UC1);
		cv::Mat cam1_depth = cam1_depth_ptr->image.clone();
		cv::Mat cam2_depth = cam2_depth_ptr->image.clone();
		int img_height = cam1_depth.rows;
		int img_width = cam1_depth.cols;
		std::cout << cam1_line_ptr->dim1 << " lines are matched!" << std::endl;
     	std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> image_point_cam1;
		std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> image_point_cam2;
    	
		for (int i = 0; i < cam1_line_ptr->dim1; i++)
		{	
			bool pair1x_out = cam1_line_ptr->data[i*4] <= 0 || cam1_line_ptr->data[i*4] >= img_width - 0.5;
			bool pair2x_out = cam1_line_ptr->data[i*4+2] <= 0 || cam1_line_ptr->data[i*4+2] >= img_width - 0.5;
			bool pair3x_out = cam2_line_ptr->data[i*4] <= 0 || cam2_line_ptr->data[i*4] >= img_width - 0.5;
			bool pair4x_out = cam2_line_ptr->data[i*4+2] <= 0 || cam2_line_ptr->data[i*4+2] >= img_width - 0.5;

			bool pair1y_out = cam1_line_ptr->data[i*4+1] <= 0 || cam1_line_ptr->data[i*4+1] >= img_height - 0.5;
			bool pair2y_out = cam1_line_ptr->data[i*4+3] <= 0 || cam1_line_ptr->data[i*4+3] >= img_height - 0.5;
			bool pair3y_out = cam2_line_ptr->data[i*4+1] <= 0 || cam2_line_ptr->data[i*4+1] >= img_height - 0.5;
			bool pair4y_out = cam2_line_ptr->data[i*4+3] <= 0 || cam2_line_ptr->data[i*4+3] >= img_height - 0.5;

			if(pair1x_out || pair1y_out || pair2x_out || pair2y_out || pair3x_out || pair3y_out || pair4x_out || pair4y_out) continue;
			Eigen::Vector2d pair1(cam1_line_ptr->data[i*4], cam1_line_ptr->data[i*4+1]);
			Eigen::Vector2d pair2(cam1_line_ptr->data[i*4+2], cam1_line_ptr->data[i*4+3]);
			Eigen::Vector2d pair3(cam2_line_ptr->data[i*4], cam2_line_ptr->data[i*4+1]);
			Eigen::Vector2d pair4(cam2_line_ptr->data[i*4+2], cam2_line_ptr->data[i*4+3]);
			image_point_cam1.push_back(std::make_pair(pair1, pair2));
			image_point_cam2.push_back(std::make_pair(pair3, pair4));
		}

		int before_line_num = system.lines_full.size() + system.lines_pnl.size();
		system.add_line_features(image_point_cam1, image_point_cam2, cam1_depth, cam2_depth);

		bool is_line_added = system.lines_full.size() + system.lines_pnl.size() - before_line_num > 0;

		if(is_line_added && system.lines_full.size() + system.lines_pnl.size() >= 4 ) {
			converged = system.perform_PnL();}	
	}
	bool hasConverged() const { 
		return converged; }

	 message_filters::Subscriber<gluestick_detection::CustomFloatArray> cam1_line_sub;
	 message_filters::Subscriber<gluestick_detection::CustomFloatArray> cam2_line_sub;
	 message_filters::Subscriber<gluestick_detection::CustomFloatArray> cam1_point_sub;
	 message_filters::Subscriber<gluestick_detection::CustomFloatArray> cam2_point_sub;
     message_filters::Subscriber<sensor_msgs::Image> cam1_depth_sub;
     message_filters::Subscriber<sensor_msgs::Image> cam2_depth_sub;
     ros::Publisher result_pub;
     message_filters::Synchronizer<MySyncPolicy> sync;
 	 camera_pair system;
	 bool converged = false;
	 
     
};

