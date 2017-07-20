#pragma once

#include <opencv2\opencv.hpp>

#include <iostream>
#include <map>
#include <vector>
#include <string>

#include "mxnet-cpp\MxNetCpp.h"

#include <thread>

extern mxnet::cpp::Context global_ctx;

struct Prediction {

	Prediction(std::string em, cv::Scalar co, float pr) : emotion(em), color(co), probability(pr) {}

	std::string emotion;
	cv::Scalar color;
	float probability;

	bool operator==(const Prediction& other) const {
		return emotion.compare(other.emotion) == 0;
	}

};

class FERClassifier
{
public:
	FERClassifier(std::string model_file, std::string param_file, std::string mean_file);
	std::vector<Prediction> Predict(cv::Mat image);
	~FERClassifier();

	std::string emo_map[10];
	cv::Scalar color_map[10];

private:

	int net_input_height = 48;
	int net_input_width = 48;
	int net_input_channels = 3;
	
	mxnet::cpp::NDArray mean_img;
	std::map<std::string, mxnet::cpp::NDArray> args_map;
	std::map<std::string, mxnet::cpp::NDArray> aux_map;
	mxnet::cpp::Symbol network;

	bool bound = false;

};

