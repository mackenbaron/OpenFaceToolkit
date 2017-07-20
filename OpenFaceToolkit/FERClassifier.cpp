#include "stdafx.h"
#include "FERClassifier.h"

mxnet::cpp::Context global_ctx(mxnet::cpp::kCPU, 0);

FERClassifier::FERClassifier(std::string model_file, std::string params_file, std::string mean_file) 
{
	// Load the model symbols
	this->network = mxnet::cpp::Symbol::Load(model_file).GetInternals()["fc1_output"];
	
	// Load the parameters
	std::map<std::string, mxnet::cpp::NDArray> params;
	mxnet::cpp::NDArray::Load(params_file, 0, &params);
	
	for (const auto &k : params) {
		if (k.first.substr(0, 4) == "aux:") {
			auto name = k.first.substr(4, k.first.size() - 4);
			aux_map[name] = k.second.Copy(global_ctx);
		}
		if (k.first.substr(0, 4) == "arg:") {
			auto name = k.first.substr(4, k.first.size() - 4);
			args_map[name] = k.second.Copy(global_ctx);
		}
	}
	mxnet::cpp::NDArray::WaitAll();

	// Load the mean file
	mean_img = mxnet::cpp::NDArray(mxnet::cpp::Shape(1, net_input_channels, net_input_width, net_input_height), global_ctx, false);
	mean_img.SyncCopyFromCPU(mxnet::cpp::NDArray::LoadToMap(mean_file)["mean_img"].GetData(), 1 * net_input_channels * net_input_width * net_input_height);

	mxnet::cpp::NDArray::WaitAll();

	// Setup string emotion and color maps
	this->emo_map[0] = "Neutral";
	this->emo_map[1] = "Happiness";
	this->emo_map[2] = "Sadness";
	this->emo_map[3] = "Surprise";
	this->emo_map[4] = "Fear";
	this->emo_map[5] = "Disgust";
	this->emo_map[6] = "Anger";
	this->emo_map[7] = "Contempt";

	this->color_map[0] = cv::Scalar(255, 255, 255);
	this->color_map[1] = cv::Scalar(0, 255, 255);
	this->color_map[2] = cv::Scalar(255, 144, 30);
	this->color_map[3] = cv::Scalar(255, 255, 0);
	this->color_map[4] = cv::Scalar(255, 0, 255);
	this->color_map[5] = cv::Scalar(0, 255, 0);
	this->color_map[6] = cv::Scalar(0, 0, 255);
	this->color_map[7] = cv::Scalar(0, 153, 255);



}

std::vector<Prediction> FERClassifier::Predict(cv::Mat image) {

	// Have to convert the matrix to an array and resize it to fit the network
	std::vector<float> array;
	cv::resize(image, image, cv::Size(net_input_width, net_input_height));
	cv::cvtColor(image, image, CV_BGR2RGB);

	for (int c = 0; c < net_input_channels; c++) {
		for (int i = 0; i < net_input_height; i++) {
			for (int j = 0; j < net_input_width; j++) {
				array.push_back(static_cast<float>(image.data[(i * net_input_width + j) * net_input_channels + c]));
			}
		}
	}

	// Now that we have the binary data, we need to convert to NDArray
	mxnet::cpp::NDArray data(mxnet::cpp::Shape(1, net_input_channels, net_input_width, net_input_height), global_ctx, false);
	data.SyncCopyFromCPU(array.data(), 1 * net_input_channels * net_input_width * net_input_height);
	mxnet::cpp::NDArray::WaitAll();
	
	// Setup the data
	data.Slice(0, 1) -= mean_img; // Subtract the mean file
	args_map["data"] = data;

	mxnet::cpp::NDArray::WaitAll();

	// Construct and bind the executor, this has to be done every frame. 
	// If I could change this, I would. But MxNet sucks. 
	mxnet::cpp::Executor* exec = this->network.SimpleBind(global_ctx, args_map, std::map<std::string, mxnet::cpp::NDArray>(),
			std::map<std::string, mxnet::cpp::OpReqType>(), this->aux_map);

	mxnet::cpp::NDArray::WaitAll();

	// Run the forward pass
	exec->Forward(false);
	auto output_array = exec->outputs[0].Copy(mxnet::cpp::Context(mxnet::cpp::kCPU,0));

	mxnet::cpp::NDArray::WaitAll();
	

	// Print the results ans store the vector
	std::vector<Prediction> to_ret;
	for (int i = 0; i < output_array.Size(); i++) {
		to_ret.push_back(Prediction(emo_map[i], color_map[i], abs(output_array.At(0, i))));
	}
	std::cout << std::endl;

	// Clean up the executor (No memory leaks here :) )
	delete exec;

	return to_ret;
}


FERClassifier::~FERClassifier()
{
	//delete exec;
}
