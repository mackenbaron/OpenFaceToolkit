// OpencvFERWin32Console.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <functional>
#include <atomic>
#include <chrono>
#include <utility>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>

#include "../Lib3000fps/lbf_lbf.hpp"

#include "FERClassifier.h"




void capture_thread(cv::VideoCapture& capture, cv::Mat& loaded_frame, std::mutex& loading_mutex, bool& running) {
	while (running) {
		loading_mutex.lock();
		capture.read(loaded_frame);
		loading_mutex.unlock();
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}
}

void extraction_thread(cv::Mat& input_frame, cv::Mat& output_frame, cv::Mat& cropped_frame, std::mutex& if_mutex, std::mutex& out_mutex, 
					   cv::CascadeClassifier& face_cascade, lbf::LbfCascador& face_alignment_cascador, bool& running, bool& face_detected, unsigned int& fn) {
	
	cv::Mat local_frame;
	cv::Mat grayscale_frame;
	cv::Mat cropped;
	cv::Mat fa_results;
	std::vector<cv::Rect> faces;
	int bounding_offset = 20;

	int frame_h;
	int frame_w;
	
	while (running) {
		if_mutex.lock();
		local_frame = input_frame;
		if_mutex.unlock();

		std::clock_t start_time = std::clock();

		if (!local_frame.empty()) {

			// Setup frame height and width
			frame_h = local_frame.rows;
			frame_w = local_frame.cols;

			// Convert the frame to grayscale
			cv::cvtColor(local_frame, grayscale_frame, cv::COLOR_BGR2GRAY);
			cv::equalizeHist(grayscale_frame, grayscale_frame);

			// Detect faces
			face_cascade.detectMultiScale(grayscale_frame, faces);

			// Get the largest face
			int area = 0;
			cv::Rect largest;
			if (faces.size() >= 1) {
				largest = faces[0];
				area = faces[0].height*faces[0].width;
				for (int i = 1; i < faces.size(); i++) {
					if (faces[i].height*faces[i].width > area) {
						largest = faces[i];
						area = faces[i].height*faces[i].width;
					}
				}
			}

			// Process the largest face
			if (faces.size() >= 1 && area > 10000) {

				// Generate the cropped patch
				int x_min = std::max<int>(0, largest.x - bounding_offset);
				int y_min = std::max<int>(0, largest.y - bounding_offset);
				int x_max = std::min<int>(frame_w, largest.x + largest.width + bounding_offset);
				int y_max = std::min<int>(frame_h, largest.y + largest.height + bounding_offset);
				lbf::BBox face_bounding(largest.x, largest.y, largest.width, largest.height);
				cv::Rect roi(x_min, y_min, x_max - x_min, y_max - y_min);
				cropped = local_frame(roi);

				out_mutex.lock();
				cropped_frame = local_frame(roi).clone();
				out_mutex.unlock();

				// Perform face alignment
				fa_results = face_alignment_cascador.Predict(grayscale_frame, face_bounding);
				for (int i = 0; i < fa_results.rows; i++) {
					cv::circle(local_frame, cv::Point(fa_results.at<double>(i, 0), fa_results.at<double>(i, 1)), 2, cv::Scalar(0, 255, 0), -1);
				}

				// Perform pose estimation
				std::vector<cv::Point2d> image_points;
				image_points.push_back(cv::Point2d(fa_results.at<double>(33, 0), fa_results.at<double>(33, 1))); // Tip of nose
				image_points.push_back(cv::Point2d(fa_results.at<double>(8, 0), fa_results.at<double>(8, 1)));   // Chin
				image_points.push_back(cv::Point2d(fa_results.at<double>(36, 0), fa_results.at<double>(36, 1))); // Left Eye left corner
				image_points.push_back(cv::Point2d(fa_results.at<double>(45, 0), fa_results.at<double>(45, 1))); // Right eye right corner
				image_points.push_back(cv::Point2d(fa_results.at<double>(48, 0), fa_results.at<double>(48, 1))); // Left mouth corner
				image_points.push_back(cv::Point2d(fa_results.at<double>(54, 0), fa_results.at<double>(54, 1))); // Right mouth corner

				std::vector<cv::Point3d> model_points;
				model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
				model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
				model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
				model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
				model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
				model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner

				double focal_length = local_frame.cols; // Approximate focal length.
				cv::Point2d center = cv::Point2d(local_frame.cols / 2, local_frame.rows / 2);
				cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
				cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

				// Output rotation and translation
				cv::Mat rotation_vector; // Rotation in axis-angle form
				cv::Mat translation_vector;

				// Solve for pose
				cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

				std::vector<cv::Point3d> nose_end_point3D;
				std::vector<cv::Point2d> nose_end_point2D;
				nose_end_point3D.push_back(cv::Point3d(0, 0, 500.0));

				cv::projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);

				// Draw the line
				cv::line(local_frame, image_points[0], nose_end_point2D[0], cv::Scalar(0, 0, 255), 2);

				// Draw as rectangle around the face
				cv::rectangle(local_frame, roi, cv::Scalar(255, 255, 255), 1, 8, 0);

				out_mutex.lock();
				output_frame = local_frame.clone();
				face_detected = true;
				fn = 0;
				out_mutex.unlock();
			}
			else {
				out_mutex.lock();
				cropped_frame = cv::Mat();
				output_frame = local_frame;
				fn++;
				if (fn > 10) {
					face_detected = false;
				}
				out_mutex.unlock();
			}
		}
		else {
			std::cout << "Local frame was empty... skipping..." << std::endl;
		}

		double duration = (std::clock() - start_time) / (double)CLOCKS_PER_SEC;
		duration *= 1000;
		//std::cout << duration << std::endl;
		//std::this_thread::sleep_for(std::chrono::milliseconds(std::max<int>(0, (int)(40 - duration))));
	}
}

void prediction_thread(cv::Mat& pred_frame, std::vector<Prediction>& prediction, std::mutex& frame_mutex, std::mutex& pred_mutex, FERClassifier& predictor, bool& running, bool& face_detected) {
	
	cv::Mat local_frame;
	
	while (running) {
		frame_mutex.lock();
		local_frame = pred_frame;
		frame_mutex.unlock();

		if (!local_frame.empty()) {
			std::vector<Prediction> pred = predictor.Predict(local_frame);

			pred_mutex.lock();
			prediction = pred;
			pred_mutex.unlock();
		}
		else {
			std::vector<Prediction> pred;
			pred_mutex.lock();
			prediction = pred;
			pred_mutex.unlock();
			std::this_thread::sleep_for(std::chrono::milliseconds(30));
		}
	}
}


int main(int argc, char** argv)
{

	if (argc != 6) {
		std::cout << "Usage: OpencvFERWin32Console.exe <cascade xml> <lbf cascador model> <mxnet symbol json> <mxnet params file> <mxnet mean file>" << std::endl;
		return 2;
	}

	std::string fcc_string = std::string(argv[1]);
	std::string lbf_string = std::string(argv[2]);
	std::string mxs_string = std::string(argv[3]);
	std::string mxp_string = std::string(argv[4]);
	std::string mxm_string = std::string(argv[5]);

	std::cout << "Loading files: " << std::endl;
	std::cout << "\tFace Detection Cascade: " << fcc_string << std::endl;
	std::cout << "\tFacial Landmark Cascade: " << lbf_string << std::endl;
	std::cout << "\tMxNet Symbols: " << mxs_string << std::endl;
	std::cout << "\tMxNet Parameters: " << mxp_string << std::endl;
	std::cout << "\tMxNet Mean: " << mxm_string << std::endl;


	// Start the capture
	cv::VideoCapture input_stream(0);
	if (!input_stream.isOpened()) {
		std::cerr << "Cannot open video stream. Error in creation." << std::endl;
		system("pause");
		return 1;
	}

	// Initialize the viola jones detector
	cv::CascadeClassifier face_cascade(fcc_string);
	if (face_cascade.empty()) {
		std::cerr << "Failed to load the face cascade, terminating." << std::endl;
		system("pause");
		return 1;
	}

	// Initialize the LBF Cascador for landmark extraction
	lbf::LbfCascador face_alignment_cascador;
	FILE* fa_model = fopen(lbf_string.c_str(), "rb");
	if (fa_model == NULL) {
		std::cerr << "Failed to read the LBF Model, terminating." << std::endl;
		system("pause");
		return 1;
	}
	face_alignment_cascador.Read(fa_model);
	fclose(fa_model);

	// Initialize the FER
	FERClassifier face_rec(mxs_string, mxp_string, mxm_string);


	// Setup global vars
	cv::Mat captured_frame, processed_frame, c_processed_frame;
	std::mutex c_frame_mtx, p_frame_mtx, pr_mtx;
	std::vector<Prediction> prediction;
	bool running = true;
	bool face_detected = false;
	unsigned int frame_smoothing_number;

	// Setup and start the threads
	std::thread cap_thread(capture_thread, std::ref(input_stream), std::ref(captured_frame), std::ref(c_frame_mtx), std::ref(running));
	std::thread proc_thread(extraction_thread, std::ref(captured_frame), std::ref(processed_frame), std::ref(c_processed_frame),
		std::ref(c_frame_mtx), std::ref(p_frame_mtx), std::ref(face_cascade), std::ref(face_alignment_cascador), std::ref(running), std::ref(face_detected), std::ref(frame_smoothing_number));
	std::thread pred_thread(prediction_thread, std::ref(c_processed_frame), std::ref(prediction), std::ref(p_frame_mtx), std::ref(pr_mtx),
		std::ref(face_rec), std::ref(running), std::ref(face_detected));

	// Initialize the prediction vectors
	int NUM_SMOOTHING_FRAMES = 10;
	std::vector<std::vector<Prediction>> last_preds;
	for (int i = 0; i < NUM_SMOOTHING_FRAMES; i++)
		last_preds.push_back(std::vector<Prediction>());
	unsigned int modulo = 0;


	// Run the display thread
	while (running) {
		cv::Mat my_frame;

		p_frame_mtx.lock();
		my_frame = processed_frame;
		p_frame_mtx.unlock();


		
		if (modulo > NUM_SMOOTHING_FRAMES - 1)
			modulo = 0;
		last_preds.at(modulo).clear();

		// Obtain the latest predictions
		pr_mtx.lock();
		for (auto pred : prediction) {
			last_preds.at(modulo).push_back(pred);
		}
		pr_mtx.unlock();

		// Calculate the average value and best prediction
		float avgs[8];
		for (int i = 0; i < 8; i++)
			avgs[i] = 0.;
		int count = 0;
		for (auto frame_preds : last_preds) {
			if (frame_preds.size() == 0) {
				continue;
			}
			for (int i = 0; i < frame_preds.size(); i++ ) {
				avgs[i] += (float) frame_preds.at(i).probability;
			}
			count++;
		}
		for (int i = 0; i < 8; i++) {
			avgs[i] = avgs[i] / (float)count;
		}

		Prediction best("", 0, -1.);
		float best_val = 0.0001;
		std::vector<Prediction> average_predictions;

		for (int i = 0; i < 8; i++) {
			if (avgs[i] > best_val) {
				best = Prediction(face_rec.emo_map[i], face_rec.color_map[i], avgs[i]);
				best_val = avgs[i];
			}
			average_predictions.push_back(Prediction(face_rec.emo_map[i], face_rec.color_map[i], avgs[i]));
		}
		modulo++;


		// Actually do the drawing

		if (!my_frame.empty()) {
			if (best.probability > 0) {
				cv::rectangle(my_frame, cv::Rect(cv::Point(my_frame.cols - 260, 0), cv::Point(my_frame.cols, 135)), cv::Scalar(0, 0, 0), -1);
			}
			else {
				cv::rectangle(my_frame, cv::Rect(cv::Point(my_frame.cols - 100, 0), cv::Point(my_frame.cols, 135)), cv::Scalar(0, 0, 0), -1);
			}

			// Draw the prediction
			if (best.probability > 0)
			{
				int base_width = my_frame.cols - 100;
				int base_ext = 150;

				int fh = -5;
				for (auto pred : average_predictions) {
					fh += 15;
					if (pred == best) {
						cv::line(my_frame, cv::Point(base_width, fh), cv::Point(my_frame.cols - (100 + (int)((pred.probability / (float)best.probability)*base_ext)), fh), cv::Scalar(0, 0, 255), 2, 8, 0);
					}
					else {
						cv::line(my_frame, cv::Point(base_width, fh), cv::Point(my_frame.cols - (100 + (int)((pred.probability / (float)best.probability)*base_ext)), fh), cv::Scalar(0, 255, 0), 2, 8, 0);
					}
					
				}
			}

			// Draw the emotions
			int f_width = my_frame.cols;
			for (int i = 0; i < 8; i++) {
				cv::putText(my_frame, face_rec.emo_map[i], cv::Point(f_width - 90, 15 * (i + 1)), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, face_rec.color_map[i], 1);
			}

			// Display to output
			cv::imshow("Camera Output", my_frame);
		}

		if (cv::waitKey(30) >= 0) {
			running = false;
			break;
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(40));
	}

		cap_thread.join();
		proc_thread.join();
		pred_thread.join();

		return 0;

}



//
//	// Setup the basic variables
//	cv::Mat frame;
//	cv::Mat grayscale_frame;
//	cv::Mat cropped;
//	cv::Mat fa_results;
//	std::vector<cv::Rect> faces;
//	int bounding_offset = 50;
//
//	int frame_h;
//	int frame_w;
//
//	while (true) {
//		// Read the frame input from the camera
//		input_stream.read(frame);
//
//
//
//		if (!frame.empty()) {
//
//			// Setup frame height and width
//			frame_h = frame.rows;
//			frame_w = frame.cols;
//
//			cv::flip(frame, frame, 1);
//
//			// Convert the frame to grayscale
//			cv::cvtColor(frame, grayscale_frame, cv::COLOR_BGR2GRAY);
//			cv::equalizeHist(grayscale_frame, grayscale_frame);
//
//			// Detect faces
//			face_cascade.detectMultiScale(grayscale_frame, faces);
//			
//			// Process the largest face
//			if (faces.size() >= 1) {
//
//				// Get the largest face
//				cv::Rect largest = faces[0];
//				int area = faces[0].height*faces[0].width;
//				for (int i = 1; i < faces.size(); i++) {
//					if (faces[i].height*faces[i].width > area) {
//						largest = faces[i];
//						area = faces[i].height*faces[i].width;
//					}
//				}
//
//
//				// Generate the cropped patch
//				int x_min = std::max<int>(0, largest.x - bounding_offset);
//				int y_min = std::max<int>(0, largest.y - bounding_offset);
//				int x_max = std::min<int>(frame_w,largest.x +largest.width + bounding_offset);
//				int y_max = std::min<int>(frame_h, largest.y + largest.height + bounding_offset);
//				lbf::BBox face_bounding(largest.x, largest.y, largest.width, largest.height);
//				cv::Rect roi(x_min, y_min, x_max - x_min, y_max - y_min);
//				cropped = frame(roi);
//
//				// Perform face alignment
//				fa_results = face_alignment_cascador.Predict(grayscale_frame, face_bounding);
//				for (int i = 0; i < fa_results.rows; i++) {
//					cv::circle(frame, cv::Point(fa_results.at<double>(i, 0), fa_results.at<double>(i, 1)), 2, cv::Scalar(0, 255, 0), -1);
//				}
//
//				// Perform pose estimation
//				std::vector<cv::Point2d> image_points;
//				image_points.push_back(cv::Point2d(fa_results.at<double>(33, 0), fa_results.at<double>(33, 1))); // Tip of nose
//				image_points.push_back(cv::Point2d(fa_results.at<double>(8, 0), fa_results.at<double>(8, 1))); // Chin
//				image_points.push_back(cv::Point2d(fa_results.at<double>(36, 0), fa_results.at<double>(36, 1))); // Left Eye left corner
//				image_points.push_back(cv::Point2d(fa_results.at<double>(45, 0), fa_results.at<double>(45, 1))); // Right eye right corner
//				image_points.push_back(cv::Point2d(fa_results.at<double>(48, 0), fa_results.at<double>(48, 1))); // Left mouth corner
//				image_points.push_back(cv::Point2d(fa_results.at<double>(54, 0), fa_results.at<double>(54, 1))); // Right mouth corner
//
//				std::vector<cv::Point3d> model_points;
//				model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
//				model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
//				model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
//				model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
//				model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
//				model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner
//
//				double focal_length = frame.cols; // Approximate focal length.
//				cv::Point2d center = cv::Point2d(frame.cols / 2, frame.rows / 2);
//				cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
//				cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion
//
//				// Output rotation and translation
//				cv::Mat rotation_vector; // Rotation in axis-angle form
//				cv::Mat translation_vector;
//
//				// Solve for pose
//				cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);
//
//				std::vector<cv::Point3d> nose_end_point3D;
//				std::vector<cv::Point2d> nose_end_point2D;
//				nose_end_point3D.push_back(cv::Point3d(0, 0, 500.0));
//
//				cv::projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);
//
//				// Draw the line
//				cv::line(frame, image_points[0], nose_end_point2D[0], cv::Scalar(0, 0, 255), 2);
//
//				// Perform classification
//				face_rec.Predict(cropped);
//
//				// Draw as rectangle around the face
//				cv::rectangle(frame, roi, cv::Scalar(255, 0, 0), 1, 8, 0);
//			}
//				
//			
//
//			// Show the output image
//			cv::imshow("Camera Output",frame);
//		}
//
//		if (cv::waitKey(30) >= 0) {
//			break;
//		}
//
//	}
//
//    return 0;
//}
