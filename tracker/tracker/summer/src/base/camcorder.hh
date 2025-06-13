#pragma once

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace tk {

class Camcorder {
public:
	Camcorder(std::string video_file, int frame_width, int frame_height, bool active, int fps = 30) {
		is_active = active;
		if (is_active) {
			//writer.open(video_file, CV_FOURCC('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height), true);
			//writer.open(video_file, CV_FOURCC('D', 'I', 'V', 'X'), fps, cv::Size(frame_width, frame_height), true);
			writer.open(video_file, CV_FOURCC('F','M','P','4'), fps, cv::Size(frame_width, frame_height), true);
			CHECK(writer.isOpened()) << "failed to open video file";
		}
	}

	~Camcorder() {
		if (is_active) {
			writer.release();
		}
	}

	void Record(cv::Mat& frame) {
		if(is_active) {
			writer << frame;
		}
	}
	
	cv::VideoWriter writer;
	bool is_active;
};

} // namespace tk