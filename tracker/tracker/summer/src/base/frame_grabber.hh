#pragma once

#include <climits>
#include <string>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "compatible.h"

#ifdef WIN32
#include "base/dirent_win.h"
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#endif

namespace tk {

class VideoFrameGrabber;
class FrameGrabber {
public:
	enum {
		LIVE = 0,
		VIDEO = 1,
		IMAGES = 2,
	};

	static FrameGrabber* GetFrameGrabber(const std::string& frames);

	virtual bool GrabFrame(cv::Mat& frame) = 0;
	virtual bool GrabFrame(cv::Mat& frame, int idx) = 0;

	int GetFrameIndex() {
		return frame_index_-1;
	}

	int width;
	int height;
	int type;

protected:
	FrameGrabber() {
		width = -1;
		height = -1;
	}

	int frame_index_;
};

class VideoFrameGrabber: public FrameGrabber {
public:
	VideoFrameGrabber(const std::string& frames) {
		frame_index_ = 0;

		if (cap_.open(frames)) {
			width = (int)cap_.get(CV_CAP_PROP_FRAME_WIDTH);
			height = (int)cap_.get(CV_CAP_PROP_FRAME_HEIGHT);
			return;
		}
		return;
	}

	VideoFrameGrabber(int cam_id) {
		frame_index_ = 0;

		if (cap_.open(cam_id)) {
			width = (int)cap_.get(CV_CAP_PROP_FRAME_WIDTH);
			height = (int)cap_.get(CV_CAP_PROP_FRAME_HEIGHT);
			return;
		}
		return;
	}

	virtual bool GrabFrame(cv::Mat& frame) override {
		bool ret = false;
		ret = cap_.read(frame);
		frame_index_++;
		return ret;
	}

	virtual bool GrabFrame(cv::Mat& frame, int idx) override {
		CHECK(false) << "VideoFrameGrabber NOT IMPLEMENT GrabFrame(cv::Mat& frame, int idx)";
		return false;
	}

	cv::VideoCapture cap_;
};

static int only_png(const struct dirent *entry) {
	std::string fname(entry->d_name);
	int len = fname.length();
	if (len < 4) {
		return 0;
	}

	int pass;
	std::string dot_ext(fname.substr(len - 4, 4));
	if (dot_ext == std::string(".png")) {
		pass = 1;
	}
	else {
		pass = 0;
	}

	return pass;
}

static int AlphaSort(const void *a, const void *b) {
	struct dirent **pa = (struct dirent **)a;
	struct dirent **pb = (struct dirent **)b;

	return strcoll((*pa)->d_name, (*pb)->d_name);
}

class ImageFrameGrabber: public FrameGrabber{
public:
	ImageFrameGrabber(const std::string& image_dir) {
		image_dir_ = image_dir;
		frame_index_ = 0;

		frame_count_ = scandir(image_dir.c_str(), &files, only_png, alphasort);

		std::stringstream stm;
		stm << image_dir << '/' << files[0]->d_name;
		cv::Mat frame = cv::imread(stm.str());
		if (!frame.empty()) {
			width = frame.cols;
			height = frame.rows;
		}
	}

	~ImageFrameGrabber() {
		for (int i = 0; i < frame_count_; i++) {
			free (files[i]);
		}
		free (files);
	}

	virtual bool GrabFrame(cv::Mat& frame) override {
		bool ret = false;

		if (frame_index_ == frame_count_)
			return false;

		std::stringstream stm;
		stm << image_dir_ << '/' << files[frame_index_]->d_name;
		frame = cv::imread(stm.str());
		ret = !frame.empty();

		frame_index_++;
		return ret;
	}

	virtual bool GrabFrame(cv::Mat& frame, int idx) override {
		bool ret = false;

		if (idx >= frame_count_)
			return false;

		std::stringstream stm;
		stm << image_dir_ << '/' << files[idx]->d_name;
		frame = cv::imread(stm.str());
		ret = !frame.empty();

		return ret;
	}

	std::string image_dir_;
	struct dirent **files;
	int frame_count_;
};

} // namespace tk