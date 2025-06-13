#include <iostream>
#include <fstream>
#include <limits>
#include <glog/logging.h>
#include <opencv2/core.hpp>
#include "global_param.hh"
//#include "base/util.hh"

namespace tk {

GlobalParam* GlobalParam::instance = NULL;

GlobalParam::GlobalParam() {

}

GlobalParam* GlobalParam::Instance() {
	if (instance == NULL) {
		instance = new GlobalParam();
	}
	return instance;
}

template <typename T>
void ReadValue(cv::FileStorage& fs, const std::string& idx, T& field) {
	cv::FileNode node = fs[idx];
	CHECK(!node.empty()) << "check " << '<' << idx <<'>' << " in config file.";
	node >> field;
}

template <typename T>
void ReadArray(cv::FileStorage& fs, const std::string& idx, std::vector<T>& field) {
	cv::FileNode node = fs[idx];
	CHECK(!node.empty()) << "check " << '<' << idx <<'>' << " in config file.";
	for (cv::FileNodeIterator it = node.begin(); it != node.end(); it++) {
		field.push_back(T(*it));
	}
	//std::cout << std::endl;
}

void GlobalParam::ParseConfig(const std::string& config_file) {
	std::cout << "Parsing config file: " << config_file << std::endl;

	cv::FileStorage fs(config_file, cv::FileStorage::READ);
	CHECK(fs.isOpened()) << "failed to read config: " << config_file;

	cv::FileNode node;

/////////////////////////////////////
// input
/////////////////////////////////////

	//ReadValue(fs, "model_file", model_file);
	ReadArray(fs, "model_file", model_file);
	ReadValue(fs, "unit_model", unit_model);

	//ReadValue(fs, "init_pose", init_pose);

	ReadValue(fs, "frames", frames);

	ReadValue(fs, "fx", fx);
	ReadValue(fs, "fy", fy);
	ReadValue(fs, "cx", cx);
	ReadValue(fs, "cy", cy);

	ReadValue(fs, "zn", zn);
	ReadValue(fs, "zf", zf);

/////////////////////////////////////
// histogram
/////////////////////////////////////
	ReadValue(fs, "hist_offset", hist_offset);
	ReadValue(fs, "hist_rad", hist_rad);
	ReadValue(fs, "search_rad", search_rad);
	ReadValue(fs, "alpha_fg", alpha_fg);
	ReadValue(fs, "alpha_bg", alpha_bg);

/////////////////////////////////////
// tracker
/////////////////////////////////////
	ReadValue(fs, "line_len", line_len);
	ReadValue(fs, "sl_seg", sl_seg);

/////////////////////////////////////
// tracker
/////////////////////////////////////
	ReadValue(fs, "tracker_mode", tracker_mode);
	ReadValue(fs, "timeout", timeout);
	ReadValue(fs, "target_frame", target_frame);
	ReadValue(fs, "show_result", show_result);

	// field tracker
	//ReadValue(fs, "grid_step", grid_step);
	//ReadValue(fs, "is_normalize_fields", is_normalize_fields);
	//ReadArray(fs, "pyramid_variance", pyramid_variance);

/////////////////////////////////////
// solver
/////////////////////////////////////

	//ReadValue(fs, "solver_mode", solver_mode);
	//ReadValue(fs, "delta_threshold", delta_threshold);
	//ReadValue(fs, "error_threshold", error_threshold);

	//ReadArray(fs, "scales", scales);
	//ReadArray(fs, "iters", iters);

	//ReadValue(fs, "lambda", lambda);
	//ReadValue(fs, "u_rate", u_rate);
	//ReadValue(fs, "d_rate", d_rate);

/////////////////////////////////////
// output
/////////////////////////////////////
	//ReadValue(fs, "out_path", out_path);
	//ReadValue(fs, "video_file_on", video_file_on);
	//ReadValue(fs, "pose_file_on", pose_file_on);
	ReadValue(fs, "tk_pose_file", tk_pose_file);
	ReadValue(fs, "fps_file", fps_file);
	ReadValue(fs, "result_video_file", result_video_file);
	ReadValue(fs, "result_img_dir", bench_case);
/////////////////////////////////////
// benchmark
/////////////////////////////////////
	ReadValue(fs, "report_file", report_file);
	ReadValue(fs, "bench_mode", bench_mode);
	ReadValue(fs, "gt_pose_file", gt_pose_file);
	//ReadValue(fs, "marker_len", marker_len);
	//ReadValue(fs, "offset_x", offset_x);
}

template <typename T>
void DumpValue(std::ostream& os, const std::string& idx, const T& field) {
	os << idx << ": " << field << std::endl;
}

void DumpValue(std::ostream& os, const std::string& idx, const std::string& field) {
	os << idx << ": \"" << field << '\"' << std::endl;
}

template <typename T>
void DumpArray(std::ostream& os, const std::string& idx, const std::vector<T>& field) {
	os << idx << ": [ ";
	int array_size = field.size();
	for (int i = 0; i < array_size; ++i) {
		os << field[i];
		if (i != array_size-1)
			os << ", ";
	}
	os << " ]" << std::endl;
}

std::ostream& operator<<(std::ostream &os, const GlobalParam& gp) {
	os << "%YAML:1.0" << std::endl;
/////////////////////////////////////
// input
/////////////////////////////////////

	//DumpValue(os, "model_file", gp.model_file);
	DumpValue(os, "unit_model", gp.unit_model);

	//DumpValue(os, "init_pose", gp.init_pose);

	DumpValue(os, "frames", gp.frames);

	DumpValue(os, "fx", gp.fx);
	DumpValue(os, "fy", gp.fy);
	DumpValue(os, "cx", gp.cx);
	DumpValue(os, "cy", gp.cy);

/////////////////////////////////////
// tracker
/////////////////////////////////////

	DumpValue(os, "tracker_mode", gp.tracker_mode);
	DumpValue(os, "line_len", gp.line_len);
	DumpValue(os, "alpha_fg", gp.alpha_fg);
	DumpValue(os, "alpha_bg", gp.alpha_bg);

	// field tracker
	DumpValue(os, "grid_step", gp.grid_step);
	DumpValue(os, "is_normalize_fields", gp.is_normalize_fields);
	DumpArray(os, "pyramid_variance", gp.pyramid_variance);

/////////////////////////////////////
// solver
/////////////////////////////////////

	DumpValue(os, "solver_mode", gp.solver_mode);
	DumpValue(os, "delta_threshold", gp.delta_threshold);
	DumpValue(os, "error_threshold", gp.error_threshold);
	
	DumpArray(os, "scales", gp.scales);
	DumpArray(os, "iters", gp.iters);

	DumpValue(os, "lambda", gp.lambda);
	DumpValue(os, "u_rate", gp.u_rate);
	DumpValue(os, "d_rate", gp.d_rate);

/////////////////////////////////////
// output
/////////////////////////////////////
	DumpValue(os, "out_path", gp.out_path);
	DumpValue(os, "video_file_on", gp.video_file_on);
	DumpValue(os, "pose_file_on", gp.pose_file_on);

/////////////////////////////////////
// benchmark
/////////////////////////////////////
	DumpValue(os, "gt_pose_file", gp.gt_pose_file);
	DumpValue(os, "marker_len", gp.marker_len);
	DumpValue(os, "offset_x", gp.offset_x);

	return os;
}

void GlobalParam::DumpConfig(std::ostream& os) {
	os << *this;
}

} // namespace tk