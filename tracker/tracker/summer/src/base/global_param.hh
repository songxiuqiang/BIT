#pragma once

#include <string>
#include <vector>
#include <map>

namespace tk {

class GlobalParam {
public:
	static GlobalParam* Instance();
	void ParseConfig(const std::string& config_file);
	void DumpConfig(std::ostream& os);
	friend std::ostream& operator<<(std::ostream &os, const GlobalParam& gp);

	// input
	std::vector<std::string> model_file;
	bool unit_model{false};
	
	//std::string init_pose;

	std::string frames;

	int image_width, image_height;
	float fx, fy, cx, cy;
	float zn, zf;
	
	// histogram
	int hist_offset;
	int hist_rad;
	int search_rad;
	float alpha_fg;
	float alpha_bg;
	
	// search line
	int sl_seg;
	int line_len;

	// tracker
	int timeout;
	int tracker_mode;
	float weight_color;
	int sample_count;
	int target_frame;
	bool show_result;

	// field tracker
	int grid_step;
	bool is_normalize_fields;
	std::vector<float> pyramid_variance;

	// sovler
	int solver_mode;
	float delta_threshold;
	float error_threshold;
	float lambda;
	float u_rate, d_rate;
	std::vector<float> scales;
	std::vector<int> iters;

	// output
	std::string out_path;
	bool video_file_on;
	bool pose_file_on;
	std::string result_video_file;
	std::string bench_case;

	// benchmark
	std::string report_file;
	bool bench_mode;
	float marker_len;
	float offset_x;
	std::string gt_pose_file;
	std::string tk_pose_file;
	std::string fps_file;

protected:
	GlobalParam();

private:
	static GlobalParam* instance;
};

std::ostream& operator<<(std::ostream &os, const GlobalParam& gp);

} // namespace tk