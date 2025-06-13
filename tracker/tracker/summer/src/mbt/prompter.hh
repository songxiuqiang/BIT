#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace summer
{

	class Prompter
	{

	public:
		Prompter();

		~Prompter();

		static std::shared_ptr<Prompter> Instance()
		{
			std::lock_guard<std::mutex> lock(mtx_);
			if (!instance_)
			{
				instance_ = std::shared_ptr<Prompter>(new Prompter());
			}
			return instance_;
		}

		bool CheckRoi(const cv::Rect &roi, const int &w, const int &h, const int &margin = 0);

		void ExpandRoi(cv::Rect &roi, const float &ratio = 1.0f);

		float ComScaleFactor(const int &max_w, const int &max_h,
							 const int &w, const int &h,
							 const float &base_scale);

		cv::Point2f MaskCenter(const cv::Mat &mask);

		void Segment(const cv::Mat &src, const int &parts, cv::Mat &visual, std::vector<float> &confidence);

		std::vector<std::pair<cv::Mat, float>> Segment(const cv::Mat &src, const cv::Mat &mask, const int &parts, cv::Mat &target, const int &n, std::vector<float> &confidence);

		cv::Mat ComConfidenceMaskFromProb(const cv::Mat &prob, const int &allParts, const int &targetParts);

	private:
		static std::shared_ptr<Prompter> instance_;
		static std::mutex mtx_;

		cv::Mat prob_map_;
	};

}