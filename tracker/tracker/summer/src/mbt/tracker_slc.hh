#pragma once

#include "mbt/tracker.hh"

#include "../base/compatible.h"

namespace summer
{

	class SLCTracker : public Tracker
	{
	public:
		SLCTracker(const cv::Matx33f &K, std::vector<Object3D *> &objects);

		~SLCTracker();

		virtual void Track(std::vector<cv::Mat> &imagePyramid, std::vector<Object3D *> &objects, int runs = 1) override;

		void RunIteration(std::vector<Object3D *> &objects, const std::vector<cv::Mat> &imagePyramid, int level, int sl_len, int sl_seg, float band_width, float ss, int run_type = 0);

		void ComputeJac(Object3D *object, int m_id, const cv::Mat &frame, const cv::Mat &mask_map, const cv::Mat &masks_map, const cv::Mat &depth_map, const cv::Mat &depth_inv_map, SearchLine *search_line, cv::Matx66f &wJTJM, cv::Matx61f &JTM, float band_width, float ss);

		void UpdateHist(cv::Mat &frame);

		void FilterOccludedPoint(const cv::Mat &mask, const cv::Mat &depth, SearchLine *sl);

		cv::Mat prob_map(const cv::Mat &frame, const int &objIndex) override;

		void FindMatchPoint(SearchLine *search_line, float diff);

		bool UseLargerImage(int &numInitialized, int &level, const int &width, const int &height, const bool &use);

		cv::Mat SetMask(const int &numInitialized, const cv::Mat &depth);

		cv::Rect compute2DROI(Object3D *object, const cv::Size &maxSize, int offset);

	private:
		SearchLine *search_line;

		std::vector<float> scores;
	};

} // ns slot