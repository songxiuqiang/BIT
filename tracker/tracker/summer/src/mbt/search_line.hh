#ifndef __SUMMER_SEARCH_LINE_H__
#define __SUMMER_SEARCH_LINE_H__
#include <vector>

#include "../base/compatible.h"
namespace summer
{

	class SearchLine
	{
	public:
		SearchLine();

		void FindContours(const cv::Mat &projection_mask, int seg, bool all_contours);

		void FindSearchLine(const cv::Mat &mask, const cv::Mat &frame, int len, int seg, bool use_all);
		void DrawSearchLine(cv::Mat &line_mask) const;
		void DrawContours(cv::Mat &contour_mask) const;
		void FilterContourPoints(std::vector<std::pair<cv::Mat, float>> masksAndConfidences, const int &n);

		std::vector<std::vector<cv::Point>> contours;
		std::vector<std::vector<cv::Point>> search_points;
		std::vector<std::vector<cv::Point2f>> bundle_prob;
		std::vector<uchar> actives;
		std::vector<cv::Point2f> norms;

	protected:
		void getLine(float k, const cv::Point &center, int len, const cv::Mat &mask, std::vector<cv::Point> &search_points, cv::Point2f &norm);
	};

} // ns summer

#endif // __SUMMER_SEARCH_LINE_H__