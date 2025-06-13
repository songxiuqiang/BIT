#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <glog/logging.h>
#include "mbt/tclc_histograms.hh"
#include "mbt/search_line.hh"
namespace summer
{

	SearchLine::SearchLine()
	{
		// line_len = Nr;
	}

	void SearchLine::FindContours(const cv::Mat &projection_mask, int seg, bool all_contours)
	{
		if (all_contours)
			cv::findContours(projection_mask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		else
			cv::findContours(projection_mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	}

	void SearchLine::FilterContourPoints(std::vector<std::pair<cv::Mat, float>> masksAndConfidences, const int &n)
	{
		int originSize = contours[0].size();
		for (int i = n; i < masksAndConfidences.size(); ++i)
		{
			if (masksAndConfidences[i].second > 0.5f)
				continue;
			cv::Mat mask = masksAndConfidences[i].first;

			for (auto &contour : contours)
			{
				contour.erase(std::remove_if(contour.begin(), contour.end(), [&mask](const cv::Point &point)
											 { return mask.at<uchar>(point) != 0; }),
							  contour.end());
			}
		}
		return;
	}

	void SearchLine::DrawContours(cv::Mat &buf) const
	{
		std::cout << "contour size: " << contours.size() << std::endl;

		CHECK(!buf.empty());
		static std::vector<cv::Vec3b> colors{
			cv::Vec3b(255, 0, 0),
			cv::Vec3b(0, 255, 0),
			cv::Vec3b(0, 0, 255),
			cv::Vec3b(255, 255, 0),
			cv::Vec3b(160, 32, 240),
			cv::Vec3b(255, 140, 105),
			cv::Vec3b(255, 181, 197)};

		for (int i = 0; i < contours.size(); ++i)
			for (auto pt : contours[i])
			{
				buf.at<cv::Vec3b>(pt) = colors[i];
			}
	}

	void SearchLine::DrawSearchLine(cv::Mat &buf) const
	{
		float alpha = 0.3f;
		for (int i = 0; i < search_points.size(); i++)
		{
			if (actives[i])
				for (int j = 0; j < search_points[i].size() - 1; j++)
				{
					cv::Vec3b &vf = buf.at<cv::Vec3b>(search_points[i][j]);
					cv::Vec3b vc(0, 255, 0);
					buf.at<cv::Vec3b>(search_points[i][j]) = (1.0f - alpha) * vc + alpha * vf;
				}
		}
	}

	void SearchLine::FindSearchLine(const cv::Mat &mask, const cv::Mat &frame, int line_len, int seg, bool use_all)
	{
		search_points.clear();
		norms.clear();
		actives.clear();

		for (int j = 0; j < contours.size(); ++j)
		{
			if (contours[j].size() < 20)
				continue;

			std::vector<cv::Point> ctr_pts;
			for (int i = 0; i < contours[j].size(); i += seg)
			{
				ctr_pts.push_back(contours[j][i]);
			}

			int size = (int)ctr_pts.size();

			cv::Point p1, p2, p3;

			for (int i = 0; i < size; i++)
			{
				int x = ctr_pts[i].x;
				int y = ctr_pts[i].y;

				if (0 == x || frame.cols - 1 == x || 0 == y || frame.rows - 1 == y)
					continue;

				float k = 0.0f;

				int l, r;
				if (0 <= i - 1 && i + 1 < size)
				{
					l = i - 1;
					r = i + 1;
				}
				else
				{
					l = (i - 1 + size) % size;
					r = (i + 1 + size) % size;
				}

				p1.x = ctr_pts[r].x - ctr_pts[i].x;
				p1.y = ctr_pts[r].y - ctr_pts[i].y;
				p2.x = ctr_pts[i].x - ctr_pts[l].x;
				p2.y = ctr_pts[i].y - ctr_pts[l].y;

				float d1 = p1.x * p1.x + p1.y * p1.y;
				float d2 = p2.x * p2.x + p2.y * p2.y;

				float nx = p1.x * d1 + p2.x * d2;
				float ny = p1.y * d1 + p2.y * d2 + 0.0000001f;

				k = -nx / ny;

				std::vector<cv::Point> sl;

				float nl = sqrt(nx * nx + ny * ny);
				cv::Point2f norm(fabs(ny / nl), fabs(nx / nl));

				getLine(k, ctr_pts[i], line_len, mask, sl, norm);

				search_points.push_back(sl);
				norms.push_back(norm);
				actives.push_back(1);
			}
		}
	}

	static bool PtInFrame(const cv::Point &pt, int width, int height)
	{
		return (pt.x < width && pt.y < height && pt.x >= 0 && pt.y >= 0);
	}

	void SearchLine::getLine(float k, const cv::Point &center, int line_len, const cv::Mat &fill_img, std::vector<cv::Point> &points, cv::Point2f &norm)
	{
		static std::vector<cv::Point> decrease;
		static std::vector<cv::Point> increase;
		decrease.resize(0);
		increase.resize(0);

		int width = fill_img.cols;
		int height = fill_img.rows;

		float eps = 0;

		if (k <= 1 && k >= 0)
		{
			int dy = 0;
			for (int dx = 1; dx <= line_len; dx++)
			{
				eps += k;
				if (eps >= 0.5f)
				{
					dy++;
					eps -= 1.0f;
				}

				cv::Point rpt = center + cv::Point(dx, dy);
				if (PtInFrame(rpt, width, height))
					increase.push_back(rpt);

				cv::Point lpt = center - cv::Point(dx, dy);
				if (PtInFrame(lpt, width, height))
					decrease.push_back(lpt);
			}
		}

		if (k > 1)
		{
			int dx = 0;
			for (int dy = 1; dy <= line_len; dy++)
			{
				eps += 1 / k;
				if (eps >= 0.5f)
				{
					dx++;
					eps -= 1.0f;
				}

				cv::Point rpt = center + cv::Point(dx, dy);
				if (PtInFrame(rpt, width, height))
					increase.push_back(rpt);

				cv::Point lpt = center - cv::Point(dx, dy);
				if (PtInFrame(lpt, width, height))
					decrease.push_back(lpt);
			}
		}

		if (k < -1)
		{
			int dx = 0;
			for (int dy = 1; dy <= line_len; dy++)
			{
				eps -= 1 / k;
				if (eps >= 0.5f)
				{
					dx--;
					eps -= 1.0f;
				}

				cv::Point rpt = center + cv::Point(dx, dy);
				if (PtInFrame(rpt, width, height))
					increase.push_back(rpt);

				cv::Point lpt = center - cv::Point(dx, dy);
				if (PtInFrame(lpt, width, height))
					decrease.push_back(lpt);
			}
		}

		if (k >= -1 && k < 0)
		{
			int dy = 0;
			for (int dx = 1; dx <= line_len; dx++)
			{
				eps -= k;
				if (eps >= 0.5f)
				{
					dy--;
					eps -= 1.0f;
				}

				cv::Point rpt = center + cv::Point(dx, dy);
				if (PtInFrame(rpt, width, height))
					increase.push_back(rpt);

				cv::Point lpt = center - cv::Point(dx, dy);
				if (PtInFrame(lpt, width, height))
					decrease.push_back(lpt);
			}
		}

		uchar i_dist = 0, d_dist = 0;
		if (increase.size() > 1 && decrease.size() > 1)
		{
			i_dist = fill_img.at<uchar>(cv::Point(increase[1].x, increase[1].y));
			d_dist = fill_img.at<uchar>(cv::Point(decrease[1].x, decrease[1].y));
		}
		else if (increase.size() > 1 && decrease.size() == 0)
		{
			i_dist = fill_img.at<uchar>(cv::Point(increase[1].x, increase[1].y));
			if (i_dist > 0)
				d_dist = 0;
			else
				d_dist = 255;
		}
		else if (increase.size() == 0 && decrease.size() > 1)
		{
			d_dist = fill_img.at<uchar>(cv::Point(decrease[1].x, decrease[1].y));
			if (d_dist > 0)
				i_dist = 0;
			else
				i_dist = 255;
		}

		if (i_dist > d_dist)
		{
			for (int i = decrease.size() - 1; i >= 0; i--)
				points.push_back(decrease[i]);

			points.push_back(center);

			for (int i = 0; i < increase.size(); i++)
				points.push_back(increase[i]);

			points.push_back(cv::Point(decrease.size(), 0));
			if (k <= 1 && k >= 0)
			{
				norm.x = -norm.x;
				norm.y = -norm.y;
			}
			else if (k > 1)
			{
				norm.x = -norm.x;
				norm.y = -norm.y;
			}
			else if (k < -1)
			{
				norm.y = -norm.y;
			}
			else if (k >= -1 && k < 0)
			{
				norm.x = -norm.x;
			}
		}
		else
		{
			for (int i = increase.size() - 1; i >= 0; i--)
				points.push_back(increase[i]);

			points.push_back(center);

			for (int i = 0; i < decrease.size(); i++)
				points.push_back(decrease[i]);

			points.push_back(cv::Point(increase.size(), 0));

			if (k < -1)
			{
				norm.x = -norm.x;
			}
			else if (k >= -1 && k < 0)
			{
				norm.y = -norm.y;
			}
		}
	}

} // ns slot