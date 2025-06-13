#include "prompter.hh"

#include "glog/logging.h"
namespace summer
{

	// 初始化静态成员变量
	std::shared_ptr<Prompter> Prompter::instance_ = nullptr;
	std::mutex Prompter::mtx_;

	Prompter::Prompter()
	{
	}

	Prompter::~Prompter()
	{
	}

	bool Prompter::CheckRoi(const cv::Rect &roi, const int &w, const int &h, const int &margin)
	{
		if (roi.x <= margin || roi.y <= margin || roi.x + roi.width >= (int)(w - margin) || roi.y + roi.height >= (int)(h - margin))
		{
			VLOG(0) << "Invalid ROI " << roi.x << " " << roi.y << " " << roi.x + roi.width << " " << roi.height;
			return false;
		}
		return true;
	}

	void Prompter::ExpandRoi(cv::Rect &roi, const float &ratio)
	{
		float dx = float(roi.width) * (ratio - 1.0f);
		float dy = float(roi.height) * (ratio - 1.0f);
		roi.x -= dx;
		roi.y -= dy;
		roi.width += (2 * dx);
		roi.height += (2 * dy);
		return;
	}

	float Prompter::ComScaleFactor(const int &max_w, const int &max_h,
								   const int &w, const int &h,
								   const float &base_scale)
	{
		float scale_factor = base_scale * std::min(float(max_w) / float(w), float(max_h) / float(h));
		if (scale_factor < 1.1f)
			scale_factor = 1.0f;
		return scale_factor;
	}

	cv::Point2f Prompter::MaskCenter(const cv::Mat &mask)
	{
		cv::Mat bin;
		cv::threshold(mask, bin, 0, 255, cv::THRESH_BINARY);
		int left, right, top, down;
		for (int i = 0; i < bin.rows; i++)
		{
			if (cv::countNonZero(bin.row(i)) != 0)
			{
				top = i;
				break;
			}
		}
		for (int i = bin.rows - 1; i > 0; i--)
		{
			if (cv::countNonZero(bin.row(i)) != 0)
			{
				down = i;
				break;
			}
		}
		for (int i = 0; i < bin.cols; i++)
		{
			if (cv::countNonZero(bin.col(i)) != 0)
			{
				left = i;
				break;
			}
		}
		for (int i = bin.cols - 1; i > 0; i--)
		{
			if (cv::countNonZero(bin.col(i)) != 0)
			{
				right = i;
				break;
			}
		}
		float xCenter = float(left) + 0.5f * float(right - left);
		float yCenter = float(top) + 0.5f * float(down - top);
		return cv::Point2f(xCenter, yCenter);
	}

	cv::Mat Prompter::ComConfidenceMaskFromProb(const cv::Mat &prob, const int &allParts, const int &targetParts)
	{
		if (allParts % 4 != 0)
		{
			throw std::invalid_argument("parts must be a multiple of 4");
		}

		cv::Mat res = cv::Mat::zeros(prob.size(), CV_8UC1);

		std::vector<std::pair<float, int>> confidences;
		confidences.reserve(allParts);

		cv::Point2f center(prob.cols / 2.0f, prob.rows / 2.0f);

		float angleStep = 360.0f / allParts;

		for (int i = 0; i < allParts; ++i)
		{
			float angle1 = i * angleStep;
			float angle2 = (i + 1) * angleStep;

			cv::Mat mask = cv::Mat::zeros(prob.size(), CV_8UC1);

			std::vector<cv::Point> points;
			points.push_back(center);

			for (float angle = angle1; angle <= angle2; angle += 1.0f)
			{
				float rad = angle * CV_PI / 180.0f;
				float x = center.x + prob.cols * cos(rad);
				float y = center.y + prob.rows * sin(rad);
				points.push_back(cv::Point(cvRound(x), cvRound(y)));
			}

			cv::fillConvexPoly(mask, points, cv::Scalar(255));

			cv::Mat part;
			prob.copyTo(part, mask);

			double sumConfidence = 0.0;
			int count = 0;

			for (int y = 0; y < part.rows; ++y)
			{
				for (int x = 0; x < part.cols; ++x)
				{
					if (mask.at<uchar>(y, x) > 0 && part.at<uchar>(y, x) != 0)
					{
						float value = float(part.at<uchar>(y, x)) / 255.0;
						float conf = pow((std::abs(value - 0.5f) * 2.0f), 2.0f);
						sumConfidence += conf;
						++count;
					}
				}
			}
			if (count > 0)
			{
				confidences.push_back({static_cast<float>(sumConfidence / count), i});
			}
		}

		std::partial_sort(confidences.begin(), confidences.begin() + targetParts, confidences.end(),
						  std::greater<std::pair<float, int>>());

		for (int j = 0; j < allParts; ++j)
		{
			if (confidences[j].first < 0.55f)
				continue;
			int maxIndex = confidences[j].second;
			float angle1 = maxIndex * angleStep;
			float angle2 = (maxIndex + 1) * angleStep;

			std::vector<cv::Point> targetPoints;
			targetPoints.push_back(center);

			for (float angle = angle1; angle <= angle2; angle += 1.0f)
			{
				float rad = angle * CV_PI / 180.0f;
				float x = center.x + prob.cols * cos(rad);
				float y = center.y + prob.rows * sin(rad);
				targetPoints.push_back(cv::Point(cvRound(x), cvRound(y)));
			}

			cv::fillConvexPoly(res, targetPoints, cv::Scalar(255));
		}

		cv::Mat resColor;
		cv::cvtColor(res, resColor, cv::COLOR_GRAY2BGR);

		for (int y = 0; y < resColor.rows; y++)
		{
			for (int x = 0; x < resColor.cols; x++)
			{
				if (resColor.at<cv::Vec3b>(y, x)[0] == 255)
				{
					resColor.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
				}
			}
		}

		cv::Mat probColor;
		cv::cvtColor(prob, probColor, cv::COLOR_GRAY2BGR);

		cv::Mat combined;
		cv::addWeighted(probColor, 0.7, resColor, 0.3, 0, combined);

		return res;
	}

	void Prompter::Segment(const cv::Mat &src, const int &parts, cv::Mat &visual, std::vector<float> &confidence)
	{
		// Check if parts is a multiple of 4
		if (parts % 4 != 0)
		{
			throw std::invalid_argument("parts must be a multiple of 4");
		}

		// Initialize the confidence vector
		confidence.resize(parts, 0.0f);

		// Create a visual image with the same size as src
		visual = cv::Mat::zeros(src.size(), CV_8UC3);

		// Get the center of the image
		cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);

		// Calculate the angle step
		float angleStep = 360.0f / parts;

		for (int i = 0; i < parts; ++i)
		{
			// Calculate the angles for the current part
			float angle1 = i * angleStep;
			float angle2 = (i + 1) * angleStep;

			// Create a mask for the current part
			cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);

			std::vector<cv::Point> points;
			points.push_back(center);

			for (float angle = angle1; angle <= angle2; angle += 1.0f)
			{
				float rad = angle * CV_PI / 180.0f;
				float x = center.x + src.cols * cos(rad);
				float y = center.y + src.rows * sin(rad);
				points.push_back(cv::Point(cvRound(x), cvRound(y)));
			}

			cv::fillConvexPoly(mask, points, cv::Scalar(255));

			// Calculate the confidence for the current part
			cv::Mat part;
			src.copyTo(part, mask);

			double sumConfidence = 0.0;
			int count = 0;

			for (int y = 0; y < part.rows; ++y)
			{
				for (int x = 0; x < part.cols; ++x)
				{
					if (mask.at<uchar>(y, x) > 0 && part.at<uchar>(y, x) != 0)
					{
						float value = float(part.at<uchar>(y, x)) / 255.0;
						// float conf = std::abs(value - 0.5f) * 2.0f;
						float conf = pow((std::abs(value - 0.5f) * 2.0f), 2.0f);
						// VLOG(0) << "value conf " << value <<  " " << conf;
						sumConfidence += conf;
						++count;
					}
				}
			}
			if (count > 0)
			{
				confidence[i] = static_cast<float>(sumConfidence / count);
			}

			// Draw the part on the visual image
			cv::Scalar color = cv::Scalar(0, 255 * (1.0f - confidence[i]), 255 * confidence[i]);
			cv::fillConvexPoly(visual, points, color);

			cv::imshow("visual", visual);
			cv::imshow("mask parts", mask);
			cv::imshow("part", part);
			// cv::waitKey(0);
		}
	}

	std::vector<std::pair<cv::Mat, float>> Prompter::Segment(const cv::Mat &src, const cv::Mat &mask_input, const int &parts, cv::Mat &target, const int &n, std::vector<float> &confidence)
	{
		// Check if parts is a multiple of 4
		if (parts % 4 != 0)
		{
			throw std::invalid_argument("parts must be a multiple of 4");
		}

		// Initialize the confidence vector
		confidence.resize(parts, 0.0f);

		// Create a visual image with the same size as src
		cv::Mat visual = cv::Mat::zeros(src.size(), CV_8UC3);

		// Get the center of the image
		cv::Point2f center = MaskCenter(mask_input);

		// Calculate the angle step
		float angleStep = 360.0f / parts;

		// Create a vector to store masks and their corresponding confidence
		std::vector<std::pair<cv::Mat, float>> masksAndConfidences;

		for (int i = 0; i < parts; ++i)
		{
			// Calculate the angles for the current part
			float angle1 = i * angleStep;
			float angle2 = (i + 1) * angleStep;

			// Create a mask for the current part
			cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
			std::vector<cv::Point> points;
			points.push_back(center);

			for (float angle = angle1; angle <= angle2; angle += 1.0f)
			{
				float rad = angle * CV_PI / 180.0f;
				float x = center.x + src.cols * cos(rad);
				float y = center.y + src.rows * sin(rad);
				points.push_back(cv::Point(cvRound(x), cvRound(y)));
			}

			cv::fillConvexPoly(mask, points, cv::Scalar(255));

			// Calculate the confidence for the current part
			cv::Mat part;
			src.copyTo(part, mask);

			double sumConfidence = 0.0;
			int count = 0;

			for (int y = 0; y < part.rows; ++y)
			{
				for (int x = 0; x < part.cols; ++x)
				{
					if (mask.at<uchar>(y, x) > 0 && part.at<uchar>(y, x) != 0)
					{
						float value = float(part.at<uchar>(y, x)) / 255.0;
						float conf = std::abs(value - 0.5f) * 2.0f;
						sumConfidence += conf;
						++count;
					}
				}
			}
			if (count > 0)
			{
				confidence[i] = static_cast<float>(sumConfidence / count);
			}

			// Store the mask and its confidence
			masksAndConfidences.push_back(std::make_pair(mask, confidence[i]));

			// Draw the part on the visual image
			cv::Scalar color = cv::Scalar(0, 255 * (1.0f - confidence[i]), 255 * confidence[i]);
			cv::fillConvexPoly(visual, points, color);

			cv::imshow("visual", visual);
		}

		// Sort the parts by confidence in descending order
		std::sort(masksAndConfidences.begin(), masksAndConfidences.end(), [](const std::pair<cv::Mat, float> &a, const std::pair<cv::Mat, float> &b)
				  { return a.second > b.second; });

		return masksAndConfidences;
	}

}