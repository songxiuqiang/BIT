#pragma once

#include <memory>

#include <opencv2/core.hpp>

#include "object3d.hh"
#include "signed_distance_transform2d.hh"
#include "template_view.hh"
#include "search_line.hh"
#include "prompter.hh"

#include "../base/compatible.h"

namespace summer
{

	class BundleHist;
	class Tracker
	{
	public:
		Tracker(const cv::Matx33f &K, std::vector<Object3D *> &objects);

		static Tracker *GetTracker(const cv::Matx33f &K, const cv::Matx14f &distCoeffs, std::vector<Object3D *> &objects);

		virtual void ToggleTracking(cv::Mat &frame, int objectIndex, bool undistortFrame = true);
		virtual void Track(std::vector<cv::Mat> &imagePyramid, std::vector<Object3D *> &objects, int runs = 1) = 0;
		virtual void EstimatePoses(cv::Mat &frame, bool check_lost);
		virtual void UpdateHist(cv::Mat &frame);
		virtual cv::Mat prob_map(const cv::Mat &frame, const int &objIndex);
		void reset();
		void ChangeModels(std::vector<Object3D *> &objects);

		cv::Rect Compute2DROI(Object3D *object, const cv::Size &maxSize, int offset);
		cv::Rect computeBoundingBox(const std::vector<cv::Point3i> &centersIDs, int offset, int level, const cv::Size &maxSize);

		static void ConvertMask(const cv::Mat &maskm, uchar oid, cv::Mat &mask);
		static void ConvertMask(const cv::Mat &maskm, uchar oid, cv::Rect &roi, cv::Mat &mask);
		static void ShowMask(const cv::Mat &masks, cv::Mat &buf);

		bool initialized() const { return initialized_; }
		cv::Matx33f K() const { return K_; }
		std::shared_ptr<Prompter> prompter() { return prompter_; }

	protected:
		bool initialized_{false};

		cv::Mat prob_map_;

		cv::Matx33f K_;

		std::vector<Object3D *> objects_;

		Renderer *renderer_;

		BundleHist *hists;

		std::shared_ptr<Prompter> prompter_;
	};

	inline float GetDistance(const cv::Point &p1, const cv::Point &p2)
	{
		float dx = float(p1.x - p2.x);
		float dy = float(p1.y - p2.y);
		return sqrt(dx * dx + dy * dy);
	}

	/**
	 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
	 *  computations. Within the corresponding for loop, the RGB values per pixel
	 *  of a color input image are converted to their corresponding histogram bin
	 *  index.
	 */
	class Parallel_For_convertToBins : public cv::ParallelLoopBody
	{
	private:
		cv::Mat _frame;
		cv::Mat _binned;

		uchar *frameData;
		int *binnedData;

		int _numBins;

		int _binShift;

		int _threads;

	public:
		Parallel_For_convertToBins(const cv::Mat &frame, cv::Mat &binned, int numBins, int threads)
		{
			_frame = frame;

			binned.create(_frame.rows, _frame.cols, CV_32SC1);
			_binned = binned;

			frameData = _frame.data;
			binnedData = (int *)_binned.ptr<int>();

			_numBins = numBins;

			_binShift = 8 - log(numBins) / log(2);

			_threads = threads;
		}

		virtual void operator()(const cv::Range &r) const
		{
			int range = _frame.rows / _threads;

			int yEnd = r.end * range;
			if (r.end == _threads)
			{
				yEnd = _frame.rows;
			}

			for (int y = r.start * range; y < yEnd; y++)
			{
				uchar *frameRow = frameData + y * _frame.cols * 3;
				int *binnedRow = binnedData + y * _binned.cols;

				int idx = 0;
				for (int x = 0; x < _frame.cols; x++, idx += 3)
				{
					int ru = (frameRow[idx] >> _binShift);
					int gu = (frameRow[idx + 1] >> _binShift);
					int bu = (frameRow[idx + 2] >> _binShift);

					int binIdx = (ru * _numBins + gu) * _numBins + bu;

					binnedRow[x] = binIdx;
				}
			}
		}
	};

} // ns slot