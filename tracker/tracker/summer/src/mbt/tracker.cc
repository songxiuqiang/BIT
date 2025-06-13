#include <iomanip>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>
#include "mbt/renderer.hh"
#include "mbt/bundel_hist.hh"
#include "tracker.hh"
#include "base/global_param.hh"
#include "mbt/object3d.hh"
#include "mbt/search_line.hh"
#include "mbt/tracker_slc.hh"

namespace summer
{

	Tracker::Tracker(const cv::Matx33f &K, std::vector<Object3D *> &objects)
	{
		initialized_ = false;
		K_ = K;
		renderer_ = Renderer::Instance();

		for (size_t i = 0; i < objects.size(); i++)
		{
			objects[i]->setModelID(i + 1);
			objects_.push_back(objects[i]);
			objects_[i]->initBuffers();
			objects_[i]->reset();
		}

		hists = new RBOTHist(objects);
	}

	Tracker *Tracker::GetTracker(const cv::Matx33f &K, const cv::Matx14f &distCoeffs, std::vector<Object3D *> &objects)
	{
		Tracker *poseEstimator = NULL;
		tk::GlobalParam *gp = tk::GlobalParam::Instance();
		poseEstimator = new SLCTracker(K, objects);
		CHECK(poseEstimator) << "Check |tracker_mode| in yml file";
		return poseEstimator;
	}

	void Tracker::ChangeModels(std::vector<Object3D *> &objs)
	{
		for (size_t i = 0; i < objects_.size(); i++)
		{
			objects_[i]->reset();
		}
		initialized_ = false;
		objects_.clear();

		for (size_t i = 0; i < objs.size(); i++)
		{
			objs[i]->setModelID(i + 1);
			objects_.push_back(objs[i]);
			objects_[i]->initBuffers();
			objects_[i]->reset();
		}
		return;
	}

	void Tracker::reset()
	{
		for (size_t i = 0; i < objects_.size(); i++)
			objects_[i]->reset();
		initialized_ = false;
	}

	cv::Rect Tracker::Compute2DROI(Object3D *object, const cv::Size &maxSize, int offset)
	{
		cv::Rect boundingRect;
		std::vector<cv::Point2f> projections;

		renderer_->ProjectBoundingBox(object, projections, boundingRect);

		if (boundingRect.x >= maxSize.width || boundingRect.y >= maxSize.height || boundingRect.x + boundingRect.width <= 0 || boundingRect.y + boundingRect.height <= 0)
		{
			return cv::Rect(0, 0, 0, 0);
		}

		cv::Rect roi = cv::Rect(boundingRect.x - offset, boundingRect.y - offset, boundingRect.width + 2 * offset, boundingRect.height + 2 * offset);

		if (roi.x < 0)
		{
			roi.width += roi.x;
			roi.x = 0;
		}

		if (roi.y < 0)
		{
			roi.height += roi.y;
			roi.y = 0;
		}

		if (roi.x + roi.width > maxSize.width)
			roi.width = maxSize.width - roi.x;
		if (roi.y + roi.height > maxSize.height)
			roi.height = maxSize.height - roi.y;

		return roi;
	}

	cv::Rect Tracker::computeBoundingBox(const std::vector<cv::Point3i> &centersIDs, int offset, int level, const cv::Size &maxSize)
	{
		int minX = INT_MAX, minY = INT_MAX;
		int maxX = -1, maxY = -1;

		for (int i = 0; i < centersIDs.size(); i++)
		{
			cv::Point3i p = centersIDs[i];
			int x = p.x / pow(2, level);
			int y = p.y / pow(2, level);

			if (x < minX)
				minX = x;
			if (y < minY)
				minY = y;
			if (x > maxX)
				maxX = x;
			if (y > maxY)
				maxY = y;
		}

		minX -= offset;
		minY -= offset;
		maxX += offset;
		maxY += offset;

		if (minX < 0)
			minX = 0;
		if (minY < 0)
			minY = 0;
		if (maxX > maxSize.width)
			maxX = maxSize.width;
		if (maxY > maxSize.height)
			maxY = maxSize.height;

		return cv::Rect(minX, minY, maxX - minX, maxY - minY);
	}

	void Tracker::ConvertMask(const cv::Mat &src_mask, uchar oid, cv::Mat &mask)
	{
		mask = cv::Mat(src_mask.size(), CV_8UC1, cv::Scalar(0));
		uchar depth = src_mask.type() & CV_MAT_DEPTH_MASK;

		if (CV_8U == depth && oid > 0)
		{
			for (int r = 0; r < src_mask.rows; ++r)
				for (int c = 0; c < src_mask.cols; ++c)
				{
					if (oid == src_mask.at<uchar>(r, c))
						mask.at<uchar>(r, c) = 255;
				}
		}
		else if (CV_32F == depth)
		{
			for (int r = 0; r < src_mask.rows; ++r)
				for (int c = 0; c < src_mask.cols; ++c)
				{
					if (src_mask.at<float>(r, c))
						mask.at<uchar>(r, c) = 255;
				}
		}
		else
		{
			LOG(ERROR) << "WRONG IMAGE TYPE";
		}
	}

	void Tracker::ConvertMask(const cv::Mat &src_mask, uchar oid, cv::Rect &roi, cv::Mat &mask)
	{
		mask = cv::Mat(src_mask.size(), CV_8UC1, cv::Scalar(0));
		uchar depth = src_mask.type() & CV_MAT_DEPTH_MASK;

		cv::Mat roi_src_mask = src_mask(roi);
		cv::Mat roi_mask = mask(roi);

		if (CV_8U == depth && oid > 0)
		{
			for (int r = 0; r < roi_src_mask.rows; ++r)
				for (int c = 0; c < roi_src_mask.cols; ++c)
				{
					if (oid == roi_src_mask.at<uchar>(r, c))
						roi_mask.at<uchar>(r, c) = 255;
				}
		}
		else if (CV_32F == depth)
		{
			for (int r = 0; r < roi_src_mask.rows; ++r)
				for (int c = 0; c < roi_src_mask.cols; ++c)
				{
					if (roi_src_mask.at<float>(r, c))
						roi_mask.at<uchar>(r, c) = 255;
				}
		}
		else
		{
			LOG(ERROR) << "WRONG IMAGE TYPE";
		}
	}

	void Tracker::ShowMask(const cv::Mat &masks, cv::Mat &buf)
	{
		uchar depth = masks.type() & CV_MAT_DEPTH_MASK;

		if (CV_8U == depth)
		{
			for (int r = 0; r < masks.rows; ++r)
				for (int c = 0; c < masks.cols; ++c)
				{
					if (1 == masks.at<uchar>(r, c))
					{
						buf.at<cv::Vec3b>(r, c)[0] = 255;
						buf.at<cv::Vec3b>(r, c)[1] = 255;
						buf.at<cv::Vec3b>(r, c)[2] = 255;
					}
					else if (2 == masks.at<uchar>(r, c))
					{
						buf.at<cv::Vec3b>(r, c)[0] = 128;
						buf.at<cv::Vec3b>(r, c)[1] = 128;
						buf.at<cv::Vec3b>(r, c)[2] = 128;
					}
				}
		}
		else
		{
			LOG(ERROR) << "WRONG IMAGE TYPE";
		}
	}

	void Tracker::ToggleTracking(cv::Mat &frame, int objectIndex, bool undistortFrame)
	{
		if (objectIndex >= objects_.size())
		{
			VLOG(0) << "object index " << objectIndex << " must <= " << objects_.size();
			return;
		}

		if (!objects_[objectIndex]->isInitialized())
		{
			objects_[objectIndex]->initialize();

			renderer_->setLevel(0);
			renderer_->RenderSilhouette(std::vector<Model *>(objects_.begin(), objects_.end()), GL_FILL);
			cv::Mat mask_map = renderer_->DownloadFrame(Renderer::MASK);
			cv::Mat depth_map = renderer_->DownloadFrame(Renderer::DEPTH);

			VLOG(0) << "Update hists...";
			hists->Update(frame, mask_map, depth_map, objectIndex, 0.1, 0.2);

			initialized_ = true;
		}
		else
		{
			objects_[objectIndex]->reset();

			initialized_ = false;
			for (int o = 0; o < objects_.size(); o++)
			{
				initialized_ |= objects_[o]->isInitialized();
			}
		}
	}

	cv::Mat Tracker::prob_map(const cv::Mat &frame, const int &objIndex)
	{
		if (!prob_map_.empty())
			return prob_map_;
		else
		{
			VLOG(0) << "Warning: Prob map is empty.";
			return cv::Mat(1, 1, 0);
		}
	}

	void ResizeByNN(uchar *src, uchar *dst,
					const int &height_in, const int &width_in,
					const int &channels,
					const int &height_out, const int &width_out)
	{
		int srcColsElementNums = width_in * 3;
		int dstColsElementNums = width_out * 3;

		int rowsOffsetInSrc = 0;
		int colsOffsetInSrc = 0.0;

		uchar *firstElementOfRowsInSrc = nullptr;
		uchar *elementOfRowsInDst = nullptr;
		float ratioInCols = float(width_in) / float(width_out);
		float ratioInRows = float(height_in) / float(height_out);

		for (int i = 0; i < height_out; i++)
		{
			for (int j = 0; j < width_out; j++)
			{
				rowsOffsetInSrc = int(ratioInRows * i);
				colsOffsetInSrc = int(ratioInCols * j) * channels;
				firstElementOfRowsInSrc = src + rowsOffsetInSrc * srcColsElementNums;
				elementOfRowsInDst = dst + i * dstColsElementNums + j * channels;
				memcpy(elementOfRowsInDst, firstElementOfRowsInSrc + colsOffsetInSrc, channels);
			}
		}
		return;
	}

	void Tracker::EstimatePoses(cv::Mat &frame, bool check_lost)
	{
		cv::Mat pym0, pym1, pym2;
		std::vector<cv::Mat> imagePyramid;
		pym0 = frame.clone();
		cv::resize(frame, pym1, cv::Size(frame.cols / 2, frame.rows / 2));
		cv::resize(frame, pym2, cv::Size(frame.cols / 4, frame.rows / 4));
		imagePyramid.push_back(pym0);
		imagePyramid.push_back(pym1);
		imagePyramid.push_back(pym2);

		float afg = 0.1, abg = 0.2;
		if (initialized_)
		{
			Track(imagePyramid, objects_);
		}
	}

	void Tracker::UpdateHist(cv::Mat &frame)
	{
		float afg = 0.1, abg = 0.2;
		if (initialized_)
		{
			renderer_->setLevel(0);
			renderer_->RenderSilhouette(std::vector<Model *>(objects_.begin(), objects_.end()), GL_FILL);
			cv::Mat masks_map = renderer_->DownloadFrame(Renderer::MASK);
			cv::Mat depth_map = renderer_->DownloadFrameFromROI(Renderer::DEPTH, objects_[0]);

			for (int oid = 0; oid < objects_.size(); oid++)
			{
				hists->Update(frame, masks_map, depth_map, oid, afg, abg);
			}
		}
	}

} // ns summer