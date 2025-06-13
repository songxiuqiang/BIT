#ifndef _TRACKER_BASE_H_
#define _TRACKER_BASE_H_

#include "system_base.h"

#include <typeinfo>
#include <type_traits>

#include "glog/logging.h"

namespace ar3dv
{

	class TrackerBase : public SystemBase
	{
	public:
		TrackerBase();

		~TrackerBase();

		float GetRoationError(const cv::Matx44f &estimate);

		float GetTranslationError(const cv::Matx44f &estimate);

		virtual TrackingResult *GetResult();

		virtual void ChangeModels(const MeshModel &meshModel);

		virtual void SetData();

		virtual void Init();

		virtual void Estimate();

		virtual void GenRefers();

		virtual void DoNotEstimate();

		virtual bool IsSuccess(const cv::Matx44f &currPose);

		virtual bool IsSuccessByMue(const cv::Matx44f &currPose, const float &diag);

		virtual void SetErrorThresh(const float &rErrorThresh, const float &tErrorThresh);

		virtual void SetPose(const cv::Matx44f &pose);

		virtual void SetZnearZfar();

		virtual void SetInitPose(const cv::Matx44f &initPose);

		virtual cv::Rect CalculateROI(const cv::Mat &mask, const int &extendPixels = 4, const bool &ReturnEmptyWhenBorder = false);

		virtual cv::Vec2f CalculateRatioBetweenROI(const cv::Rect &roi1, const cv::Rect &roi2);

		virtual int SetTrackingResult(const int &resType, const cv::Matx44f &pose);
		virtual int SetTrackingResult(const int &resType, const cv::Matx33f &K);
		virtual int SetTrackingResult(const int &resType, const cv::Rect &roi);
		virtual int SetTrackingResult(const int &resType, const int &value);
		virtual int SetTrackingResult(const int &resType, const float &value);
		virtual int SetTrackingResult(const int &resType, const cv::Mat &img);
		virtual int SetTrackingResult(const int &resType, const std::string &info);
		virtual int SetTrackingResult(const int &resType, const DataGroup &dataGroup);

	protected:
		cv::Matx44f m_initPose = cv::Matx44f::eye();

	private:
		bool m_checkPose{true};
		int m_failCnts{0};

		float m_RErrorThresh{5.0f};
		float m_tErrorThresh{50.0f};

		TrackingResult m_trackingResult;

	}; // class

} // ns ar3dv

#endif // _TRACKER_BASE_H_
