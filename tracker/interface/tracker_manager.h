#ifndef _TRACKER_MANAGER_H_
#define _TRACKER_MANAGER_H_

#include <memory>
#include <mutex>

#include <opencv2/opencv.hpp>

#include "tracker_base.h"
#include "interface/tracker_summer/tracker_summer.h"
#include "interface/tracker_null/tracker_null.h"
#include "definition/global_definition.h"

namespace ar3dv
{

	class TrackerManager
	{
	public:
		TrackerManager();

		~TrackerManager();

		static TrackerManager *Instance();

		void InitTracker();

		void ChangeModels(const MeshModel &meshModel);

		TrackingResult *GetResult();

		void SetData();

		void SetPose(const cv::Matx44f &pose);

		void ProcessData();

		void GenRefers();

		void UnprocessData();

		void SetTrackerType(const TrackerType &trackerType, const bool &use);

		void SetErrorThresh(const float &rErrorThresh, const float &tErrorThresh);

		void SetInitPose(const cv::Matx44f &pose);

	private:
		static TrackerManager *m_instance;

		TrackerBase *m_tracker;

	}; // class

} // ns ar3dv

#endif // _TRACKER_MANAGER_H_