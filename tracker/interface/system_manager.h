#ifndef _SYSTEM_MANAGER_H_
#define _SYSTEM_MANAGER_H_

#include <memory>
#include <mutex>

#include <opencv2/opencv.hpp>

#include "tracker_manager.h"

namespace ar3dv
{

	class SystemManager
	{
	public:
		SystemManager();

		~SystemManager();

		static SystemManager &Instance();

		void InitSystem();

		void PushData(StdData *stdData);

		void ProcessData();

		void GenRefers();

		void UnprocessData();

		void SetCameras(const CamParams &camParams);

		void SetErrorThresh(const float &rErrorThresh, const float &tErrorThresh);

		void ChangeModels(const MeshModel &meshModel);

		void SetModels(const MeshModel &meshModel);

		void SetPose(const cv::Matx44f &pose);

		void SetZnearZfar(const float &zn, const float &zf);

		void SetDetectorType(const DetectorType &detectorType, const bool &use = true);

		void SetTrackerType(const TrackerType &trackerType, const bool &use = true);

		ProcessResult *GetResult();

	private:
	}; // class

} // ns ar3dv

#endif // _SYSTEM_MANAGER_H_