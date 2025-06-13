#include "interface.h"

#include <glog/logging.h>
#include <gflags/gflags.h>

namespace ar3dv
{

	void InitSystem()
	{
		return SystemManager::Instance().InitSystem();
	}

	void ProcessData()
	{
		return SystemManager::Instance().ProcessData();
	}

	void GenRefers()
	{
		return SystemManager::Instance().GenRefers();
	}

	void UnprocessData()
	{
		return SystemManager::Instance().UnprocessData();
	}

	void PushData(StdData *stdData)
	{
		return SystemManager::Instance().PushData(stdData);
	}

	void SetCameras(const CamParams &camParams)
	{
		return SystemManager::Instance().SetCameras(camParams);
	}

	void SetErrorThresh(const float &rErrorThresh, const float &tErrorThresh)
	{
		return SystemManager::Instance().SetErrorThresh(rErrorThresh, tErrorThresh);
	}

	void ChangeModels(const MeshModel &meshModel)
	{
		return SystemManager::Instance().ChangeModels(meshModel);
	}

	void SetModels(const MeshModel &meshModel)
	{
		return SystemManager::Instance().SetModels(meshModel);
	}

	void SetPose(const cv::Matx44f &pose)
	{
		return SystemManager::Instance().SetPose(pose);
	}

	void SetZnearZfar(const float &zn, const float &zf)
	{
		return SystemManager::Instance().SetZnearZfar(zn, zf);
	}

	void SetDetectorType(const DetectorType &detectorType, const bool &use)
	{
		return SystemManager::Instance().SetDetectorType(detectorType, use);
	}

	void SetTrackerType(const TrackerType &trackerType, const bool &use)
	{
		return SystemManager::Instance().SetTrackerType(trackerType, use);
	}

	ProcessResult *GetResult()
	{
		return SystemManager::Instance().GetResult();
	}

} // ns ar3dv