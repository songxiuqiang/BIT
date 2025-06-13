#include "system_manager.h"

#include <glog/logging.h>
#include <gflags/gflags.h>

using namespace ar3dv;

SystemManager::SystemManager() {}

SystemManager::~SystemManager() {}

SystemManager &SystemManager::Instance()
{
	static SystemManager instance;
	return instance;
}

void SystemManager::InitSystem()
{
	TrackerManager::Instance()->InitTracker();
}

ProcessResult *SystemManager::GetResult()
{

	TrackerManager::Instance()->GetResult();
	return SystemBase::Instance()->GetResult();
}

void SystemManager::PushData(StdData *stdData)
{
	SystemBase::Instance()->SetData(stdData);
	TrackerManager::Instance()->SetData();
	return;
}

void SystemManager::ProcessData()
{
	TrackerManager::Instance()->ProcessData();
}

void SystemManager::GenRefers()
{
	TrackerManager::Instance()->GenRefers();
}

void SystemManager::UnprocessData()
{
	TrackerManager::Instance()->UnprocessData();
}

void SystemManager::SetCameras(const CamParams &camParams)
{
	SystemBase::Instance()->SetCameras(camParams);
}

void SystemManager::SetErrorThresh(const float &rErrorThresh, const float &tErrorThresh)
{
	TrackerManager::Instance()->SetErrorThresh(rErrorThresh, tErrorThresh);
}

void SystemManager::ChangeModels(const MeshModel &meshModel)
{
	return TrackerManager::Instance()->ChangeModels(meshModel);
}

void SystemManager::SetModels(const MeshModel &meshModel)
{
	SystemBase::Instance()->SetModels(meshModel);
}

void SystemManager::SetPose(const cv::Matx44f &pose)
{
	TrackerManager::Instance()->SetPose(pose);
	return;
}

void SystemManager::SetZnearZfar(const float &zn, const float &zf)
{
	SystemBase::Instance()->SetZnearZfar(zn, zf);
}

void SystemManager::SetDetectorType(const DetectorType &detectorType, const bool &use)
{
	SystemBase::Instance()->m_useDetector = use;
	return;
}

void SystemManager::SetTrackerType(const TrackerType &trackerType, const bool &use)
{
	TrackerManager::Instance()->SetTrackerType(trackerType, use);
	SystemBase::Instance()->m_useTracker = use;
	return;
}
