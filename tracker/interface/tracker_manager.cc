#include "tracker_manager.h"

#include <glog/logging.h>
#include <gflags/gflags.h>

namespace ar3dv
{

	TrackerManager::TrackerManager() {}

	TrackerManager::~TrackerManager()
	{
		if (m_instance)
		{
			VLOG(0) << "delete instance in tracker";
			delete m_instance;
		}
	}

	TrackerManager *TrackerManager::m_instance;
	TrackerManager *TrackerManager::Instance()
	{
		if (m_instance == nullptr)
			m_instance = new TrackerManager();
		return m_instance;
	}

	void TrackerManager::InitTracker()
	{
		m_tracker->Init();
	}

	void TrackerManager::ChangeModels(const MeshModel &meshModel)
	{
		return m_tracker->ChangeModels(meshModel);
	}

	TrackingResult *TrackerManager::GetResult()
	{
		return m_tracker->GetResult();
	}

	void TrackerManager::SetData()
	{
		m_tracker->SetData();
	}

	void TrackerManager::SetPose(const cv::Matx44f &pose)
	{
		m_tracker->SetPose(pose);
	}

	void TrackerManager::ProcessData()
	{
		m_tracker->Estimate();
	}

	void TrackerManager::GenRefers()
	{
		m_tracker->GenRefers();
	}
	void TrackerManager::UnprocessData()
	{
		m_tracker->DoNotEstimate();
	}

	void TrackerManager::SetTrackerType(const TrackerType &trackerType, const bool &use)
	{
		if (!use)
		{
			m_tracker = new NullTracker();
			return;
		}

		switch (trackerType)
		{
		case kSummer:
		{
#ifdef ACTIVE_SUMMER
			m_tracker = new SummerTracker();
#else
			m_tracker = new NullTracker();
#endif
			break;
		}
		default:
		{
			LOG(ERROR) << "Error: Unknown Tracker Type";
			break;
		}
		}
		return;
	}

	void TrackerManager::SetErrorThresh(const float &rErrorThresh, const float &tErrorThresh)
	{
		return m_tracker->SetErrorThresh(rErrorThresh, tErrorThresh);
	}

	void TrackerManager::SetInitPose(const cv::Matx44f &pose)
	{
		m_tracker->SetInitPose(pose);
		return;
	}
} // ns ar3dv