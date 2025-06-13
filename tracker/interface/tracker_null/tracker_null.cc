#include "tracker_null.h"

using namespace ar3dv;

NullTracker::NullTracker()
{
	VLOG(0) << "NullTracker used, this tracker will do nothong";
}

NullTracker::~NullTracker() {}

void NullTracker::Init() {}

void NullTracker::StartTracking() {}

void NullTracker::Estimate()
{
	cv::Mat overlay = cv::Mat::zeros(cv::Size(200, 200), CV_8UC3);
	SetTrackingResult(ResultType::kResOverlay, overlay);
}

void NullTracker::SetData() {}

void NullTracker::SetModels() {}

void NullTracker::SetCameras() {}