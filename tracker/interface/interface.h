#pragma once

#include <stddef.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <stddef.h>
#include <opencv2/opencv.hpp>

#include "definition/global_definition.h"
#include "system_manager.h"

#ifdef __cplusplus
extern "C"
{
#endif

	namespace ar3dv
	{

		void InitSystem();

		ProcessResult *GetResult();

		void PushData(StdData *stdData);

		void ProcessData();

		void GenRefers();

		void UnprocessData();

		void SetCameras(const CamParams &camParams);

		void SetErrorThresh(const float &rErrorThresh, const float &tErrorThresh);

		void SetModels(const MeshModel &meshModel);

		void ChangeModels(const MeshModel &meshModel);

		void SetPose(const cv::Matx44f &pose);

		void SetZnearZfar(const float &zn, const float &zf);

		void SetDetectorType(const DetectorType &detectorType, const bool &use = true);

		void SetTrackerType(const TrackerType &trackerType, const bool &use = true);

	} // ns ar3dv

#ifdef __cplusplus
}
#endif
