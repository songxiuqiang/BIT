#pragma once

#include "../tracker_base.h"
#include "../communication.h"

#include <glog/logging.h>
#include <iostream>
#include <sstream>
#include <fstream>

#include <QApplication>
#include <QThread>

#include "tracker/summer/src/mbt/object3d.hh"
#include "tracker/summer/src/mbt/renderer.hh"
#include "tracker/summer/src/mbt/tracker.hh"
#include "ar_utils/timer/timer.h"
#include "definition/global_definition.h"

#include "ar_utils/pose_converter/pose_converter.h"

namespace ar3dv
{

	class SummerTracker : public TrackerBase
	{

	public:
		SummerTracker();

		~SummerTracker();

		void Init() override;

		void StartTracking();

		void Estimate() override;

		void GenRefers();

		void DoNotEstimate() override;

		void SetCameras();

		void SetModels();

		void SetPose(const cv::Matx44f &pose);

		void PrintInfo(const int &flag = 30);

		void ChangeModels(const MeshModel &meshModel);

		cv::Matx33f ComCenteredScaledIntrinsics(const cv::Matx33f &K, const float &w, const float &h, const cv::Rect &roi, const float &scale);

		DataGroup ComDataGroup(const cv::Mat &frame, const cv::Mat &prob, const bool &genRefer = false);

		void ConvertToThreeChannels(cv::Mat &input);
		cv::Mat GetInternalContours(const cv::Mat &inputImage);
		cv::Vec2f ComRoiCenter(const cv::Rect &roi);

	private:
		bool m_startTracking{true};

		int m_failCnts{0};

		cv::Matx14f m_distCoeffs;

		std::vector<float> m_distances;

		std::vector<summer::Object3D *> m_objects;

		summer::Renderer *renderer_;

		summer::Tracker *tracker_;

		cv::Matx33f m_K;
		int m_width{0}, m_height{0};
		float m_fx{0}, m_fy{0}, m_cx{0}, m_cy{0};
		cv::Vec3f prevso3 = cv::Vec3f(0, 0, 0);
		float prevAngle = 0;
	};

} // namespace ar3dv