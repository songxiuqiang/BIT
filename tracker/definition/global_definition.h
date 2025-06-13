#pragma once

#include "global_controller.h"

#include "opencv2/opencv.hpp"
#include <eigen3/Eigen/Eigen>

#if defined(ANDROID) || defined(__ANDOIRD__)
#define MOBILE_PLATFORM
#elif defined(_WIN32) || defined(__linux__)
#define DESKTOP_PLATFORM
#endif


struct StdData      // standard data
{
	int sourceId;	// data source ID
	int index;
	cv::Mat frame;
	cv::Mat depth;
	cv::Mat mask;
	cv::Matx44f gt;
    float poseScale;
	float timeStamp;
};

struct CamParams
{
	int sourceId;   // cam ID
	int width;
	int height;
	float fx;
	float fy;
	float cx;
	float cy;
};

struct MeshModel
{
	int id;
	std::string path;
};

enum FunctionalModule
{
	kTracker = 0,
	kDetector,
	kReconstructor
};

struct DataGroup
{	
	bool valid{false};
	cv::Mat img{cv::Mat::zeros(1,1,CV_8UC3)};
	cv::Mat prob{cv::Mat::zeros(1,1,CV_16UC1)};
	cv::Mat mask{cv::Mat::zeros(1,1,CV_8UC1)};
	cv::Matx33f K{0,0,0,0,0,0,0,0,0};
	cv::Matx44f pose{2.0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
	cv::Rect originRoi{0,0,0,0};
	cv::Rect targetRoi{0,0,0,0};
	cv::Vec2f targetSize{0,0};
	int targetWidth{0};
	int targetHeight{0};
	Eigen::Vector3f view{0,0,0};
};

enum DetectorType
{
	kLinemod = 0,
	kYOLOv6 = 1,
	kFDP = 2
};

enum TrackerType
{
    kEDF = 0,
    kSearchLine = 1,
	kRBOT = 2,
	kSLOT = 3,
    kAWLB = 4,
    kIP = 5,
    kRBGT = 6,
    kSRT3D = 7,
    kICG = 8,
	kRMT = 9,
	kSummer = 10
};

enum ResultType
{
	kResIndex = 0,
	kCamParams,
	kResGt,
	kResPose,
	kResFilteredPose,
	kResCompensatedPose,
	kResTime,
	kResSuccess,
	kResModelRadius,
	kResROI,
	kResPrevCorners,
	kResMask,
	kResDepth,
	kResFineDepth,
	kResOrigin,
	kResOverlay,
	kResProbability,
	kResValid,
	kResInfo,
	kResDataGroup
};

struct TrackingResult
{
	int index;
	cv::Matx33f K;
	cv::Matx44f gt;
	cv::Matx44f pose;
	cv::Matx44f filteredPose;
	cv::Matx44f compensatedPose;
    float time;
	int success;
	float modelRadius;
	float bbxLongestSide;
	cv::Rect roi;
	cv::Mat prevCorners;
	cv::Mat mask;
	cv::Mat depth;
	cv::Mat fineDepth;
	cv::Mat origin;
	cv::Mat overlay;
	cv::Mat probability;
	int valid;
	std::string info;
	DataGroup dataGroup;
};


struct DetectorResult
{
	int index;
	cv::Matx44f pose;
};

struct ProcessResult
{
	TrackingResult* trackingResult;
	DetectorResult* detectorResult;
};

#if __cplusplus > 199711L
#define register      // Deprecated in C++11.
#endif  // #if __cplusplus > 199711L
