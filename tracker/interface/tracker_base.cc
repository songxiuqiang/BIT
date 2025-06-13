#include "tracker_base.h"
#include "glog/logging.h"

#include "algorithm"

using namespace ar3dv;

TrackerBase::TrackerBase()
{
}

TrackerBase::~TrackerBase()
{
}

void TrackerBase::Init()
{
	VLOG(0) << "TrackerBase Init";
}

void TrackerBase::ChangeModels(const MeshModel &meshModel)
{
	VLOG(0) << "Change Models is not impletemented in current tracker";
	return;
}

void TrackerBase::Estimate()
{
	VLOG(0) << "TrackerBase Estimate";
}

void TrackerBase::DoNotEstimate()
{
	return;
}

void TrackerBase::GenRefers()
{
	VLOG(0) << "TrackerBase GenRefers";
}

cv::Rect TrackerBase::CalculateROI(const cv::Mat &mask, const int &extendPixels, const bool &ReturnEmptyWhenBorder)
{
	cv::Mat gray = mask.clone(), binary, sumRows, sumCols;
	if (mask.channels() > 1)
		cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
	cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY);
	cv::reduce(binary, sumRows, 1, cv::REDUCE_SUM, CV_32S);
	cv::reduce(binary, sumCols, 0, cv::REDUCE_SUM, CV_32S);

	int top = mask.rows, bottom = 0, left = mask.cols, right = 0;
	for (size_t i = 0; i < sumRows.rows; ++i)
	{
		if (sumRows.at<int>(i, 0))
		{
			top = i;
			break;
		}
	}
	for (size_t i = sumRows.rows - 1; i > 0; --i)
	{
		if (sumRows.at<int>(i, 0))
		{
			bottom = i;
			break;
		}
	}

	for (size_t i = 0; i < sumCols.cols; ++i)
	{
		if (sumCols.at<int>(0, i))
		{
			left = i;
			break;
		}
	}
	for (size_t i = sumCols.cols - 1; i > 0; --i)
	{
		if (sumCols.at<int>(0, i))
		{
			right = i;
			break;
		}
	}

	if (top >= bottom || left >= right)
		return cv::Rect(0, 0, 0, 0);

	left = std::max(0, left - extendPixels);
	top = std::max(0, top - extendPixels);
	right = std::min(mask.cols - 1, right + extendPixels);
	bottom = std::min(mask.rows - 1, bottom + extendPixels);

	if (ReturnEmptyWhenBorder && (left == 0 || top == 0 || right == mask.cols - 1 || bottom == mask.rows - 1))
		return cv::Rect(0, 0, 0, 0);
	else
		return cv::Rect(left, top, right - left, bottom - top);
}

cv::Vec2f TrackerBase::CalculateRatioBetweenROI(const cv::Rect &roi1, const cv::Rect &roi2)
{
	if (!roi2.width || !roi2.height)
		return cv::Vec2f(0, 0);
	float w_ratio = float(roi1.width) / float(roi2.width);
	float h_ratio = float(roi1.height) / float(roi2.height);
	return cv::Vec2f(w_ratio, h_ratio);
}

TrackingResult *TrackerBase::GetResult()
{
	SystemBase::Instance()->m_processResult.trackingResult = &m_trackingResult;
	return &m_trackingResult;
}

void TrackerBase::SetData()
{
	m_stdDatas = SystemBase::Instance()->m_stdDatas;
}

void TrackerBase::SetPose(const cv::Matx44f &pose)
{
	return;
}

bool TrackerBase::IsSuccess(const cv::Matx44f &currPose)
{
	if (!m_checkPose)
		return true;
	float rError = GetRoationError(currPose);
	float tError = GetTranslationError(currPose);
	VLOG(30) << "t r error " << tError << " " << rError;
	if (rError >= m_RErrorThresh || tError >= m_tErrorThresh)
		return false;
	else
		return true;
}

bool TrackerBase::IsSuccessByMue(const cv::Matx44f &currPose, const float &diag)
{
	if (!m_checkPose)
		return true;
	float rError = GetRoationError(currPose);

	cv::Matx44f gt = m_stdDatas[0]->gt;
	float transError0 = abs(gt(0, 3) - currPose(0, 3));
	float transError1 = abs(gt(1, 3) - currPose(1, 3));
	float transError2 = abs(gt(2, 3) - currPose(2, 3));

	if (rError >= 15 || transError0 >= diag * 0.05 || transError1 >= diag * 0.05 || transError2 >= diag * 0.15)
		return false;
	else
		return true;
}

float TrackerBase::GetRoationError(const cv::Matx44f &estimate)
{
	cv::Matx44f gt = m_stdDatas[0]->gt;
	cv::Matx33f gtRotation, estimateRotation;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			gtRotation(i, j) = gt(i, j);
			estimateRotation(i, j) = estimate(i, j);
		}
	}
	cv::Matx33f tmp_origin = estimateRotation.t() * gtRotation;
	float trace_origin = tmp_origin(0, 0) + tmp_origin(1, 1) + tmp_origin(2, 2);
	float error_origin = acos((trace_origin - 1) / 2.0f) * 180.0f / 3.14159265f;
	return error_origin;
#ifdef g_regular_rotation_error
	cv::Mat w, u, vt;
	cv::SVD::compute(cv::Mat(gtRotation), w, u, vt);
	cv::Matx33f orthogonalizedGTRotation = cv::Mat(u * vt);

	cv::SVD::compute(cv::Mat(estimateRotation), w, u, vt);
	cv::Matx33f orthogonalizedEstimateRotation = cv::Mat(u * vt);

	cv::Matx33f tmp = orthogonalizedEstimateRotation.t() * orthogonalizedGTRotation;
	float trace = tmp(0, 0) + tmp(1, 1) + tmp(2, 2);
	float error = acos(std::max(-1.0f, std::min(1.0f, (trace - 1) / 2.0f))) * 180.0f / 3.14159265f;
	return error;
#else
	cv::Matx33f tmp = estimateRotation.t() * gtRotation;
	float trace = tmp(0, 0) + tmp(1, 1) + tmp(2, 2);
	return acos((trace - 1) / 2.0f) * 180.0f / 3.14159265f;
#endif
}

float TrackerBase::GetTranslationError(const cv::Matx44f &estimate)
{
	cv::Matx44f gt = m_stdDatas[0]->gt;
	float transError0 = gt(0, 3) - estimate(0, 3);
	float transError1 = gt(1, 3) - estimate(1, 3);
	float transError2 = gt(2, 3) - estimate(2, 3);
	return sqrt(transError0 * transError0 + transError1 * transError1 + transError2 * transError2);
}

void TrackerBase::SetErrorThresh(const float &rErrorThresh, const float &tErrorThresh)
{
	m_RErrorThresh = rErrorThresh;
	m_tErrorThresh = tErrorThresh;
}

void TrackerBase::SetZnearZfar()
{
	m_zn = SystemBase::Instance()->m_zn;
	m_zf = SystemBase::Instance()->m_zf;
}

void TrackerBase::SetInitPose(const cv::Matx44f &initPose)
{
	m_initPose = initPose;
}

int TrackerBase::SetTrackingResult(const int &resType, const cv::Matx33f &K)
{
	switch (resType)
	{
	case kCamParams:
		if (g_save_K)
			m_trackingResult.K = K;
		break;
	default:
		break;
	}
	return resType;
}

int TrackerBase::SetTrackingResult(const int &resType, const cv::Matx44f &pose)
{
	switch (resType)
	{
	case kResGt:
		if (g_save_gt)
			m_trackingResult.gt = pose;
		break;
	case kResPose:
		if (g_save_pose)
			m_trackingResult.pose = pose;
		break;
	case kResFilteredPose:
		if (g_save_filtered_pose)
			m_trackingResult.filteredPose = pose;
		break;
	case kResCompensatedPose:
		if (g_save_compensated_pose)
			m_trackingResult.compensatedPose = pose;
		break;
	default:
		break;
	}
	return resType;
}

int TrackerBase::SetTrackingResult(const int &resType, const float &value)
{
	switch (resType)
	{
	case kResTime:
		if (g_save_time)
			m_trackingResult.time = value;
		break;
	case kResModelRadius:
		if (g_save_model_radius)
			m_trackingResult.modelRadius = value;
		break;
	}
	return resType;
}

int TrackerBase::SetTrackingResult(const int &resType, const cv::Rect &roi)
{
	if (g_save_roi)
		m_trackingResult.roi = roi;
	return resType;
}

int TrackerBase::SetTrackingResult(const int &resType, const int &value)
{
	switch (resType)
	{
	case kResIndex:
		if (g_save_index)
			m_trackingResult.index = value;
		break;
	case kResSuccess:
		if (g_save_success)
			m_trackingResult.success = value;
		break;
	case kResValid:
		if (g_save_valid)
			m_trackingResult.valid = value;
		break;
	default:
		break;
	}
	return resType;
}

int TrackerBase::SetTrackingResult(const int &resType, const cv::Mat &img)
{
	switch (resType)
	{
	case kResPrevCorners:
		if (g_save_prev_corners)
			m_trackingResult.prevCorners = img.clone();
		break;
	case kResMask:
		if (g_save_mask)
			m_trackingResult.mask = img.clone();
		break;
	case kResDepth:
		if (g_save_depth)
			m_trackingResult.depth = img.clone();
		break;
	case kResFineDepth:
		if (g_save_fine_depth)
			m_trackingResult.fineDepth = img.clone();
		break;
	case kResOrigin:
		if (g_save_origin)
			m_trackingResult.origin = img.clone();
		break;
	case kResOverlay:
		if (g_show_overlay || g_save_overlay)
			m_trackingResult.overlay = img.clone();
		break;
	case kResProbability:
		if (g_show_probability || g_save_probability)
			m_trackingResult.probability = img.clone();
	default:
		break;
	}
	return resType;
}

int TrackerBase::SetTrackingResult(const int &resType, const std::string &info)
{
	switch (resType)
	{
	case kResInfo:
		if (g_save_info)
			m_trackingResult.info = info;
		break;
	default:
		break;
	}
	return resType;
}

int TrackerBase::SetTrackingResult(const int &resType, const DataGroup &dataGroup)
{
	switch (resType)
	{
	case kResDataGroup:
		if (g_save_data_group)
			m_trackingResult.dataGroup = dataGroup;
		break;
	default:
		break;
	}
	return resType;
}
