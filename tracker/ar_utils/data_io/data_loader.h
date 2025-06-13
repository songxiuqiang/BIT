#pragma once

#include <stdio.h>
#include <dirent.h>
#include <string>
#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <initializer_list>

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include "sophus/se3.hpp"
#include "glog/logging.h"
#include <nlohmann/json.hpp>
#include <assimp/Importer.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "definition/global_definition.h"
#include "../pose_converter/pose_converter.h"

#include <filesystem>
#include <string>
#include <initializer_list>
#include <vector>
#include <sstream>
#include <utility>
namespace ar3dv
{

#define FRONT_2_BACK (int)0
#define BACK_2_FRONT (int)1

	enum TimeFormat
	{
		kTimeFull = 0,	 // 2000-01-01-00-00-00
		kTimeFullNoYear, // 01-01-00-00-00
		kTimeMonDay,	 // 01-01
		kTimeHourMinSec, // 03:08:57
		kTimeYear,
		kTimeMon,
		kTimeDay,
		kTimeHour,
		kTimeMin,
		kTimeSec
	};

	enum PoseType
	{
		kMat = 0,
		kLie = 1,
		kEuler = 2,
		kQuaternion = 3,
		kSo3Tans = 4,
		kMoped = 5
	};

	struct BBX3D
	{
		cv::Vec3f lfu; // left front up corner
		cv::Vec3f lfd; // left front down corner
		cv::Vec3f lbu; // left back up corner
		cv::Vec3f lbd; // left back down corner
		cv::Vec3f rfu; // right front up
		cv::Vec3f rfd; // right front down
		cv::Vec3f rbu; // right back up corner
		cv::Vec3f rbd; // right back down corner
	};

	bool FileExists(const std::string &filename);

	/**
	 * @brief Load camera params
	 * @param configPath camera config's yml file path
	 * @param camID camera id
	 * @return CamParams
	 */
	CamParams LoadCamera(const std::string &configPath, const int &camID = 0);

	CamParams LoadCameraFromJson(const std::string &jsonPath, const int &camID = 0);

	/**
	 * load all images path to vector<string>
	 *
	 * @param path the images file path
	 * @param prefix image filename prefix, use "" when non-prefix
	 * @param digit name digit, use 0 when name number is self growth
	 * @param startNum name start num
	 * @param fileType type of image file
	 * @example if your dataset first image is "a_regular_0000.png",  use LoadImages("your image path", "a_regular_", 4, 0, "png").
	 * @example if first image is "1.jpg",  use LoadImages("your image path", "", 0, 1, "jpg")
	 * @return the full pathname of each image
	 */
	std::vector<std::string> LoadImages(const std::string &path, const std::string &prefix = "", const int &digit = 0, const int &startNum = 0, const std::string &fileType = "png");

	std::vector<std::string> LoadImagesGenmop(const std::string &path);

	std::vector<std::string> LoadImagesNocs(const std::string &path, const std::string &prefix = "", const std::string &suffix = "", const int &digit = 0, const int &startNum = 0, const std::string &fileType = "png");

	void ExpandImage2MultipleOfInterger(cv::Mat &img, const int &ratio = 8);
	void ChangeK2MutipleOfInterger(CamParams &camParams, const int &ratio = 8);

	std::vector<cv::Vec3f> GetModelPoints(const std::string &modelPath);
	BBX3D ComputeBBX3D(const std::vector<cv::Vec3f> &modelPoints);
	float ComADD(const std::vector<cv::Matx44f> &gt, const std::vector<cv::Matx44f> &pose, const std::vector<cv::Vec3f> &modelPoints);
	float ComAddAUC(const std::vector<cv::Matx44f> &gt, const std::vector<cv::Matx44f> &pose, const std::vector<cv::Vec3f> &modelPoints,
					const float &maxThreshholdInMM = 10, const float &stepInMM = 0.1);

	/**
	 * load matrix gt poses as Matx44f from txt
	 *
	 * @param path the pose file path
	 * @param fromWhichRow which line to read from
	 * @param fromWhichColumn which Column to read from
	 * @param split the split between the pose value, including ' ', ',', and '\\t'
	 * @param scale the unit of the translation vector (in mm)
	 * @param poseType type of input pose: kMat, kLie, kEuler, kQuaternion
	 * @attention mat format: r11 r12 r13 r21 r22 r23 r31 r32 r33 tx ty tz
	 * @attention lieAlgebra format: w1 w2 w3 v1 v2 v3
	 * @attention eulerAngle format: x y z tx ty tz
	 * @attention quaternion format: qx qy qz qw tx ty tz
	 * @example if your data is like "14053.12533, 1, qx, qy, qz, qw, tx, ty, tz",translation is in meters and your first row is data, use LoadPoses("your data path", 0, 2, ',' ,1000, ar3dv::PoseType::kQuaternion)
	 * @return the poses int cv::Matx44f format
	 */
	std::vector<cv::Matx44f> LoadPoses(const std::string &path, const int &fromWhichRow = 0, const int &fromWhichColumn = 0,
									   const char &split = ' ', const float &scale = 1.0, const ar3dv::PoseType &poseType = PoseType::kMat);

	std::vector<cv::Matx44f> LoadMopedPoses(const std::string &path);

	cv::Matx44f LoadRegistration(const std::string &dir);

	cv::Matx44f LoadBody2World(const std::string &modelName, const std::string &modelType);

	cv::Matx44f NormalizeMopedModel(const std::string &inputPath, const std::string &outputPath);

	void NormalizeGenmopModel(const std::string &inputPath, const std::string &modelName, const std::string &outputPath);

	template <typename T>
	T LoadSingleInfo(const std::string &configPath, const std::string &keyWord)
	{
		cv::FileStorage configs(configPath, cv::FileStorage::READ);
		T res = configs[keyWord];
		VLOG(0) << "res yml " << res;
		configs.release();
		return res;
	}

	/**
	 * @brief load depth map from txt
	 */
	cv::Mat LoadFloatImg(const std::string &path);

	/**
	 * @brief convert string to specified type
	 */
	template <typename T>
	T CvtStr(const std::string &str)
	{
		T res;
		const char *cstr = str.c_str();
		if (typeid(T) == typeid(int))
		{
			res = std::stoi(cstr);
			VLOG(0) << "INT";
		}
		else if (typeid(T) == typeid(float))
			res = (float)std::stof(cstr);
		else if (typeid(T) == typeid(double))
			res = std::stof(cstr);
		else
			LOG(ERROR) << "No supported type convert of string " << str;
		return res;
	};

	/**
	 * @brief Load txt, then save and return as a vector<vecto<T> >
	 * @tparam T data type
	 * @param path the txt path
	 * @param ignore1stLine not read the first line if this param is true
	 * @param split spliter
	 * @return std::vector<std::vector<T> >
	 */
	template <typename T>
	std::vector<std::vector<T>> Loadtxt(const std::string &path, const int &fromWhichRow, const char &split)
	{
		std::vector<std::vector<T>> res;

		std::ifstream ifs(path);
		CHECK(ifs.is_open()) << "Can not open file: " << path;

		int startLine = fromWhichRow;

		std::string str;
		while (getline(ifs, str))
		{
			if (startLine)
			{
				startLine--;
				continue;
			}
			std::istringstream ss(str);
			std::string s;
			std::vector<T> tmp;
			while (getline(ss, s, split))
			{
				tmp.push_back(CvtStr<T>(s));
			}
			res.push_back(tmp);
		}

		return res;
	}

	bool CopyFile(const std::string &sourceFile, const std::string &destinationFile);

	void DeleteFilesInDir(const std::string &dirPath);

	/**
	 * @brief Delete the specified parts of the string before or after the specified separator, and return the remaining part.
	 * @param s string
	 * @param split separator
	 * @param direction delete direction, from front to back or from back to front, FRONT_2_BACK = 0, BACK_2_FRONT = 1
	 * @param nums the position of the separator
	 * @example string str = "/1/2/3/", split = "/", direction = FRONT_2_BACK, segs = 2, return = "2/3/"
	 */
	std::string DeleteStrParts(const std::string &s, const std::string &split, const int &direction = FRONT_2_BACK, const int &nums = 1);

	float GenPoseScale(const TrackerType &trackerType);

	/**
	 * @brief Get the current time and return
	 * @param format time format, refer to the format in the enum TimeFormat{}
	 * @param split separator
	 * @param format time format
	 * @param split connection symbol
	 */
	std::string GetCurrTime(const int &format = kTimeFullNoYear, const std::string &split = "-");

	/**
	 * @brief Get the current time and return
	 * @param timeKeyWords variable-length parameter, can combine any of the following keywords
	 *	# "Y" "year"
	 *	# "M" "mon" "month"
	 *	# "D" "day"
	 *	# "h" "hour"
	 *	# "m" "min"
	 *	# "s" "sec"
	 * @param split connection symbol
	 */
	std::string GetCurrTime(const std::initializer_list<std::string> &timeKeyWords, const std::string &split = "-");

	/**
	 * @brief 获取指定目录下指定格式的文件的数量并返回 Get the number of files in the specified directory with the specified format
	 * @param root the root directory of the images
	 * @param fileFormat the format of the images, such as ".png"
	 */
	int GetFilesNums(const std::string &root, const std::string &fileFormat = ".png");

	/**
	 * @brief Get the Start Index
	 * @param root
	 * @param prefix prefix of the image file name, such as "a_regular_"
	 * @param digit the number of digits of the image file name, such as 4
	 * @param format img format, such as ".png"
	 * @return int
	 */
	int GetStartIndex(const std::string &root, const std::string &prefix = "", const int digit = 0,
					  const std::string &format = ".png", const int &maxJudgeRange = 10000);

	/**
	 * @brief 获得指定目录下所有文件(夹)的名字并返回 get all the file (folder) names in the specified directory
	 * @param root
	 * @param hiddenDir include the hidden/parent dir or not, deafult is false
	 */
	std::vector<std::string> GetFilesName(const std::string &root, const bool &hiddenDir = false);

	/**
	 * @brief get all the file (folder) paths in the specified directory
	 * @param root
	 * @param hiddenDirinclude the hidden/parent dir or not, deafult is false
	 */
	std::vector<std::string> GetFilesPath(const std::string &root, const bool &hiddenDir = false);

	/**
	 * @brief Get every parts of the string split by the separator
	 */
	std::vector<std::string> GetStrParts(const std::string &s, const char &split = ' ');

	std::string GetTrackerName(const int &tracker);

	/**
	 * @brief make dir and return 1 if success and vice vesa
	 * @param dir
	 * @param make if make == 1, perfome make dir
	 */
	bool MakeDir(const std::string &dir, const bool &make = true);

	/**
	 * @brief mkdir and return the path, the path format is like root/summer/08-04/17-51-52-ape/
	 * @param root save the tracking result to the root directory, all the results of all trackers are saved in this directory
	 * @param tracker the specific directory for the specific tracker, such as SLOT/
	 * @param additionInfo additional info append to the save path
	 * @param split
	 * @param make mkdir or not
	 */
	std::string MakeTrackingResultDir(const std::string &root, const int &tracker, const std::string &additionInfo,
									  const std::string &split = "-", const bool &make = true);

	std::string MakeBCOTTrackingResultDir(const std::string &root, const int &tracker, const std::string &additionInfo,
										  const std::string &split = "-", const bool &make = true);

	/**
	 * @brief Save float image to file
	 */
	bool SaveFloatImg(cv::Mat &img, const std::string &savePath);

	/**
	 * @brief write data and pose to txt file
	 * @param ofs ofstream pointer
	 * @param data data to write, int
	 * @param pose pose to write, cv::Matx44f
	 * @param split connection symbol
	 */
	void SavePose2txt(std::ofstream *ofs, std::initializer_list<int> data, const cv::Matx44f &pose, const std::string &split = "-");

	void SaveFloat2txt(std::ofstream *ofs, std::initializer_list<float> data, const std::string &split = " ");

	void SaveTrackingResult(const std::string &root, TrackingResult *tr, const std::string &split = " ");

	void SplitTxt2MultiTxtFiles(const std::string &inputTxt, const std::string &outputDir, const bool &deleteIndex = false);

	std::string ZeroPadding(const int &idx, const int &number);

} // namespace ar3dv

template <typename T>
std::string to_string_any(const T &value)
{
	std::ostringstream oss;
	oss << value;
	return oss.str();
}

inline std::string to_string_any(const std::string &value)
{
	return value;
}
inline std::string to_string_any(const char *value)
{
	return std::string(value);
}

// convert to string in batch
template <typename... Args>
std::vector<std::string> to_string_vec(Args &&...args)
{
	return {to_string_any(std::forward<Args>(args))...};
}

// path join
template <typename... Args>
std::string pjoin(Args &&...args)
{
	std::filesystem::path p;
	for (const auto &s : to_string_vec(std::forward<Args>(args)...))
	{
		p /= s;
	}
	return p.string();
}
