/**
 * This file is designed for communication between C++ and Python.
 * The envisioned design includes four communication methods:
 * 1. Disk read/write (implemented in class LocalStorageCommunicator)
 * 2. Shared memory (not implemented)
 * 3. Input/output streams (not implemented)
 * 4. Network communication (not implemented)
 */

#pragma once
#ifndef __COMMUNICATION_H__
#define __COMMUNICATION_H__

#include <iostream>
#include <string>
#include <fstream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <memory>
#include <experimental/filesystem>
#include <algorithm>
#include <any>
#include <map>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "../ar_utils/data_io/data_loader.h"
#include "../definition/global_definition.h"
#include "tracker/summer/src/mbt/prompter.hh"

#include "model_processor.h"

namespace ar3dv
{

	struct SummerConfigs
	{
		std::string refersRoot;
		std::string cmc_root;
		std::string gts_file;
		std::string imgs_dir;
		std::string model_name;
		std::string model_path;
		std::string scaled_model_path;
		int reference_imgs_nums;
		int estimated_imgs_nums;
		int max_model_deformed_nums;
	};

	class Picker
	{
	public:
		Picker() {}

		~Picker() {}

		void GenGeodesicPoints();

		void GenGeodesicPoints(int n);

		void VisualizeGeodesicPoints();

		void VisualizePoints(const std::string &window_name, int image_size = 500);

		void GenTemplateViews(const int &n = 12);

		int PickClosetViews(const Eigen::Vector3f &template_view, const std::vector<cv::Matx44f> &body2cam_poses);

		int PickClosetViews(const Eigen::Vector3f &template_view,
							const std::vector<cv::Matx44f> &body2cam_poses,
							const std::vector<int> &targetIds);

		std::vector<Eigen::Vector3f> template_views() const;

	private:
		std::vector<Eigen::Vector3f> template_views_;
		std::vector<Eigen::Vector3f> geodesic_points_;
		std::vector<std::pair<int, int>> geodesic_edges_;
		std::vector<Eigen::Transform<float, 3, 2>> camera2body_poses_;

		float sphere_radius_{1.0f};
	};

	/*
	 * Communication between C++ and Python through local disk read/write.
	 */
	class LocalStorageCommunicator
	{
	public:
		LocalStorageCommunicator();

		~LocalStorageCommunicator();

		static LocalStorageCommunicator *Instance();

		bool InRefers(const int &index);

		bool ClearCache();

		bool UseRefers(const std::string &root, const int &n = 12);

		bool UseMultiReferenceMoped(const std::string &evaRoot, const std::string &refRoot, const int &n);

		bool SetUp(const SummerConfigs &sc);

		bool SetUp(const std::string &root, const std::string &type = "cmc");

		bool SentDatas2Py(TrackingResult *trackingResult, const int &index);

		bool WritePose(const cv::Matx44f &pose, const int &index);

		template <typename T>
		bool WriteData(const T &data, const int &index, const std::string &info = "")
		{
			std::string errorInfo = "Can not write";
			if constexpr (std::is_same<T, TrackingResult *>::value)
			{
				return true;
			}
			else if constexpr (std::is_same<T, cv::Mat>::value)
			{
				if (info == "img")
					cv::imwrite(img_save_dir_ + ZeroPadding(index, 4) + ".png", data);
				else if (info == "prob")
				{
					cv::imwrite(prob_save_dir_ + ZeroPadding(index, 4) + ".png", data);
				}
				else if (info == "mask")
				{
					cv::imwrite(mask_save_dir_ + ZeroPadding(index, 4) + ".png", data);
				}
				else
					LOG(ERROR) << "No cv::Mat data is written!";
				return true;
			}
			else if constexpr (std::is_same<T, cv::Rect>::value)
			{
				std::ofstream ofs;
				std::string save_file;
				if (info == "origin_roi")
					save_file = origin_roi_save_dir_ + ZeroPadding(index, 4) + ".roi";
				else if (info == "target_roi")
					save_file = target_roi_save_dir_ + ZeroPadding(index, 4) + ".roi";
				else
					LOG(ERROR) << "No roi data is written!";
				ofs.open(save_file, std::ios::out | std::ios::trunc);
				CHECK(ofs.is_open()) << "Error::Roi can not be written!";
				ofs << data.x << " " << data.y << " " << data.width << " " << data.height << "\n";
				ofs.close();
				return true;
			}
			else if constexpr (std::is_same<T, cv::Matx44f>::value)
			{
				std::string save_file = pose_save_dir_ + ZeroPadding(index, 4) + ".pose";
				std::ofstream ofs(save_file, std::ios::out | std::ios::trunc);
				CHECK(ofs.is_open()) << "Error::Pose can not be written!";
				ofs << data(0, 0) << " " << data(0, 1) << " " << data(0, 2) << " "
					<< data(1, 0) << " " << data(1, 1) << " " << data(1, 2) << " "
					<< data(2, 0) << " " << data(2, 1) << " " << data(2, 2) << " "
					<< data(0, 3) << " " << data(1, 3) << " " << data(2, 3) << "\n";
				ofs.close();
				return true;
			}
			else if constexpr (std::is_same<T, cv::Matx33f>::value)
			{
				std::string save_file = K_save_dir_ + ZeroPadding(index, 4) + ".K";
				std::ofstream ofs(save_file, std::ios::out | std::ios::trunc);
				CHECK(ofs.is_open()) << "Error::K can not be written!";
				ofs << data(0, 0) << " " << data(1, 1) << " " << data(0, 2) << " " << data(1, 2) << "\n";
				ofs.close();
				return true;
			}
			else if constexpr (std::is_same<T, cv::Vec2f>::value)
			{
				std::string save_file = target_size_save_dir_ + ZeroPadding(index, 4) + ".size";
				std::ofstream ofs(save_file, std::ios::out | std::ios::trunc);
				CHECK(ofs.is_open()) << "Error::Size can not be written!";
				ofs << data[0] << " " << data[1] << "\n";
				ofs.close();
				return true;
			}
			else
			{
				return false;
			}
		}

		void PickViews(const std::vector<cv::Matx44f> &poses, const int &n = 10);

		bool ScanNewFile(const std::string &directoryPath);

		std::string newest_model_path() const;

		std::string models_dir() const;

		bool set_up(const std::string &type) const;

		template <typename T>
		void SetDatas(const std::string &name, const T &value)
		{
			m_datas[name] = value;
			return;
		};

		bool WriteDataGroup(const DataGroup &dataGroup, const int &index);

		void SetExpectedDataNumsSent2Py(std::vector<int> &datas);
		void DataNumsSent2PyAddOne();
		int data_nums_sent_2_py() const;
		int expected_data_nums_sent_2_py_up_2_current(const int &n) const;

		std::shared_ptr<Picker> picker();

	private:
		static LocalStorageCommunicator *m_instance;

		std::ofstream m_ofs_K;
		std::ofstream m_ofs_pose;
		std::ofstream m_ofs_origin_roi;
		std::ofstream m_ofs_target_roi;
		std::ofstream m_ofs_target_size;

		std::string cmc_root_;
		std::string refers_root_;

		std::string K_save_dir_;
		std::string pose_save_dir_;
		std::string img_save_dir_;
		std::string prob_save_dir_;
		std::string mask_save_dir_;
		std::string origin_roi_save_dir_;
		std::string target_roi_save_dir_;
		std::string target_size_save_dir_;
		std::string models_dir_;

		std::vector<std::string> m_models_path = {};

		std::vector<int> refers_ids_;

		int data_nums_sent_2_py_{0};

		std::vector<int> expected_data_nums_sent_2_py_;

		bool set_up_{false};

		bool cache_cleared_{false};

		std::map<std::string, std::any> m_datas;

		std::shared_ptr<Picker> picker_ = std::make_shared<Picker>();
	};

	class InsideCommunicator
	{
	public:
		InsideCommunicator();

		~InsideCommunicator();

		template <typename T>
		void SetVariable(const std::string &varName, const T &value)
		{
			m_vars[varName] = value;
			return;
		};

		template <typename T>
		T GetValue(const std::string &varName)
		{
			if (m_vars.find(varName) != m_vars.end())
				return std::any_cast<T>(m_vars[varName]);
			else
				return NULL;
		};

		std::shared_ptr<InsideCommunicator> Instance();

	private:
		std::shared_ptr<InsideCommunicator> m_instance;

		std::map<std::string, std::any> m_vars;
	};

	bool ProvideRefers(const std::vector<std::string> &sourceDirs,
					   const std::vector<std::string> &targetDirs,
					   const std::vector<int> &filesIds, const int &maxCpNums = 6);

	class DataPool
	{
	public:
		DataPool(const int &capacity = 20);

		~DataPool();

		bool Clear();

		bool SetCapacity(const int &capacity);

		int current_size() const;

		bool SetMinViewAngleDiffer(const float &angle);

		Eigen::Vector3f CalculateView(const cv::Matx44f &pose) const;

		bool CheckView(const cv::Matx44f &pose) const;

		bool PushData(const DataGroup &dataGroup);

		bool PrintInfo(const std::string &var) const;

		template <typename T>
		T GetValue(const std::string &varName) const
		{
			if (varName == "capacity")
				return static_cast<T>(capacity_);
			else if (varName == "angle")
				return static_cast<T>(min_view_angle_differ_);

			VLOG(0) << "No variable named \'" << varName << "\'";
			return static_cast<T>(nullptr);
		};

	private:
		int capacity_{20};

		float min_view_angle_differ_{10.0f};

		std::vector<DataGroup> datas_;
	};

} // namespace ar3dv;

#endif // __COMMUNICATION_H__