#include <iostream>
#include <chrono>
#include <thread>

#include "opencv2/opencv.hpp"
#include "glog/logging.h"

#include "interface/interface.h"
#include "interface/communication.h"
#include "interface/model_processor.h"

#include "definition/opencv3_definition.h"
#include "ar_utils/data_io/data_loader.h"

#include <filesystem>

using namespace ar3dv;
using namespace cv;
using namespace std;

using LSC = LocalStorageCommunicator;

SummerConfigs LoadSummerConfig(const std::string &yml)
{
	std::string root = LoadSingleInfo<string>(yml, "root");
	std::string cmc_root = root + LoadSingleInfo<string>(yml, "cmc");
	VLOG(0) << "cmcRoot " << cmc_root;
	std::string gts_file = LoadSingleInfo<string>(yml, "gt");
	std::string imgs_dir = LoadSingleInfo<string>(yml, "frames");
	string model_name = LoadSingleInfo<string>(yml, "modelName");
	string model_path = LoadSingleInfo<string>(yml, "modelPath");
	string scaled_model_path = LoadSingleInfo<string>(yml, "scaledModelPath");
	VLOG(0) << "scaled_model_path in loading summer " << scaled_model_path;

	cv::FileStorage configs(yml, cv::FileStorage::READ);
	int reference_imgs_nums = configs["reference_imgs_nums"];
	int estimated_imgs_nums = configs["estimated_imgs_nums"];
	int max_model_deformed_nums = configs["max_model_deformed_nums"];

	SummerConfigs sc = {root, cmc_root, gts_file, imgs_dir, model_name, model_path, scaled_model_path,
						reference_imgs_nums, estimated_imgs_nums, max_model_deformed_nums};
	return sc;
}

bool GetModelFromPy(LSC *communicator, int &cur_model_deformed_nums)
{
	if (communicator->data_nums_sent_2_py() == communicator->expected_data_nums_sent_2_py_up_2_current(cur_model_deformed_nums))
	{
		while (true)
		{
			std::this_thread::sleep_for(std::chrono::seconds(3));
			VLOG(0) << "Waiting to model " << cur_model_deformed_nums << " from python...";
			if (communicator->ScanNewFile(communicator->models_dir()))
			{
				++cur_model_deformed_nums;
				break;
			}
		}
		return true;
	}
	return false;
}

void UpdateModel(const string &updated_model_path, const string &scaled_model_path, const float &bbx_longestSide)
{
	ChangeModelScale(updated_model_path, scaled_model_path, bbx_longestSide);
	MeshModel updated_model{0, scaled_model_path};
	ChangeModels(updated_model);
	return;
}

float calculateAngle(const Eigen::Vector3f &v1, const Eigen::Vector3f &v2)
{
	float dot = v1.dot(v2);
	float magnitude1 = v1.norm();
	float magnitude2 = v2.norm();
	float cosTheta = dot / (magnitude1 * magnitude2);

	if (cosTheta < -1.0f)
		cosTheta = -1.0f;
	if (cosTheta > 1.0f)
		cosTheta = 1.0f;

	float theta = acos(cosTheta);
	float thetaDegrees = theta * 180.0f / M_PI;

	return thetaDegrees;
}

bool DifferEnough(const Eigen::Vector3f &newView, const std::vector<Eigen::Vector3f> &oldViews, const float &thresh)
{
	if (newView[0] == 0 && newView[1] == 0 && newView[2] == 0)
		return false;

	for (auto oldView : oldViews)
	{
		float differ = calculateAngle(newView, oldView);
		if (differ < thresh)
			return false;
	}
	return true;
}

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	std::string summerYml = argv[1];
	std::string configDir = pjoin(std::filesystem::current_path().string(), "config");

	std::vector<cv::Matx44f> gts, poses;
	std::vector<std::string> ips;

	std::string dataset = LoadSingleInfo<std::string>(summerYml, "dataset");

	std::string camConfigYaml, cameraYml, KPath, summerRoot;
	std::string originModelPath, normallizedModel, scaled_model_path;
	SummerConfigs sc;
	CamParams camParams;
	BoundingBox bbx;
	MeshModel meshModel;
	cv::Matx44f b2b, b2w, w2w;

	if (dataset == "rbot")
	{
		camConfigYaml = configDir + "/" + "camera/rbot_camera.yml";
		summerRoot = LoadSingleInfo<std::string>(summerYml, "root");
		sc = LoadSummerConfig(summerYml);
		gts = LoadPoses(sc.gts_file, 1, 0, '\t', 1.0f, kMat);
		ips = LoadImages(summerRoot + sc.imgs_dir, "a_regular", 4);
		camParams = LoadCamera(camConfigYaml, 0);
		scaled_model_path = sc.cmc_root + sc.scaled_model_path;
		bbx = ComputeBoundingBox(LoadObjModel(sc.model_path));
		MeshModel meshModel_temp{0, sc.model_path};
		meshModel = meshModel_temp;
	}

	if (dataset == "moped")
	{
		cameraYml = configDir + "/" + "camera/moped_camera.yml";
		summerRoot = LoadSingleInfo<std::string>(summerYml, "root");
		sc = LoadSummerConfig(summerYml);

		KPath = LoadSingleInfo<std::string>(summerYml, "K");
		camParams = LoadCameraFromJson(KPath, 0);

		originModelPath = LoadSingleInfo<std::string>(summerYml, "modelPath");
		normallizedModel = "model.obj";
		b2b = NormalizeMopedModel(originModelPath, normallizedModel);

		MeshModel meshModel_temp{0, normallizedModel};
		meshModel = meshModel_temp;
		bbx = ComputeBoundingBox(LoadObjModel(normallizedModel));
		scaled_model_path = summerRoot + sc.scaled_model_path;

		gts = LoadMopedPoses(summerRoot + sc.gts_file);
		ips = LoadImages(summerRoot + sc.imgs_dir, "", 6, 0, "jpg");
		b2w = LoadBody2World("toy_plane", "integrated_raw");
		w2w = LoadRegistration(summerRoot);
		for (auto &gt : gts)
			gt = gt.inv() * w2w.inv() * b2w * b2b.inv();
	}

	StdData stdData;
	StdData *psd = &stdData;
	cv::Mat frame;
	int index = 0;

	int cur_model_deformed_nums = 0;
	int intervel = 3;
	std::vector<int> needed_data_nums_sent_2_py;
	needed_data_nums_sent_2_py.resize(15, sc.estimated_imgs_nums);
	needed_data_nums_sent_2_py[0] = 0;

	LocalStorageCommunicator *communicator = LSC::Instance();
	if (!communicator->SetUp(summerRoot, "cmc"))
		return -1;
	communicator->ClearCache();

	int ref_nums = LoadSingleInfo<int>(summerYml, "reference_imgs_nums");
	VLOG(0) << "reference_imgs_nums is : " << ref_nums;
	if (dataset == "moped" && ref_nums > 1)
	{
		std::string refRoot = LoadSingleInfo<std::string>(summerYml, "refRoot");
		communicator->UseMultiReferenceMoped(summerRoot, refRoot, ref_nums - 1);
	}
	else
	{
		communicator->UseRefers(summerRoot, ref_nums - 1);
	}

	communicator->SetExpectedDataNumsSent2Py(needed_data_nums_sent_2_py);
	std::vector<Eigen::Vector3f> templateViews = communicator->picker()->template_views();

	float angleThresh = LoadSingleInfo<float>(summerYml, "angle");

	MeshModel updated_model;
	if (cur_model_deformed_nums < sc.max_model_deformed_nums)
	{
		if (GetModelFromPy(communicator, cur_model_deformed_nums))
		{
			ChangeModelScale(communicator->newest_model_path(), scaled_model_path, bbx.longestSide * 1.05); // 将模型放大到原来的尺寸大小
			updated_model.id = 0;
			updated_model.path = scaled_model_path;
		}
	}

	TrackerType trackerType = TrackerType::kSummer;
	DetectorType detectorType = DetectorType::kLinemod;
	float poseScale = GenPoseScale(trackerType);

	SetTrackerType(trackerType, true);
	VLOG(0) << "Set Tracker Done";
	SetDetectorType(detectorType, false);
	VLOG(0) << "Set Detector Done";
	SetCameras(camParams);
	VLOG(0) << "Set Cameras Done";
	if (dataset == "moped")
		SetZnearZfar(0.1f, 2000.0f);
	else
		SetZnearZfar(1.0f, 2000.0f);

	VLOG(0) << "Set Znear Zfar Done";

	SetModels(updated_model);
	VLOG(0) << "Set Models Done";

	if (dataset == "rbot")
		SetErrorThresh(5.0f, 50.0f * poseScale);
	else
		SetErrorThresh(100000000.0f, 100000000.f * poseScale);

	InitSystem();

	std::string resultSaveRoot = MakeTrackingResultDir("../result", trackerType, sc.model_name, "-", g_mkdir);

	for (auto ip : ips)
	{

		frame = cv::imread(ip);
		CHECK(!frame.empty()) << "frame empty from path: " << ip;

		psd->sourceId = 0;
		psd->index = index;
		psd->frame = frame;
		psd->gt = gts[index];
		psd->poseScale = poseScale;

		PushData(psd);
		ProcessData();

		ProcessResult *pRes = GetResult();
		SaveTrackingResult(resultSaveRoot, pRes->trackingResult);
		// cv::imshow("res", pRes->trackingResult->overlay);

		pRes->trackingResult->bbxLongestSide = bbx.longestSide;
		poses.push_back(pRes->trackingResult->pose);

		Eigen::Vector3f view = pRes->trackingResult->dataGroup.view;
		DifferEnough(view, templateViews, angleThresh); // just for test

		VLOG(0) << "sented " << communicator->data_nums_sent_2_py() << " " << communicator->expected_data_nums_sent_2_py_up_2_current(cur_model_deformed_nums);

		if (cur_model_deformed_nums < sc.max_model_deformed_nums)
		{
			if (GetModelFromPy(communicator, cur_model_deformed_nums))
				UpdateModel(communicator->newest_model_path(), scaled_model_path, bbx.longestSide * 1.05);

			if (!communicator->InRefers(index) &&
				DifferEnough(view, templateViews, angleThresh))
			{
				VLOG(0) << "writed index is " << index;
				communicator->DataNumsSent2PyAddOne();
				communicator->SentDatas2Py(pRes->trackingResult, index);
				templateViews.push_back(view);
			}
		}

		uchar key = cv::waitKey(1);

		if (27 == key)
		{
			break;
		}
		if ('q' == key)
		{
			break;
		}

		VLOG(0) << "Index " << index;
		index++;
	}

	if (dataset == "moped")
	{
		for (auto &gt : gts)
			gt = gt * b2b;
		for (auto &pose : poses)
			pose = pose * b2b;

		VLOG(0) << "GT[0]\n " << gts[0];
		VLOG(0) << "poses[0]\n " << poses[0];
		VLOG(0) << "ADD Error in M: " << ComADD(gts, poses, GetModelPoints(originModelPath));
		VLOG(0) << "ADD AUC Score in M: " << ComAddAUC(gts, poses, GetModelPoints(originModelPath), 0.10f, 0.001f) * 1000;

		for (auto &pose : poses)
		{
			pose(0, 3) *= 1000;
			pose(1, 3) *= 1000;
			pose(2, 3) *= 1000;
		}
		for (auto &gt : gts)
		{
			gt(0, 3) *= 1000;
			gt(1, 3) *= 1000;
			gt(2, 3) *= 1000;
		}
		std::vector<cv::Vec3f> modelPointsInMM = GetModelPoints(originModelPath);
		for (auto &point : modelPointsInMM)
			point *= 1000.0f;

		VLOG(0) << "ADD Error in mm: " << ComADD(gts, poses, modelPointsInMM);
		VLOG(0) << "ADD AUC Score in mm: " << ComAddAUC(gts, poses, modelPointsInMM, 100.0f, 1.0f);
	}

	VLOG(0) << ar3dv::Timing::Print();
	VLOG(0) << ar3dv::Statistics::Print();
	return 0;
}
