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

using namespace ar3dv;
using namespace cv;
using namespace std;

using LSC = LocalStorageCommunicator;

SummerConfigs LoadSummerConfig(const std::string &yml)
{
	std::string root = LoadSingleInfo<string>(yml, "root");
	std::string cmc_root = root + LoadSingleInfo<string>(yml, "cmcRoot");
	std::string gts_file = LoadSingleInfo<string>(yml, "gt");
	std::string imgs_dir = LoadSingleInfo<string>(yml, "frames");
	string model_name = LoadSingleInfo<string>(yml, "modelName");
	string model_path = LoadSingleInfo<string>(yml, "modelPath");
	string scaled_model_path = LoadSingleInfo<string>(yml, "scaledModelPath");

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

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	std::string rootDir = "../../config/";

	std::string cameraYml, summerYml, summerRoot, modelPath;
	SummerConfigs sc;
	std::vector<cv::Matx44f> gts;
	std::vector<std::string> ips;
	std::vector<cv::String> masks;
	CamParams camParams;
	MeshModel meshModel;
	BoundingBox bbx;

	std::string dataset = argv[1];
	if (dataset == "rbot")
	{
		cameraYml = "../../config/camera/rbot_camera.yml";
		summerYml = rootDir + argv[1] + "/" + argv[2];
		summerRoot = LoadSingleInfo<std::string>(summerYml, "root");
		sc = LoadSummerConfig(summerYml);

		gts = LoadPoses(sc.gts_file, 1, 0, '\t', 1.0f, kMat);
		ips = LoadImages(sc.cmc_root + sc.imgs_dir, "a_regular", 4);

		camParams = LoadCamera(cameraYml, 0);

		modelPath = LoadSingleInfo<std::string>(summerYml, "modelPath");
		MeshModel meshModel_tmp{0, sc.model_path};
		meshModel = meshModel_tmp;
		bbx = ComputeBoundingBox(LoadObjModel(sc.model_path));
	}

	if (dataset == "moped")
	{
		summerYml = rootDir + argv[1] + "/" + argv[2];
		summerRoot = LoadSingleInfo<std::string>(summerYml, "root");
		sc = LoadSummerConfig(summerYml);

		std::string KPath = LoadSingleInfo<std::string>(summerYml, "K");
		camParams = LoadCameraFromJson(KPath, 0);

		std::string originModelPath = LoadSingleInfo<std::string>(summerYml, "modelPath");
		std::string normallizedModel = "model.obj";
		cv::Matx44f b2b = NormalizeMopedModel(originModelPath, normallizedModel);

		MeshModel meshModel_temp{0, normallizedModel};
		meshModel = meshModel_temp;
		bbx = ComputeBoundingBox(LoadObjModel(normallizedModel));

		VLOG(0) << "pose file path " << pjoin(summerRoot, sc.gts_file);
		gts = LoadMopedPoses(pjoin(summerRoot, sc.gts_file));
		ips = LoadImages(pjoin(summerRoot, sc.imgs_dir), "", 6, 0, "jpg");
		masks = LoadImages(pjoin(summerRoot, LoadSingleInfo<std::string>(summerYml, "mask")), "", 6, 0, "png");
		cv::Matx44f b2w = LoadBody2World("toy_plane", "integrated_raw");
		cv::Matx44f w2w = LoadRegistration(summerRoot);
		/**
		 * 	   Reference									    Evaluation
		 *                  cTo                                            cTo
		 *     -------------------------------                  ----------------------------
		 *     |                             |                  |                          |
		 *     |                             *                  |                          *
		 *    obj-------> world <---------- cam                 |           w2 <---------- c
		 *          wTo               wTc                       |      wTw  |
		 *                                                      |           *
		 *  cTo = wTc.inv() * wTo                               obj-------> w1
		 *                                                            wTo
		 *
		 *                                                      cTo = wTc.inv() * wTw.inv() * wTo
		 */

		for (auto &gt : gts)
		{
			{
				gt = gt.inv() * w2w.inv() * b2w * b2b.inv();
			}
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
	SetModels(meshModel);
	VLOG(0) << "Set Models Done";
	SetZnearZfar(0.1f, 2000.0f);
	VLOG(0) << "Set Znear Zfar Done";
	SetErrorThresh(50000000.0f, 500000000.f * poseScale);
	InitSystem();

	std::string resultSaveRoot = MakeTrackingResultDir("../result", trackerType, sc.model_name, "-", g_mkdir);

	StdData stdData;
	StdData *psd = &stdData;
	cv::Mat frame, mask;
	int index = 0;

	LSC *communicator = LSC::Instance();
	if (!communicator->SetUp(summerRoot, "refers"))
		return -1;

	for (auto ip : ips)
	{
		frame = cv::imread(ip);
		if (frame.empty())
		{
			VLOG(0) << "End of frames or frame empty from path: " << ip;
			return -1;
		}

		if (dataset == "moped")
		{
			mask = cv::imread(masks[index]);
			if (mask.empty())
			{
				VLOG(0) << "End of frames or mask empty from path: " << masks[index];
				return -1;
			}
			psd->mask = mask;
		}

		psd->sourceId = 0;
		psd->index = index;
		psd->frame = frame;
		psd->gt = gts[index];
		psd->poseScale = poseScale;

		PushData(psd);
		GenRefers();

		ProcessResult *pRes = GetResult();
		pRes->trackingResult->bbxLongestSide = bbx.longestSide;

		uchar key = cv::waitKey(1);

		if (1 || pRes->trackingResult->dataGroup.valid)
		{
			communicator->SentDatas2Py(pRes->trackingResult, index);
		}

		if (27 == key || 'q' == key || 'Q' == key)
		{
			break;
		}

		VLOG(30) << "Index " << index;
		index++;
	}

	VLOG(0) << ar3dv::Timing::Print();
	VLOG(0) << ar3dv::Statistics::Print();
	return 0;
}