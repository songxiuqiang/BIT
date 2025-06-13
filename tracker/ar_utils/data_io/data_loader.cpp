#include "data_loader.h"

using namespace std;
using std::string;

namespace ar3dv
{

	bool CopyFile(const std::string &sourceFile, const std::string &destinationFile)
	{
		std::ifstream source(sourceFile, std::ios::binary);
		std::ofstream destination(destinationFile, std::ios::binary);

		if (!source.is_open())
		{
			std::cerr << "Error: Unable to open source file from " << sourceFile << std::endl;
			return false;
		}

		if (!destination.is_open())
		{
			std::cerr << "Error: Unable to open destination file." << std::endl;
			return false;
		}

		destination << source.rdbuf();

		source.close();
		destination.close();

		return true;
	}

	bool FileExists(const std::string &filename)
	{
		std::ifstream file(filename);
		return file.good();
	}

	void DeleteFilesInDir(const std::string &dirPath)
	{
		DIR *dir = opendir(dirPath.c_str());
		struct dirent *entry;

		while ((entry = readdir(dir)) != nullptr)
		{
			std::string filePath = dirPath + "/" + entry->d_name;
			struct stat fileInfo;

			if (lstat(filePath.c_str(), &fileInfo) == -1)
			{
				LOG(ERROR) << "Error getting file information.";
				continue;
			}

			if (S_ISDIR(fileInfo.st_mode))
			{
				// 忽略目录 . 和 ..
				if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
				{
					continue;
				}

				// 递归删除子目录
				DeleteFilesInDir(filePath);
			}
			else
			{
				// 删除文件
				if (remove(filePath.c_str()) != 0)
				{
					LOG(ERROR) << "Error deleting file: " << filePath;
				}
				else
				{
					VLOG(30) << "File deleted: " << filePath;
				}
			}
		}
		closedir(dir);
	}

	CamParams LoadCamera(const std::string &configPath, const int &camID)
	{
		cv::FileStorage configs(configPath, cv::FileStorage::READ);

		CamParams cam;
		cam.sourceId = camID;
		cam.width = configs["width"];
		cam.height = configs["height"];
		cam.fx = configs["fx"];
		cam.fy = configs["fy"];
		cam.cx = configs["cx"];
		cam.cy = configs["cy"];

		VLOG(0) << "cam w h fx fy cx cy: " << cam.width << " " << cam.height << " " << cam.fx << " " << cam.fy << " " << cam.cx << " " << cam.cy;
		configs.release();

		return cam;
	}

	CamParams LoadCameraFromJson(const std::string &jsonPath, const int &camID)
	{
		std::ifstream file(jsonPath);
		if (!file.is_open())
		{
			std::cerr << "Failed to open file: " << jsonPath << std::endl;
		}

		nlohmann::json j;
		file >> j;

		// Confirm required fields are present in the JSON
		if (!j.contains("width") || !j.contains("height") || !j.contains("intrinsic_matrix"))
		{
			std::cerr << "JSON file missing required fields." << std::endl;
			throw std::runtime_error("JSON file missing fields");
		}

		CamParams cam;
		cam.sourceId = 0;
		cam.width = j["width"];
		cam.height = j["height"];

		// Ensure intrinsic_matrix is valid and contains 9 elements (3x3 matrix)
		if (!j["intrinsic_matrix"].is_array() || j["intrinsic_matrix"].size() != 9)
		{
			std::cerr << "Invalid intrinsic_matrix format." << std::endl;
			throw std::runtime_error("Invalid intrinsic_matrix");
		}

		// Extract fx, fy, cx, and cy from the intrinsic_matrix
		auto intrinsic_matrix = j["intrinsic_matrix"];
		VLOG(0) << intrinsic_matrix;
		cam.fx = intrinsic_matrix[0];
		cam.fy = intrinsic_matrix[4];
		cam.cx = intrinsic_matrix[6];
		cam.cy = intrinsic_matrix[7];

		return cam;
	}

	std::vector<std::string> LoadImages(const std::string &path, const std::string &prefix, const int &digit, const int &startNum, const std::string &fileType)
	{
		std::vector<std::string> imageFiles;

		if (digit != 0)
		{
			std::vector<std::string> allImageFiles;
			std::string pattern = path + prefix + "*." + fileType;
			cv::glob(pattern, allImageFiles);
			CHECK(!(allImageFiles.empty())) << "image path empty error: " << path;

			// 过滤隐藏文件
			for (const auto &file : allImageFiles)
			{
				std::string filename = file.substr(file.find_last_of("/\\") + 1);
				if (filename[0] != '.')
				{
					imageFiles.push_back(file);
				}
			}
			CHECK(!(imageFiles.empty())) << "image path empty error: " << path;
			VLOG(0) << "imageFiles[0] " << imageFiles[0];
		}
		else
		{
			std::vector<std::string> temp;
			std::string name;
			std::string pattern = path + prefix + "*." + fileType;
			cv::glob(pattern, temp);
			auto count = temp.size();
			CHECK(!(temp.empty())) << "image path empty error: " << path;
			auto countNum = startNum;
			do
			{
				name = path + prefix + std::to_string(countNum) + "." + fileType;
				imageFiles.push_back(name);
				countNum++;
			} while (count--);
		}
		return imageFiles;
	}

	std::vector<std::string> LoadImagesGenmop(const std::string &path)
	{
		std::vector<std::string> imageFiles;

		std::vector<std::string> temp;
		std::string name;
		std::string pattern = path + "frame*.jpg";
		cv::glob(pattern, temp);
		auto count = temp.size();
		int countNum = 0;
		CHECK(!(temp.empty())) << "image path empty error: " << path;
		do
		{
			name = path + "frame" + std::to_string(countNum) + ".jpg";
			imageFiles.push_back(name);
			countNum += 10;
		} while (count--);
		return imageFiles;
	}

	void ExpandImage2MultipleOfInterger(cv::Mat &img, const int &ratio)
	{
		// 计算新长宽, 使其都成为 ratio 的倍数
		int newWidth = (img.cols % ratio == 0) ? img.cols : (img.cols / ratio + 1) * ratio;
		int newHeight = (img.rows % ratio == 0) ? img.rows : (img.rows / ratio + 1) * ratio;

		// 只在需要时才进行填充
		// 只在需要时才进行填充
		if (newWidth != img.cols || newHeight != img.rows)
		{
			// 左, 上, 右, 下 边界扩展大小
			int top = 0;
			int bottom = newHeight - img.rows;
			int left = 0;
			int right = newWidth - img.cols;

			// 增加边界到图像
			cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
		}
	}

	void ChangeK2MutipleOfInterger(CamParams &camParams, const int &ratio)
	{
		// 原始长宽
		int w = camParams.width;
		int h = camParams.height;

		int xOffset = 8 - w % 8;
		int yOffset = 8 - h % 8;
		if (xOffset == 8)
			xOffset = 0;
		if (yOffset == 8)
			yOffset = 0;
		if (xOffset == 0 && yOffset == 0)
			return;

		// 扩展后的长宽
		int w2 = w + xOffset;
		int h2 = h + yOffset;
		int fx2 = camParams.fx * (float)w2 / float(w);
		int fy2 = camParams.fy * (float)h2 / float(h);
		return;
	}

	cv::Matx44f LoadBody2World(const std::string &modelName, const std::string &modelType)
	{
		cv::Matx44f b2w = cv::Matx44f::eye();
		cv::Vec3f translation(0, 0, 0);

		cv::Vec3f plane(0.0836, -0.0489, 0.4899);

		if (modelName == "toy_plane")
			translation = plane;
		else
			LOG(ERROR) << "LOGERROR: " << __func__ << ": 未知的模型名称 \"" << modelName << "\"";

		if (modelType == "integrated_raw")
		{
			return b2w;
		}
		else if (modelType == "integrated_registered_processed")
		{
			b2w(0, 3) = translation[0];
			b2w(1, 3) = translation[1];
			b2w(2, 3) = translation[2];
		}
		else
		{
			LOG(ERROR) << "LOGERROR: " << __func__ << ": 未知的模型种类 \"" << modelType << "\"";
		}

		return b2w;
	}

	// 从 JSON 文件加载矩阵
	cv::Matx44f LoadMatFromJson(const std::string &filepath)
	{
		std::ifstream file(filepath);
		if (!file.is_open())
		{
			std::cerr << "Failed to open file: " << filepath << std::endl;
			return cv::Matx44f::eye(); // 返回单位矩阵
		}

		nlohmann::json j;
		file >> j;

		cv::Matx44f transform;
		auto tf = j["transform"];

		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				transform(i, j) = tf[i][j];
			}
		}

		return transform;
	}

	cv::Matx44f LoadRegistration(const std::string &dir)
	{
		VLOG(0) << "Man!";
		cv::Matx44f w2w = cv::Matx44f::eye(); // 默认单位矩阵
		VLOG(0) << "Manba!";
		std::string manual = dir + "registration/manual.json";
		std::string registration = dir + "registration/registration.json";
		if (FileExists(manual))
		{
			VLOG(0) << "Muaual exist";
			w2w = LoadMatFromJson(manual);
		}
		else if (FileExists(registration))
		{
			VLOG(0) << "registration exist";
			w2w = LoadMatFromJson(registration);
		}
		else
		{
			VLOG(0) << "Warning! Find no registration file! from " << manual << " or " << registration;
		}

		return w2w;
	}

	// BoundingBox 类定义
	class BoundingBox
	{
	public:
		cv::Point3f min;
		cv::Point3f max;

		BoundingBox()
			: min(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()),
			  max(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()) {}

		void update(const cv::Point3f &point)
		{
			if (point.x < min.x)
				min.x = point.x;
			if (point.y < min.y)
				min.y = point.y;
			if (point.z < min.z)
				min.z = point.z;
			if (point.x > max.x)
				max.x = point.x;
			if (point.y > max.y)
				max.y = point.y;
			if (point.z > max.z)
				max.z = point.z;
		}

		cv::Point3f center() const
		{
			return cv::Point3f(
				(min.x + max.x) / 2.0f,
				(min.y + max.y) / 2.0f,
				(min.z + max.z) / 2.0f);
		}

		float maxDimension() const
		{
			return std::max({max.x - min.x, max.y - min.y, max.z - min.z});
		}
	};

	// 计算包围盒
	BoundingBox ComputeBoundingBox(aiScene *scene)
	{
		BoundingBox bbox;

		for (unsigned int i = 0; i < scene->mNumMeshes; ++i)
		{
			const aiMesh *mesh = scene->mMeshes[i];
			for (unsigned int j = 0; j < mesh->mNumVertices; ++j)
			{
				aiVector3D tmp = mesh->mVertices[j];
				bbox.update(cv::Point3f(tmp.x, tmp.y, tmp.z));
			}
		}

		return bbox;
	}

	// 读取模型并归一化
	cv::Matx44f NormalizeMopedModel(const std::string &inputPath, const std::string &outputPath)
	{
		Assimp::Importer importer;
		aiScene *scene = (aiScene *)importer.ReadFile(inputPath, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);

		if (!scene)
		{
			std::cerr << "Error: " << importer.GetErrorString() << std::endl;
			return cv::Matx44f::eye();
		}

		// 计算包围盒
		BoundingBox bbox = ComputeBoundingBox(scene);

		// 计算包围盒中心和最大尺寸
		cv::Point3f center = bbox.center();
		float maxDim = bbox.maxDimension();

		// 创建变换矩阵
		cv::Matx44f transform;
		// 平移部分
		transform = cv::Matx44f(1, 0, 0, -center.x,
								0, 1, 0, -center.y,
								0, 0, 1, -center.z,
								0, 0, 0, 1);

		// 缩放部分
		cv::Matx44f scale = cv::Matx44f(1.0f / maxDim, 0, 0, 0,
										0, 1.0f / maxDim, 0, 0,
										0, 0, 1.0f / maxDim, 0,
										0, 0, 0, 1);

		transform = scale * transform;

		// 应用变换到模型的顶点
		for (unsigned int i = 0; i < scene->mNumMeshes; ++i)
		{
			aiMesh *mesh = scene->mMeshes[i];
			for (unsigned int j = 0; j < mesh->mNumVertices; ++j)
			{
				aiVector3D &v = mesh->mVertices[j];

				// 减去中心并缩放
				v.x = (v.x - center.x) / maxDim;
				v.y = (v.y - center.y) / maxDim;
				v.z = (v.z - center.z) / maxDim;
			}
		}

		// 保存调整后的模型
		Assimp::Exporter exporter;
		if (exporter.Export(scene, "obj", outputPath) != AI_SUCCESS)
		{
			std::cerr << "Failed to export the adjusted model: " << exporter.GetErrorString() << std::endl;
		}

		return transform;
	}

	Eigen::Matrix3f compute_rotation(const Eigen::Vector3f &vert, const Eigen::Vector3f &forward)
	{
		Eigen::Vector3f y = vert.cross(forward);
		Eigen::Vector3f x = y.cross(vert);
		Eigen::Vector3f normalized_vert = vert.normalized();
		Eigen::Vector3f normalized_x = x.normalized();
		Eigen::Vector3f normalized_y = y.normalized();
		Eigen::Matrix3f R;
		R.row(0) = normalized_x;
		R.row(1) = normalized_y;
		R.row(2) = normalized_vert;
		return R;
	}

	float compute_normalized_ratio(const aiScene *scene)
	{
		Eigen::Vector3f min_pt(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
		Eigen::Vector3f max_pt(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

		for (unsigned int i = 0; i < scene->mNumMeshes; ++i)
		{
			const aiMesh *mesh = scene->mMeshes[i];
			for (unsigned int j = 0; j < mesh->mNumVertices; ++j)
			{
				const aiVector3D &vertex = mesh->mVertices[j];
				min_pt = min_pt.array().min(Eigen::Vector3f(vertex.x, vertex.y, vertex.z).array());
				max_pt = max_pt.array().max(Eigen::Vector3f(vertex.x, vertex.y, vertex.z).array());
			}
		}

		float dist = (max_pt - min_pt).norm();
		float scale_ratio = 2.0f / dist;
		return scale_ratio;
	}

	void NormalizeGenmopModel(const std::string &inputPath, const std::string &modelName, const std::string &outputPath)
	{
		// Meta information map
		std::map<std::string, std::map<std::string, Eigen::Vector3f>> genmop_meta_info = {
			{"cup", {{"gravity", Eigen::Vector3f(-0.0893124, -0.399691, -0.912288)}, {"forward", Eigen::Vector3f(-0.009871, 0.693020, -0.308549)}}},
			{"tformer", {{"gravity", Eigen::Vector3f(-0.0734401, -0.633415, -0.77032)}, {"forward", Eigen::Vector3f(-0.121561, -0.249061, 0.211048)}}},
			{"chair", {{"gravity", Eigen::Vector3f(0.111445, -0.373825, -0.920779)}, {"forward", Eigen::Vector3f(0.788313, -0.139603, 0.156288)}}},
			{"knife", {{"gravity", Eigen::Vector3f(-0.0768299, -0.257446, -0.963234)}, {"forward", Eigen::Vector3f(0.954157, 0.401808, -0.285027)}}},
			{"love", {{"gravity", Eigen::Vector3f(0.131457, -0.328559, -0.93529)}, {"forward", Eigen::Vector3f(-0.045739, -1.437427, 0.497225)}}},
			{"plug_cn", {{"gravity", Eigen::Vector3f(-0.0267497, -0.406514, -0.913253)}, {"forward", Eigen::Vector3f(-0.172773, -0.441210, 0.216283)}}},
			{"plug_en", {{"gravity", Eigen::Vector3f(0.0668682, -0.296538, -0.952677)}, {"forward", Eigen::Vector3f(0.229183, -0.923874, 0.296636)}}},
			{"miffy", {{"gravity", Eigen::Vector3f(-0.153506, -0.35346, -0.922769)}, {"forward", Eigen::Vector3f(-0.584448, -1.111544, 0.490026)}}},
			{"scissors", {{"gravity", Eigen::Vector3f(-0.129767, -0.433414, -0.891803)}, {"forward", Eigen::Vector3f(1.899760, 0.418542, -0.473156)}}},
			{"piggy", {{"gravity", Eigen::Vector3f(-0.122392, -0.344009, -0.930955)}, {"forward", Eigen::Vector3f(0.079012, 1.441836, -0.524981)}}}};

		Eigen::Vector3f gravity = genmop_meta_info[modelName]["gravity"];
		Eigen::Vector3f forward = genmop_meta_info[modelName]["forward"];

		Assimp::Importer importer;
		aiScene *scene = (aiScene *)importer.ReadFile(inputPath, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);

		if (!scene)
		{
			std::cerr << "Error: " << importer.GetErrorString() << std::endl;
		}

		// 计算旋转矩阵
		Eigen::Matrix3f rotation = compute_rotation(gravity, forward);

		// 应用旋转变换到模型的顶点
		for (unsigned int i = 0; i < scene->mNumMeshes; ++i)
		{
			aiMesh *mesh = scene->mMeshes[i];
			for (unsigned int j = 0; j < mesh->mNumVertices; ++j)
			{
				aiVector3D &v = mesh->mVertices[j];
				Eigen::Vector3f vertex(v.x, v.y, v.z);
				vertex = rotation * vertex;

				// Assign the rotated vertex back
				v.x = vertex.x();
				v.y = vertex.y();
				v.z = vertex.z();
			}
		}

		// 计算归一化比例
		float scale_ratio = compute_normalized_ratio(scene);

		// 将归一化比例应用到模型的顶点
		for (unsigned int i = 0; i < scene->mNumMeshes; ++i)
		{
			aiMesh *mesh = scene->mMeshes[i];
			for (unsigned int j = 0; j < mesh->mNumVertices; ++j)
			{
				aiVector3D &v = mesh->mVertices[j];
				v.x *= scale_ratio;
				v.y *= scale_ratio;
				v.z *= scale_ratio;
			}
		}

		BoundingBox bbox = ComputeBoundingBox(scene);
		VLOG(0) << "bbox min max " << bbox.min << bbox.max;
		// 保存经过旋转和归一化后的模型
		Assimp::Exporter exporter;
		if (exporter.Export(scene, "obj", outputPath) != AI_SUCCESS)
		{
			std::cerr << "Failed to export model: " << exporter.GetErrorString() << std::endl;
		}

		return;
	}

	std::vector<cv::Vec3f> GetModelPoints(const std::string &modelPath)
	{
		// 使用 Assimp 加载 3D 模型
		Assimp::Importer importer;
		const aiScene *scene = importer.ReadFile(modelPath, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);

		// 提取模型的顶点数据
		std::vector<cv::Vec3f> model_points;
		if (scene && scene->mMeshes && scene->mMeshes[0])
		{
			aiMesh *mesh = scene->mMeshes[0];
			for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
			{
				cv::Vec3f point;
				point[0] = mesh->mVertices[i].x;
				point[1] = mesh->mVertices[i].y;
				point[2] = mesh->mVertices[i].z;
				model_points.push_back(point);
			}
		}
		VLOG(0) << "Get Model Points Down";
		CHECK(model_points.size() != 0) << "Null Model Points";
		return model_points;
	}

	BBX3D ComputeBBX3D(const std::vector<cv::Vec3f> &modelPoints)
	{
		// Initialize the bounding box corners with the first point
		cv::Vec3f minCorner = modelPoints[0];
		cv::Vec3f maxCorner = modelPoints[0];

		// Find the minimum and maximum coordinates along each axis
		for (const auto &point : modelPoints)
		{
			for (int i = 0; i < 3; ++i)
			{
				minCorner[i] = std::min(minCorner[i], point[i]);
				maxCorner[i] = std::max(maxCorner[i], point[i]);
			}
		}

		// Create a BBX3D object and assign the calculated corners
		BBX3D boundingBox;
		boundingBox.lfu = cv::Vec3f(minCorner[0], maxCorner[1], maxCorner[2]);
		boundingBox.lfd = cv::Vec3f(minCorner[0], minCorner[1], maxCorner[2]);
		boundingBox.lbu = cv::Vec3f(minCorner[0], maxCorner[1], minCorner[2]);
		boundingBox.lbd = cv::Vec3f(minCorner[0], minCorner[1], minCorner[2]);
		boundingBox.rfu = cv::Vec3f(maxCorner[0], maxCorner[1], maxCorner[2]);
		boundingBox.rfd = cv::Vec3f(maxCorner[0], minCorner[1], maxCorner[2]);
		boundingBox.rbu = cv::Vec3f(maxCorner[0], maxCorner[1], minCorner[2]);
		boundingBox.rbd = cv::Vec3f(maxCorner[0], minCorner[1], minCorner[2]);

		return boundingBox;
	}

	float ComADD(const std::vector<cv::Matx44f> &gt, const std::vector<cv::Matx44f> &pose, const std::vector<cv::Vec3f> &modelPoints)
	{
		// 检查输入的大小是否匹配
		if (gt.size() != pose.size())
		{
			throw std::invalid_argument("Ground truth and pose vector sizes do not match.");
		}

		// 初始化累计误差
		double totalError = 0.0;
		int numPoints = modelPoints.size();
		int numPoses = gt.size();

		// 逐个姿态计算误差
		for (int i = 0; i < numPoses; ++i)
		{
			cv::Matx44f gtTransform = gt[i];
			cv::Matx44f poseTransform = pose[i];

			for (const auto &modelPoint : modelPoints)
			{
				cv::Matx41f homogenousModelPoint(modelPoint[0], modelPoint[1], modelPoint[2], 1.0f);

				cv::Matx41f gtTransformed = gtTransform * homogenousModelPoint;
				cv::Vec3f gtPoint(gtTransformed(0), gtTransformed(1), gtTransformed(2));

				cv::Matx41f poseTransformed = poseTransform * homogenousModelPoint;
				cv::Vec3f posePoint(poseTransformed(0), poseTransformed(1), poseTransformed(2));

				// Calculate Euclidean distance between transformed points
				double distance = cv::norm(gtPoint - posePoint);
				totalError += distance;
			}
		}

		// 计算平均误差: 将总误差除以点数和姿态数
		float averageError = static_cast<float>(totalError / (numPoints * numPoses));
		return averageError;
	}

	// 计算单个姿态的ADD误差
	float ComputeADDSinglePose(const cv::Matx44f &gt, const cv::Matx44f &pose, const std::vector<cv::Vec3f> &modelPoints)
	{
		double totalError = 0.0;
		int numPoints = modelPoints.size();

		for (const auto &modelPoint : modelPoints)
		{
			cv::Matx41f homogenousModelPoint(modelPoint[0], modelPoint[1], modelPoint[2], 1.0f);
			cv::Matx41f gtTransformed = gt * homogenousModelPoint;
			cv::Vec3f gtPoint(gtTransformed(0), gtTransformed(1), gtTransformed(2));

			cv::Matx41f poseTransformed = pose * homogenousModelPoint;
			cv::Vec3f posePoint(poseTransformed(0), poseTransformed(1), poseTransformed(2));

			double distance = cv::norm(gtPoint - posePoint);
			totalError += distance;
		}

		return static_cast<float>(totalError / numPoints);
	}

	float ComAddAUC(const std::vector<cv::Matx44f> &gt, const std::vector<cv::Matx44f> &pose, const std::vector<cv::Vec3f> &modelPoints,
					const float &maxThreshholdInMM, const float &stepInMM)
	{

		std::vector<float> errors;
		for (size_t i = 0; i < gt.size(); ++i)
		{
			float error = ComputeADDSinglePose(gt[i], pose[i], modelPoints);
			errors.push_back(error);
		}

		std::sort(errors.begin(), errors.end());

		std::vector<float> cdf;
		int numSteps = static_cast<int>(maxThreshholdInMM / stepInMM);
		for (int i = 0; i <= numSteps; ++i)
		{
			float threshold = i * stepInMM;
			auto it = std::upper_bound(errors.begin(), errors.end(), threshold);
			cdf.push_back(static_cast<float>(std::distance(errors.begin(), it)) / errors.size());
		}

		float auc = 0.0f;
		for (int i = 0; i < numSteps; ++i)
		{
			auc += (cdf[i] + cdf[i + 1]) * stepInMM * 0.5f;
		}

		return auc;
	}

	std::vector<std::string> LoadImagesNocs(const std::string &path, const std::string &prefix, const std::string &suffix,
											const int &digit, const int &startNum, const std::string &fileType)
	{
		std::vector<std::string> imageFiles;

		if (digit != 0)
		{
			std::string pattern = path + prefix + "*." + fileType;
			cv::glob(pattern, imageFiles);
			CHECK(!(imageFiles.empty())) << "image path empty error: " << path;
		}
		else
		{
			std::vector<std::string> temp;
			std::string name;
			std::string pattern = path + prefix + "*." + fileType;
			cv::glob(pattern, temp);
			auto count = temp.size();
			CHECK(!(temp.empty())) << "image path empty error: " << path;
			auto countNum = startNum;
			do
			{
				name = path + prefix + std::to_string(countNum) + suffix + fileType;
				imageFiles.push_back(name);
				countNum++;
			} while (count--);
		}
		return imageFiles;
	}

	cv::Matx44f LoadMatFromRt(const std::vector<float> &input)
	{
		CHECK(!(input.size() < 12)) << "Pose input error (type Rt): check the size and split and so on";
		return cv::Matx44f{
			input[0], input[1], input[2], input[9],
			input[3], input[4], input[5], input[10],
			input[6], input[7], input[8], input[11],
			0, 0, 0, 1};
	}

	cv::Matx44f LoadMatFromSe3(const std::vector<float> &input)
	{
		CHECK(!(input.size() < 6)) << "Pose input error (type se3): check the size and split and so on";
		Eigen::VectorXf se3(6);
		// VLOG(0) << input[3] << " " << input[4] << " " << input[5] << " " << input[0] << " " << input[1] << " " <<  input[2];
		se3 << input[3], input[4], input[5], input[0], input[1], input[2];
		Sophus::SE3<float> SE3 = Sophus::SE3<float>::exp(se3);
		auto pose = SE3.matrix();
		return cv::Matx44f{
			pose(0, 0), pose(0, 1), pose(0, 2), pose(0, 3),
			pose(1, 0), pose(1, 1), pose(1, 2), pose(1, 3),
			pose(2, 0), pose(2, 1), pose(2, 2), pose(2, 3),
			pose(3, 0), pose(3, 1), pose(3, 2), pose(3, 3)};
	}

	cv::Matx44f LoadMatFromEulerAngle(const std::vector<float> &input)
	{
		CHECK(!(input.size() < 6)) << "Pose input error (type EulerAngle): check the size and split and so on";
		Eigen::Matrix3f R;
		R = Eigen::AngleAxisf(input[0], Eigen::Vector3f::UnitX()) *
			Eigen::AngleAxisf(input[1], Eigen::Vector3f::UnitY()) *
			Eigen::AngleAxisf(input[2], Eigen::Vector3f::UnitZ());
		return cv::Matx44f{
			R(0, 0), R(0, 1), R(0, 2), input[3],
			R(1, 0), R(1, 1), R(1, 2), input[4],
			R(2, 0), R(2, 1), R(2, 2), input[5],
			0, 0, 0, 1};
	}

	cv::Matx44f LoadMatFromQuaternion(const std::vector<float> &input)
	{
		CHECK(!(input.size() < 7)) << "Pose input error: quaternion (type Quaternion): check the size and split and so on";
		Eigen::Quaternionf q(Eigen::Vector4f(input[0], input[1], input[2], input[3]));
		q = q.normalized();
		Eigen::Matrix3f R = q.matrix();
		return cv::Matx44f{
			R(0, 0), R(0, 1), R(0, 2), input[4],
			R(1, 0), R(1, 1), R(1, 2), input[5],
			R(2, 0), R(2, 1), R(2, 2), input[6],
			0, 0, 0, 1};
	}

	cv::Matx44f LoadSo3Trans(const std::vector<float> &input)
	{
		CHECK(!(input.size() < 6)) << "Pose input error: ( type So3Trans): check the size and split and so on";
		cv::Vec3f so3(input[0], input[1], input[2]);
		cv::Matx33f R = ar3dv::So32R(so3);
		return cv::Matx44f{
			R(0, 0), R(0, 1), R(0, 2), input[3],
			R(1, 0), R(1, 1), R(1, 2), input[4],
			R(2, 0), R(2, 1), R(2, 2), input[5],
			0, 0, 0, 1};
	}

	std::vector<cv::Matx44f> LoadPoses(const std::string &path, const int &fromWhichRow, const int &fromWhichColumn,
									   const char &split, const float &scale, const ar3dv::PoseType &poseType)
	{
		std::ifstream poseFile(path);
		CHECK(!poseFile.fail()) << "Can not open pose file: " << path;
		std::vector<cv::Matx44f> poses;
		std::string readline, temp;
		int rowCount = fromWhichRow;

		while (getline(poseFile, readline) && readline.length() != 0)
		{
			// step over notes
			if (rowCount > 0)
			{
				rowCount--;
				continue;
			}

			std::vector<float> input;
			std::istringstream record(readline);
			cv::Matx44f pose;
			int columnCount = fromWhichColumn;

			// step over columns
			while (getline(record, temp, split))
			{
				if (columnCount > 0)
				{
					columnCount--;
					continue;
				}
				if (temp == "\r")
					continue; // 兼容性
				input.push_back(stof(temp));
			}

			switch (poseType)
			{
			case ar3dv::PoseType::kMat:
				pose = LoadMatFromRt(input);
				break;
			case ar3dv::PoseType::kLie:
				pose = LoadMatFromSe3(input);
				break;
			case ar3dv::PoseType::kEuler:
				pose = LoadMatFromEulerAngle(input);
				break;
			case ar3dv::PoseType::kQuaternion:
				pose = LoadMatFromQuaternion(input);
				break;
			case ar3dv::PoseType::kSo3Tans:
				pose = LoadSo3Trans(input);
				break;
			default:
				LOG(ERROR) << "poseType error";
				break;
			}

			pose(0, 3) *= scale;
			pose(1, 3) *= scale;
			pose(2, 3) *= scale;

			poses.push_back(pose);
		}
		poseFile.close();
		// VLOG(0) << "Load Poses Done";
		return poses;
	}

	std::vector<cv::Matx44f> LoadMopedPoses(const std::string &path)
	{
		std::vector<cv::Matx44f> poses;
		std::ifstream file(path);
		if (!file.is_open())
		{
			std::cerr << "Failed to open file: " << path << std::endl;
			return poses;
		}

		std::string line;
		while (std::getline(file, line))
		{
			// Skip the line with image indices
			if (isdigit(line[0]))
			{
				// Read the next 4 lines for the pose matrix
				cv::Matx44f mat;
				for (int i = 0; i < 4; ++i)
				{
					std::getline(file, line);
					std::istringstream iss(line);
					iss >> mat(i, 0) >> mat(i, 1) >> mat(i, 2) >> mat(i, 3);
				}
				poses.push_back(mat);
			}
		}

		file.close();
		return poses;
	}

	cv::Mat LoadFloatImg(const std::string &path)
	{
		std::ifstream ifs(path);

		if (!ifs.is_open())
		{
			VLOG(0) << "Can not open " << path;
		}

		int rows, cols;
		ifs >> rows >> cols;
		cv::Mat img(rows, cols, CV_32FC1);

		for (int i = 0; i < img.rows; i++)
		{
			float *p = img.ptr<float>(i);
			for (int j = 0; j < img.cols; j++)
			{
				ifs >> p[j];
			}
		}
		ifs.close();

		return img;
	}

	std::string DeleteStrParts(const std::string &s, const std::string &split, const int &direction, const int &nums)
	{
		std::string res = s;

		if (direction == BACK_2_FRONT)
		{
			for (size_t i = 0; i < nums; ++i)
			{
				size_t pos = res.find_last_of(split);
				std::string retain = s.substr(0, pos);
				res = retain;
			}
		}
		else if (direction == FRONT_2_BACK)
		{
			for (size_t i = 0; i < nums; ++i)
			{
				size_t pos = res.find_first_of(split);
				res = res.substr(pos + 1, s.size());
			}
		}
		return res;
	}

	float GenPoseScale(const TrackerType &trackerType)
	{
		float poseScale = 1.0f;
		if (trackerType == TrackerType::kRBGT ||
			trackerType == TrackerType::kSRT3D ||
			trackerType == TrackerType::kICG ||
			trackerType == TrackerType::kRMT)
			poseScale = 0.001f;

		VLOG(0) << "Pose scale: " << poseScale << "\n Tracker: " << GetTrackerName(trackerType);
		return poseScale;
	}

	int GetTimeEnumType(const std::string &word)
	{
		if (word == "year" || word == "Y")
			return kTimeYear;
		if (word == "mon" || word == "month" || word == "M")
			return kTimeMon;
		if (word == "day" || word == "D")
			return kTimeDay;
		if (word == "hour" || word == "h")
			return kTimeHour;
		if (word == "min" || word == "m")
			return kTimeMin;
		if (word == "sec" || word == "s")
			return kTimeSec;
		LOG(WARNING) << "Warning::Unkown date or time type. You may get error result on GetCurrTime()";
		return -1;
	}

	std::string GetCurrTime(const int &format, const std::string &split)
	{
		std::string res;

		time_t t;
		time(&t);
		struct tm *timeinfo;
		timeinfo = localtime(&t);

		string year = std::to_string(timeinfo->tm_year + 1900);
		string mon = ZeroPadding(timeinfo->tm_mon + 1, 2);
		string day = ZeroPadding(timeinfo->tm_mday, 2);
		string hour = ZeroPadding(timeinfo->tm_hour, 2);
		string min = ZeroPadding(timeinfo->tm_min, 2);
		string sec = ZeroPadding(timeinfo->tm_sec, 2);

		switch (format)
		{
		case kTimeFull:
			res = year + split + mon + split + day + split + hour + split + min + split + sec;
			break;
		case kTimeFullNoYear:
			res = mon + split + day + split + hour + split + min + split + sec;
			break;
		case kTimeMonDay:
			res = mon + split + day;
			break;
		case kTimeHourMinSec:
			res = hour + split + min + split + sec;
			break;
		case kTimeYear:
			res = year;
			break;
		case kTimeMon:
			res = mon;
			break;
		case kTimeDay:
			res = day;
			break;
		case kTimeHour:
			res = hour;
			break;
		case kTimeMin:
			res = min;
			break;
		case kTimeSec:
			res = sec;
			break;
		default:
			res = year + "-" + mon + "-" + day + "-" + hour + "-" + min + "-" + sec;
			break;
		}

		VLOG(30) << res;
		return res;
	}

	std::string GetCurrTime(const std::initializer_list<string> &timeKeyWords, const std::string &split)
	{
		std::string res = "";

		time_t t;
		time(&t);
		struct tm *timeinfo;
		timeinfo = localtime(&t);

		string year = std::to_string(timeinfo->tm_year + 1900);
		string mon = ZeroPadding(timeinfo->tm_mon + 1, 2);
		string day = ZeroPadding(timeinfo->tm_mday, 2);
		string hour = ZeroPadding(timeinfo->tm_hour, 2);
		string min = ZeroPadding(timeinfo->tm_min, 2);
		string sec = ZeroPadding(timeinfo->tm_sec, 2);

		for (auto iter = timeKeyWords.begin(); iter != timeKeyWords.end(); iter++)
		{
			int timeType = GetTimeEnumType(*iter);
			switch (timeType)
			{
			case kTimeYear:
				res = res + split + year;
				break;
			case kTimeMon:
				res = res + split + mon;
				break;
			case kTimeDay:
				res = res + split + day;
				break;
			case kTimeHour:
				res = res + split + hour;
				break;
			case kTimeMin:
				res = res + split + min;
				break;
			case kTimeSec:
				res = res + split + sec;
				break;
			}
		}

		return DeleteStrParts(res, split, FRONT_2_BACK, 1);
	}

	int GetFilesNums(const std::string &root, const std::string &fileFormat)
	{
		std::string f = root + "*" + fileFormat;
		std::vector<cv::String> fp; // files path
		cv::glob(f, fp);
		return int(fp.size());
	}

	std::vector<std::string> GetFilesName(const std::string &root, const bool &hiddenDir)
	{
		std::vector<std::string> res;

		struct dirent *entry = nullptr;
		DIR *dp = nullptr;

		const char *cRoot = root.c_str();
		dp = opendir(cRoot);
		if (dp != nullptr)
		{
			while ((entry = readdir(dp)))
			{
				if (!hiddenDir && (std::string(entry->d_name) == "." || std::string(entry->d_name) == ".."))
					continue;
				res.push_back(std::move(std::string(entry->d_name)));
			}
		}

		closedir(dp);
		std::sort(res.begin(), res.end());
		return res;
	}

	std::vector<std::string> GetFilesPath(const std::string &root, const bool &hiddenDir)
	{
		std::vector<std::string> res;
		std::vector<std::string> names = GetFilesName(root, hiddenDir);
		for (auto &n : names)
			res.push_back(root + "/" + n);
		return res;
	}

	int GetStartIndex(const std::string &root, const std::string &prefix, const int digit, const std::string &format, const int &maxJudgeRange)
	{
		int nums = GetFilesNums(root, format);
		struct stat buffer;
		for (int i = 0; i < maxJudgeRange; ++i)
		{
			std::string path = root + prefix + ZeroPadding(i, digit) + format;
			if (stat(path.c_str(), &buffer) == 0)
			{
				VLOG(0) << "Start Index is " << i << " in " << root;
				return i;
			}
			continue;
		}
		LOG(WARNING) << "Can Not Find Start Index in " << root << ", -1 Is Returned.";
		return -1;
	}

	std::vector<std::string> GetStrParts(const std::string &s, const char &split)
	{
		std::vector<std::string> res;
		std::string retain = s;
		while (1)
		{
			size_t pos = retain.find_first_of(split);
			if (pos == std::string::npos)
			{
				res.push_back(retain);
				break;
			}
			std::string part = retain.substr(0, pos);
			retain = retain.substr(pos + 1, retain.size());
			if (part != "")
				res.push_back(part);
		}
		return res;
	}

	std::string GetTrackerName(const int &tracker)
	{
		std::string res;
		switch (tracker)
		{
		case kAWLB:
			res = "AWLB";
			break;
		case kEDF:
			res = "EDF";
			break;
		case kIP:
			res = "IP";
			break;
		case kRBOT:
			res = "RBOT";
			break;
		case kSearchLine:
			res = "SearchLine";
			break;
		case kSLOT:
			res = "SLOT";
			break;
		case kSummer:
			res = "Summer";
			break;
		case kRBGT:
			res = "RBGT";
			break;
		case kSRT3D:
			res = "SRT3D";
			break;
		case kICG:
			res = "ICG";
			break;
		case kRMT:
			res = "RMT";
			break;
		default:
			break;
		}
		return res;
	}

	bool MakeDir(const std::string &dir, const bool &make)
	{
		struct stat info;
		if (stat(dir.c_str(), &info) == 0 && S_ISDIR(info.st_mode))
		{
			VLOG(30) << "Directory already exists: " << dir << std::endl;
			return true;
		}

		if (!make)
			return false;
		// mkdir() would not run on online IDEs as the program requires the directory path in the system itself.
		// https://www.geeksforgeeks.org/create-directoryfolder-cc-program/
		const char *cdir = dir.c_str();
		int result = mkdir(cdir, 0777);
		if (result == 0)
		{
			VLOG(0) << "Directory created: " << dir << std::endl;
			return true;
		}
		else
		{
			VLOG(0) << "Failed to create directory: " << dir << std::endl;
			return false;
		}
	}

	std::string MakeTrackingResultDir(const std::string &root, const int &tracker, const std::string &additionInfo, const std::string &split, const bool &make)
	{
		std::string resultSaveDir = root;
		std::string trackerSaveDir = resultSaveDir + "/" + GetTrackerName(tracker);
		std::string todayDir = trackerSaveDir + "/" + GetCurrTime({"M", "D"}, split);
		std::string hourMinSecDir = todayDir + "/" + GetCurrTime({"h", "m", "s"}, split);
		bool dir1 = MakeDir(resultSaveDir, make);
		bool dir2 = MakeDir(trackerSaveDir, make);
		bool dir3 = MakeDir(todayDir, make);
		bool dir4 = MakeDir(hourMinSecDir + split + additionInfo, make);
		return hourMinSecDir + split + additionInfo;
	}

	bool SaveFloatImg(cv::Mat &img, const std::string &savePath)
	{
		std::ofstream ofs(savePath);
		ofs << img.rows << " " << img.cols << " ";

		for (int i = 0; i < img.rows; i++)
		{
			float *p = img.ptr<float>(i);
			for (int j = 0; j < img.cols; j++)
			{
				ofs << p[j] << " ";
			}
			ofs << "\n";
		}

		ofs.clear();
		ofs.close();
		return true;
	}

	void SavePose2txt(std::ofstream *ofs, std::initializer_list<int> data, const cv::Matx44f &pose, const std::string &split)
	{
		for (auto iter = data.begin(); iter != data.end(); iter++)
		{
			(*ofs) << *iter << split;
		}

		(*ofs) << pose(0, 0) << split << pose(0, 1) << split << pose(0, 2) << split
			   << pose(1, 0) << split << pose(1, 1) << split << pose(1, 2) << split
			   << pose(2, 0) << split << pose(2, 1) << split << pose(2, 2) << split
			   << pose(0, 3) << split << pose(1, 3) << split << pose(2, 3) << "\n";
	}

	void SaveFloat2txt(std::ofstream *ofs, std::initializer_list<float> data, const std::string &split)
	{
		for (auto iter = data.begin(); iter != data.end(); iter++)
		{
			(*ofs) << *iter << split;
		}
	}

	void SaveTrackingResult(const std::string &root, TrackingResult *tr, const std::string &split)
	{
		if (!g_save)
			return;

		string indexRoot = root + "/index.txt";
		string KRoot = root + "/K.txt";
		string successRoot = root + "/success.txt";
		string modelRadiusRoot = root + "/modelRadius.txt";
		string timeRoot = root + "/time.txt";
		string failRoot = root + "/fail.txt";
		string gtRoot = root + "/gt.txt";
		string poseRoot = root + "/pose.txt";
		string diffRoot = root + "/diff.txt";
		string filtererPoseRoot = root + "/filtered_pose.txt";
		string compensatedPoseRoot = root + "/compensated_pose.txt";
		string roiRoot = root + "/roi.txt";
		string prevCornersRoot = root + "/prev_corners";
		string maskRoot = root + "/mask";
		string depthRoot = root + "/depth";
		string fineDepthRoot = root + "/fine_depth";
		string originRoot = root + "/origin";
		string overlayRoot = root + "/overlay";
		string probabilityRoot = root + "/probability";

		MakeDir(prevCornersRoot);
		MakeDir(maskRoot);
		MakeDir(depthRoot);
		MakeDir(fineDepthRoot);
		MakeDir(originRoot);
		MakeDir(overlayRoot);
		MakeDir(probabilityRoot);

		if (g_save_index)
		{
			std::ofstream ofs(indexRoot, std::ios::app);
			ofs << tr->index << "\n";
		}

		if (g_save_K)
		{
			std::ofstream ofs(KRoot, std::ios::app);
			ofs << tr->index << split << tr->K(0, 0) << split << tr->K(1, 1)
				<< split << tr->K(0, 2) << split << tr->K(1, 2) << "\n";
		}

		if (g_save_model_radius)
		{
			std::ofstream ofs(modelRadiusRoot, std::ios::app);
			ofs << tr->modelRadius << "\n";
		}

		if (g_save_roi)
		{
			std::ofstream ofs(roiRoot, std::ios::app);
			ofs << tr->index << " " << tr->roi.x << " " << tr->roi.y << " " << tr->roi.width << " " << tr->roi.height << "\n";
		}

		if (g_save_time)
		{
			std::ofstream ofs(timeRoot, std::ios::app);
			ofs << tr->index << split << tr->time << "\n";
		}

		if (g_save_success)
		{
			std::ofstream ofs(successRoot, std::ios::app);
			std::ofstream ofs_fail(failRoot, std::ios::app);
			SavePose2txt(&ofs, {tr->index, tr->success}, tr->pose, split);
			if (!tr->success)
				SavePose2txt(&ofs_fail, {tr->index}, tr->pose, split);
		}

		if (g_save_gt)
		{
			std::ofstream ofs(gtRoot, std::ios::app);
			SavePose2txt(&ofs, {tr->index}, tr->gt, split);
		}

		if (g_save_pose)
		{
			std::ofstream ofs(poseRoot, std::ios::app);
			SavePose2txt(&ofs, {tr->index}, tr->pose, split);
		}

		if (g_save_diff)
		{
			std::ofstream ofs(diffRoot, std::ios::app);
			cv::Matx44f diff = tr->pose - tr->gt;
			SavePose2txt(&ofs, {tr->index}, diff, split);
		}

		if (g_save_filtered_pose)
		{
			std::ofstream ofs(filtererPoseRoot, std::ios::app);
			SavePose2txt(&ofs, {tr->index}, tr->filteredPose, split);
		}

		if (g_save_compensated_pose)
		{
			std::ofstream ofs(compensatedPoseRoot, std::ios::app);
			SavePose2txt(&ofs, {tr->index}, tr->compensatedPose, split);
		}

		if (g_save_prev_corners)
		{
			std::string sp = prevCornersRoot + "/" + std::to_string(tr->index) + ".png"; // depth save path
			if (!tr->prevCorners.empty())
				cv::imwrite(sp, tr->prevCorners); // 10
		}

		if (g_save_mask)
		{
			std::string sp = maskRoot + "/" + std::to_string(tr->index) + ".png"; // depth save path
			if (!tr->mask.empty())
				cv::imwrite(sp, tr->mask); // 10
		}

		if (g_save_depth)
		{
			std::string sp = depthRoot + "/" + std::to_string(tr->index) + ".txt"; // depth save path
			if (!tr->depth.empty())
			{
				cv::Mat depthSave = tr->depth.clone();
				SaveFloatImg(depthSave, sp); // 10
			}
		}

		if (g_save_fine_depth)
		{
			std::string sp = fineDepthRoot + "/" + std::to_string(tr->index) + ".png"; // depth save path
			if (!tr->fineDepth.empty())
				cv::imwrite(sp, tr->fineDepth * 2000 * 55); // 10
		}

		if (g_save_origin)
		{
			std::string sp = originRoot + "/" + std::to_string(tr->index) + ".png"; // origin save path
			if (!tr->origin.empty())
				cv::imwrite(sp, tr->origin); // 10
		}

		if (g_save_overlay)
		{
			VLOG(0) << "tr->index " << overlayRoot << " " << tr->index;
			std::string sp = overlayRoot + "/" + std::to_string(tr->index) + ".png"; // overlay save path
			if (!tr->overlay.empty())
				cv::imwrite(sp, tr->overlay); // 10
		}

		if (g_save_probability)
		{
			std::string sp = probabilityRoot + "/" + std::to_string(tr->index) + ".png"; // probability save path
			if (!tr->probability.empty())
				cv::imwrite(sp, tr->probability); // 10
		}
	}

	void SplitTxt2MultiTxtFiles(const std::string &inputTxt, const std::string &outputDir, const bool &deleteIndex)
	{
		std::ifstream inputFile(inputTxt);
		std::string line;

		if (inputFile.is_open())
		{
			int lineNumber = 0;
			while (std::getline(inputFile, line))
			{
				if (deleteIndex)
				{
					// 去掉每一行的第一个数据
					size_t pos = line.find(' ');
					if (pos != std::string::npos)
					{
						line = line.substr(pos + 1);
					}
				}

				// 将每一行数据存成一个txt文件
				std::ofstream outputFile(outputDir + std::to_string(lineNumber) + ".txt");
				outputFile << line;
				outputFile.close();

				lineNumber++;
			}

			inputFile.close();
		}
		else
		{
			VLOG(0) << "无法打开文件" << std::endl;
		}

		return;
	}

	std::string ZeroPadding(const int &idx, const int &number)
	{
		std::stringstream ss;
		ss << std::setw(number) << std::setfill('0') << std::to_string(idx);
		return ss.str();
	}

} // namesapce ar3dv;