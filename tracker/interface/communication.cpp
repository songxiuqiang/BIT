#include "communication.h"

namespace ar3dv
{

	LocalStorageCommunicator *LocalStorageCommunicator::m_instance;

	LocalStorageCommunicator::LocalStorageCommunicator()
	{
		VLOG(0) << " Init a LocalStorageCommunicator";
	}

	LocalStorageCommunicator::~LocalStorageCommunicator()
	{
		if (m_instance)
			delete m_instance;
		VLOG(0) << "Destruct a LocalStorageCommunicator";
	}

	LocalStorageCommunicator *LocalStorageCommunicator::Instance()
	{
		if (m_instance == nullptr)
			m_instance = new LocalStorageCommunicator();
		return m_instance;
	}

	bool LocalStorageCommunicator::ClearCache()
	{
		LOG(INFO) << "Clearing cache in Cpp/Python communication...";
		DeleteFilesInDir(DeleteStrParts(pose_save_dir_, "/", 1, 1));
		DeleteFilesInDir(DeleteStrParts(img_save_dir_, "/", 1, 1));
		DeleteFilesInDir(DeleteStrParts(prob_save_dir_, "/", 1, 1));
		DeleteFilesInDir(DeleteStrParts(mask_save_dir_, "/", 1, 1));
		DeleteFilesInDir(DeleteStrParts(K_save_dir_, "/", 1, 1));
		DeleteFilesInDir(DeleteStrParts(origin_roi_save_dir_, "/", 1, 1));
		DeleteFilesInDir(DeleteStrParts(target_roi_save_dir_, "/", 1, 1));
		DeleteFilesInDir(DeleteStrParts(target_size_save_dir_, "/", 1, 1));
		DeleteFilesInDir(DeleteStrParts(models_dir_, "/", 1, 1));
		LOG(INFO) << "Clear cahce done";
		cache_cleared_ = true;
		return true;
	}

	void ApplyThreshold(cv::Mat &mask, uchar a)
	{
		// Ensure the mask is of type CV_8UC3
		CV_Assert(mask.type() == CV_8UC3);

		// Iterate through each pixel in the mask
		for (int r = 0; r < mask.rows; ++r)
		{
			for (int c = 0; c < mask.cols; ++c)
			{
				cv::Vec3b &pixel = mask.at<cv::Vec3b>(r, c);

				// Check if the pixel matches (a, a, a)
				if (pixel[0] == a && pixel[1] == a && pixel[2] == a)
				{
					pixel = cv::Vec3b(255, 255, 255); // Set the pixel to (255, 255, 255)
				}
				else
				{
					pixel = cv::Vec3b(0, 0, 0); // Set the pixel to (0, 0, 0)
				}
			}
		}
	}

	cv::Matx33f ComCenteredScaledIntrinsics(const cv::Matx33f &K, const float &w, const float &h, const cv::Rect &roi, const float &scale)
	{
		float x = float(roi.x) + 0.5f * float(roi.width);
		float y = float(roi.y) + 0.5f * float(roi.height);
		cv::Vec2f currCenter(x, y);
		cv::Matx33f res = cv::Matx33f::eye();
		float delta_cx = float(currCenter[0] * scale - 0.5f * w); // 0.5w is the center of image
		float delta_cy = float(currCenter[1] * scale - 0.5f * h);
		res(0, 0) = K(0, 0) * scale;
		res(1, 1) = K(1, 1) * scale;
		res(0, 2) = K(0, 2) * scale - delta_cx;
		res(1, 2) = K(1, 2) * scale - delta_cy;
		return res;
	}

	bool LocalStorageCommunicator::SetUp(const std::string &root, const std::string &type)
	{
		set_up_ = false;

		cmc_root_ = pjoin(root, type);

		pose_save_dir_ = pjoin(cmc_root_, "pose/");
		img_save_dir_ = pjoin(cmc_root_, "img/");
		VLOG(0) << "img_save_dir_ " << img_save_dir_;
		prob_save_dir_ = pjoin(cmc_root_, "prob/");
		mask_save_dir_ = pjoin(cmc_root_, "mask/");
		K_save_dir_ = pjoin(cmc_root_, "K/");
		origin_roi_save_dir_ = pjoin(cmc_root_, "originRoi/");
		target_roi_save_dir_ = pjoin(cmc_root_, "targetRoi/");
		target_size_save_dir_ = pjoin(cmc_root_, "targetSize/");
		models_dir_ = pjoin(cmc_root_, "deformed_model/");
		std::string resclaed_models_dir = pjoin(cmc_root_, "deformed_model_rescaled/");

		MakeDir(cmc_root_);

		if (MakeDir(pose_save_dir_) &&
			MakeDir(img_save_dir_) &&
			MakeDir(prob_save_dir_) &&
			MakeDir(mask_save_dir_) &&
			MakeDir(K_save_dir_) &&
			MakeDir(origin_roi_save_dir_) &&
			MakeDir(target_roi_save_dir_) &&
			MakeDir(target_size_save_dir_) &&
			MakeDir(models_dir_) &&
			MakeDir(resclaed_models_dir))
		{
			set_up_ = true;
		}

		if (!set_up_)
			VLOG(0) << "Communicator can not be seted up!";

		return set_up_;
	}

	bool LocalStorageCommunicator::SetUp(const SummerConfigs &sc)
	{
		set_up_ = false;

		cmc_root_ = sc.cmc_root;

		pose_save_dir_ = pjoin(sc.cmc_root, "pose/");
		img_save_dir_ = pjoin(sc.cmc_root, "img/");
		prob_save_dir_ = pjoin(sc.cmc_root, "prob/");
		mask_save_dir_ = pjoin(sc.cmc_root, "mask/");
		K_save_dir_ = pjoin(sc.cmc_root, "K/");
		origin_roi_save_dir_ = pjoin(sc.cmc_root, "originRoi/");
		target_roi_save_dir_ = pjoin(sc.cmc_root, "targetRoi/");
		target_size_save_dir_ = pjoin(sc.cmc_root, "targetSize/");
		models_dir_ = pjoin(sc.cmc_root, "deformed_model/");

		MakeDir(sc.cmc_root);

		if (MakeDir(pose_save_dir_) &&
			MakeDir(img_save_dir_) &&
			MakeDir(prob_save_dir_) &&
			MakeDir(mask_save_dir_) &&
			MakeDir(K_save_dir_) &&
			MakeDir(origin_roi_save_dir_) &&
			MakeDir(target_roi_save_dir_) &&
			MakeDir(target_size_save_dir_) &&
			MakeDir(models_dir_))
		{
			set_up_ = true;
		}

		if (!set_up_)
			VLOG(0) << "Communicator can not be seted up!";

		if (set_up_)
		{
			ClearCache();

			std::string sourceDirsRoot = "/result/communication/";
			std::string targetDirsRoot = sc.cmc_root;
			std::vector<std::string> sourceDirs{"K/", "pose/", "img/", "prob/", "mask/", "originRoi/", "targetRoi/", "targetSize/"};
			std::vector<std::string> targetDirs{"K/", "pose/", "img/", "prob/", "mask/", "originRoi/", "targetRoi/", "targetSize/"};

			for (size_t dirIndex = 0; dirIndex < sourceDirs.size(); dirIndex++)
			{
				sourceDirs[dirIndex] = sourceDirsRoot + sourceDirs[dirIndex];
				targetDirs[dirIndex] = targetDirsRoot + targetDirs[dirIndex];
			}

			std::vector<std::string> posesFilesPath = GetFilesPath(sourceDirs[1]);
			std::vector<cv::Matx44f> poses;
			poses.reserve(posesFilesPath.size());

			for (const auto &file : posesFilesPath)
			{
				poses.push_back(LoadPoses(file, 0, 0, ' ')[0]);
			}

			std::vector<int> targetIds;
			picker_->GenTemplateViews();
			for (auto &view : picker_->template_views())
			{
				int index = picker_->PickClosetViews(view, poses);
				targetIds.push_back(index);
			}

			ProvideRefers(sourceDirs, targetDirs, targetIds, 6);
		}

		return set_up_;
	}
	bool LocalStorageCommunicator::InRefers(const int &index)
	{
		// Check if the index is in the refers_ids_ vector
		if (std::find(refers_ids_.begin(), refers_ids_.end(), index) != refers_ids_.end())
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	bool LocalStorageCommunicator::UseRefers(const std::string &root, const int &n)
	{
		VLOG(0) << __func__ << " " << n;
		std::string sourceDirsRoot = root + "refers/";
		std::string targetDirsRoot = root + "cmc/";
		std::vector<std::string> sourceDirs{"K/", "pose/", "img/", "prob/", "mask/", "originRoi/", "targetRoi/", "targetSize/"};
		std::vector<std::string> targetDirs{"K/", "pose/", "img/", "prob/", "mask/", "originRoi/", "targetRoi/", "targetSize/"};

		for (size_t dirIndex = 0; dirIndex < sourceDirs.size(); dirIndex++)
		{
			sourceDirs[dirIndex] = sourceDirsRoot + sourceDirs[dirIndex];
			targetDirs[dirIndex] = targetDirsRoot + targetDirs[dirIndex];
		}
		std::vector<std::string> posesFilesPath = GetFilesPath(sourceDirs[1]);
		std::vector<cv::Matx44f> poses;
		poses.reserve(posesFilesPath.size());
		for (const auto &file : posesFilesPath)
		{
			poses.push_back(LoadPoses(file, 0, 0, ' ')[0]);
		}

		std::vector<int> targetIds;
		picker_->GenTemplateViews(n);
		for (auto &view : picker_->template_views())
		{
			int index = picker_->PickClosetViews(view, poses, targetIds);
			if (index == 0)
				index += 1;
			if (index == -1)
				std::cerr << "Not enough refer views";
			targetIds.push_back(index);
			refers_ids_.push_back(index);
			VLOG(0) << "picker " << index << " " << view[0] << " " << view[1] << " " << view[2];
		}
		targetIds.push_back(0);
		refers_ids_.push_back(0);
		ProvideRefers(sourceDirs, targetDirs, targetIds, 6);
		return true;
	}

	bool LocalStorageCommunicator::UseMultiReferenceMoped(const std::string &evaRoot, const std::string &refRoot, const int &n)
	{
		VLOG(0) << __func__ << " " << n;
		std::string refDirsRoot = refRoot + "refers/";
		std::string evaDirsRoot = evaRoot + "cmc/";
		std::string evaRefersDirsRoot = evaRoot + "refers/";

		std::vector<std::string> refDirs{"K/", "pose/", "img/", "prob/", "mask/", "originRoi/", "targetRoi/", "targetSize/"};
		std::vector<std::string> evaDirs{"K/", "pose/", "img/", "prob/", "mask/", "originRoi/", "targetRoi/", "targetSize/"};
		std::vector<std::string> evaRefersDirs{"K/", "pose/", "img/", "prob/", "mask/", "originRoi/", "targetRoi/", "targetSize/"};

		for (size_t dirIndex = 0; dirIndex < refDirs.size(); dirIndex++)
		{
			refDirs[dirIndex] = refDirsRoot + refDirs[dirIndex];
			evaDirs[dirIndex] = evaDirsRoot + evaDirs[dirIndex];
			evaRefersDirs[dirIndex] = evaRefersDirsRoot + evaRefersDirs[dirIndex];
		}

		std::vector<std::string> posesFilesPath = GetFilesPath(refDirs[1]);
		std::vector<cv::Matx44f> poses;
		poses.reserve(posesFilesPath.size());
		for (const auto &file : posesFilesPath)
		{
			poses.push_back(LoadPoses(file, 0, 0, ' ')[0]);
		}

		std::vector<int> targetIds;
		picker_->GenTemplateViews(n);
		for (auto &view : picker_->template_views())
		{
			int index = picker_->PickClosetViews(view, poses, targetIds);
			if (index == 0)
				index += 1;
			if (std::find(targetIds.begin(), targetIds.end(), index) != targetIds.end())
				index += 1;
			VLOG(0) << "index " << index;
			if (index == -1)
				std::cerr << "Not enough refer views";
			targetIds.push_back(index);
			refers_ids_.push_back(index);
			VLOG(0) << "picker " << index << " " << view[0] << " " << view[1] << " " << view[2];
		}

		// refers provided by refs
		ProvideRefers(refDirs, evaDirs, targetIds, 6);

		// refers by the first frame
		for (size_t i = 0; i < evaRefersDirs.size(); i++)
		{
			std::vector<std::string> sourceFilesNames = GetFilesName(evaRefersDirs[i]);
			std::string sourceFile = evaRefersDirs[i] + sourceFilesNames[0];
			std::string targetFile = evaDirs[i] + sourceFilesNames[0];
			CopyFile(sourceFile, targetFile);
		}

		return true;
	}

	bool LocalStorageCommunicator::set_up(const std::string &type) const
	{
		return set_up_;
	}

	std::string LocalStorageCommunicator::models_dir() const
	{
		return models_dir_;
	}

	bool LocalStorageCommunicator::SentDatas2Py(TrackingResult *trackingResult, const int &index)
	{
		this->WriteData<cv::Mat>(trackingResult->dataGroup.img, index, "img");
		this->WriteData<cv::Mat>(trackingResult->dataGroup.prob, index, "prob");
		this->WriteData<cv::Mat>(trackingResult->dataGroup.mask, index, "mask");
		this->WriteData<cv::Rect>(trackingResult->dataGroup.originRoi, index, "origin_roi");
		this->WriteData<cv::Rect>(trackingResult->dataGroup.targetRoi, index, "target_roi");
		this->WriteData<cv::Vec2f>(trackingResult->dataGroup.targetSize, index, "size");
		this->WriteData<cv::Matx33f>(trackingResult->dataGroup.K, index, "K");

		cv::Matx44f pose = trackingResult->dataGroup.pose;
		for (int i = 0; i < 3; ++i)
			pose(i, 3) /= trackingResult->bbxLongestSide;
		this->WriteData<cv::Matx44f>(pose, index, "pose");

		return true;
	}

	bool LocalStorageCommunicator::WritePose(const cv::Matx44f &pose, const int &index)
	{
		std::string save_file = pose_save_dir_ + ZeroPadding(index, 4) + ".pose";
		m_ofs_pose.open(save_file, std::ios::out | std::ios::trunc);

		if (m_ofs_pose.is_open())
		{
			m_ofs_pose << pose(0, 0) << " " << pose(0, 1) << " " << pose(0, 2) << " "
					   << pose(1, 0) << " " << pose(1, 1) << " " << pose(1, 2) << " "
					   << pose(2, 0) << " " << pose(2, 1) << " " << pose(2, 2) << " "
					   << pose(0, 3) << " " << pose(1, 3) << " " << pose(2, 3) << "\n";
			m_ofs_pose.close();
			return true;
		}
		else
		{
			LOG(WARNING) << "Can Not Open " << save_file << " to Write Pose";
			m_ofs_pose.close();
			return false;
		}
		return false;
	}

	bool LocalStorageCommunicator::WriteDataGroup(const DataGroup &dataGroup, const int &index)
	{
		VLOG(0) << "dataGroup K pose " << dataGroup.K << "\n"
				<< dataGroup.pose;
		bool img = WriteData<cv::Mat>(dataGroup.prob, index, "prob");
		bool pose = WritePose(dataGroup.pose, index);
		bool k = WriteData<cv::Matx33f>(dataGroup.K, index, "");
		return k && pose && img;
	}

	bool LocalStorageCommunicator::ScanNewFile(const std::string &directoryPath)
	{
		bool get = false;
		std::vector<std::string> newFiles = GetFilesPath(directoryPath);
		for (auto &path : newFiles)
			if (std::find(m_models_path.begin(), m_models_path.end(), path) == m_models_path.end())
			{
				m_models_path.push_back(path);
				get = true;
			}

		if (!m_models_path.empty())
			std::sort(m_models_path.begin(), m_models_path.end());

		return get;
	}

	std::string LocalStorageCommunicator::newest_model_path() const
	{
		CHECK(!m_models_path.empty()) << "No model can be parded!";
		return m_models_path.back();
	}

	std::shared_ptr<Picker> LocalStorageCommunicator::picker()
	{
		return picker_;
	}

	void Normalize(cv::Vec3f &vec)
	{
		float norm = cv::norm(vec);
		if (norm > 0)
		{
			vec /= norm;
		}
	}

	float AngleBetween(const cv::Vec3f &v1, const cv::Vec3f &v2)
	{
		float dot = v1.dot(v2);
		float angle = std::acos(std::min(std::max(dot, -1.0f), 1.0f));
		return angle;
	}

	void LocalStorageCommunicator::PickViews(const std::vector<cv::Matx44f> &poses, const int &n)
	{
		std::vector<cv::Vec3f> views;
		views.reserve(poses.size());
		for (const auto &pose : poses)
		{
			cv::Matx33f R(pose(0, 0), pose(0, 1), pose(0, 2),
						  pose(1, 0), pose(1, 1), pose(1, 2),
						  pose(2, 0), pose(2, 1), pose(2, 2));
			cv::Vec3f t(pose(0, 3), pose(1, 3), pose(2, 3));
			cv::Matx33f R_inv = R.t();
			cv::Vec3f view = R_inv * (-t);
			Normalize(view);
			views.push_back(view);
		}

		std::vector<cv::Vec3f> selected_views;
		selected_views.push_back(views[0]);
		while (selected_views.size() < n)
		{
			float max_angle = 0;
			int max_idx = 0;
			for (int i = 0; i < views.size(); i++)
			{
				float min_angle = std::numeric_limits<float>::max();
				for (const auto &selected_view : selected_views)
				{
					float angle = AngleBetween(views[i], selected_view);
					min_angle = std::min(min_angle, angle);
				}
				if (min_angle > max_angle)
				{
					max_angle = min_angle;
					max_idx = i;
				}
			}
			VLOG(0) << max_idx + 1;
			selected_views.push_back(views[max_idx]);
		}

		std::cout << "Selected views:" << std::endl;
		for (const auto &view : selected_views)
		{
			std::cout << view << std::endl;
		}
	}

	void LocalStorageCommunicator::SetExpectedDataNumsSent2Py(std::vector<int> &datas)
	{
		expected_data_nums_sent_2_py_ = std::move(datas);
		return;
	}

	void LocalStorageCommunicator::DataNumsSent2PyAddOne()
	{
		++data_nums_sent_2_py_;
		return;
	}
	int LocalStorageCommunicator::data_nums_sent_2_py() const
	{
		return data_nums_sent_2_py_;
	}
	int LocalStorageCommunicator::expected_data_nums_sent_2_py_up_2_current(const int &n) const
	{
		return std::accumulate(expected_data_nums_sent_2_py_.begin(), expected_data_nums_sent_2_py_.begin() + n + 1, 0);
	}

	std::vector<Eigen::Vector3f> Picker::template_views() const
	{
		return template_views_;
	}

	void Picker::VisualizeGeodesicPoints()
	{

		cv::Mat visual(800, 800, CV_8UC3, cv::Scalar(255, 255, 255));

		cv::Matx44f T = cv::Matx44f::eye();
		T(2, 3) = 2.0f;

		cv::Matx44f K = cv::Matx44f::eye();
		K(0, 2) = 400;
		K(1, 2) = 400;
		K(0, 0) = 200;
		K(1, 1) = 200;

		std::vector<cv::Vec3f> pts;
		std::vector<cv::Vec3i> Edges;
		std::vector<cv::Vec2f> uvs;
		for (auto &p : geodesic_points_)
		{
			pts.push_back(cv::Vec3f(p.x(), p.y(), p.z()));
			cv::Vec4f pHomo = cv::Vec4f(p.x(), p.y(), p.z(), 1.0f);
			cv::Vec4f uvHomo = K * T * pHomo;
			VLOG(0) << "uvHomo " << uvHomo;
			cv::Vec2f uv(uvHomo[0] / uvHomo[2], uvHomo[1] / uvHomo[2]);
			uvs.push_back(uv);
		}

		for (int i = 0; i < pts.size(); ++i)
		{
			cv::Vec3f p = pts[i];
			std::vector<std::tuple<float, int>> distances;

			// Calculate distance from p to all other points
			for (int j = 0; j < pts.size(); ++j)
			{
				if (i != j)
				{
					cv::Vec3f q = pts[j];
					float dist = cv::norm(p - q); // Euclidean distance
					distances.push_back(std::make_tuple(dist, j));
				}
			}

			// Sort distances
			std::sort(distances.begin(), distances.end());

			// Get indices of 3 nearest points
			std::vector<int> nearest_indices;
			for (int k = 0; k < 3 && k < distances.size(); ++k)
			{
				nearest_indices.push_back(std::get<1>(distances[k]));
			}

			// Store edges
			Edges.push_back(cv::Vec3i(nearest_indices[0], nearest_indices[1], nearest_indices[2]));
		}

		for (const auto &edge : geodesic_edges_)
		{
			cv::Point2f(edge.first);
			cv::line(visual, cv::Point2f(edge.first), cv::Point2f(edge.second), cv::Scalar(0, 0, 255), 2);
		}

		cv::imshow("visual", visual);
		cv::waitKey(0);
		VLOG(0) << "pts size " << pts.size();

		return;
	}

	void Picker::GenGeodesicPoints(int n)
	{
		geodesic_points_.clear();
		geodesic_points_.reserve(n);

		if (n == 1)
		{
			geodesic_points_.emplace_back(0.0f, 1.0f, 0.0f);
			VLOG(0) << "Geodesic Points  " << 0.0f << " " << 1.0f << " " << 0.0f;
			return;
		}

		const float phi = (1.0f + std::sqrt(5.0f)) / 2.0f; // Au ratio

		for (int i = 0; i < n; ++i)
		{
			float y = 1 - (i / float(n - 1)) * 2; // y goes from 1 to -1
			float radius = std::sqrt(1 - y * y);  // radius at y

			float theta = 2.0f * M_PI * i / phi; // golden angle increment

			float x = cos(theta) * radius;
			float z = sin(theta) * radius;

			geodesic_points_.emplace_back(x, y, z);
			VLOG(0) << "Geodesic Points " << x << " " << y << " " << z;
		}
		return;
	}

	void Picker::GenGeodesicPoints()
	{
		constexpr float x = 0.525731112119133606f;
		constexpr float z = 0.850650808352039932f;
		std::vector<Eigen::Vector3f> icosahedron_points{
			{-x, 0.0f, z}, {x, 0.0f, z}, {-x, 0.0f, -z}, {x, 0.0f, -z}, {0.0f, z, x}, {0.0f, z, -x}, {0.0f, -z, x}, {0.0f, -z, -x}, {z, x, 0.0f}, {-z, x, 0.0f}, {z, -x, 0.0f}, {-z, -x, 0.0f}};
		geodesic_points_.clear();
		for (const auto &icosahedron_point : icosahedron_points)
		{
			geodesic_points_.push_back(icosahedron_point);
		}
	}

	void Picker::GenTemplateViews(const int &n)
	{
		GenGeodesicPoints(n);
		Eigen::Vector3f downwards{0.0f, 1.0f, 0.0f};
		camera2body_poses_.clear();
		for (const auto &geodesic_point : geodesic_points_)
		{
			Eigen::Transform<float, 3, 2> pose;
			pose = Eigen::Translation<float, 3>{geodesic_point * sphere_radius_};
			Eigen::Matrix3f Rotation;
			Rotation.col(2) = -geodesic_point;
			Rotation.col(0) = downwards.cross(-geodesic_point).normalized();
			if (Rotation.col(0).sum() == 0)
			{
				Rotation.col(0) = Eigen::Vector3f{1.0f, 0.0f, 0.0f};
			}
			Rotation.col(1) = Rotation.col(2).cross(Rotation.col(0));
			pose.rotate(Rotation);
			camera2body_poses_.push_back(pose);
		}
		template_views_.clear();
		template_views_.resize(camera2body_poses_.size());
		for (int i = 0; i < int(template_views_.size()); ++i)
		{
			template_views_[i] = camera2body_poses_[i].matrix().col(2).segment(0, 3);
		}
		return;
	}

	void Picker::VisualizePoints(const std::string &window_name, int image_size)
	{
		cv::Mat image(image_size, image_size, CV_8UC3, cv::Scalar(255, 255, 255));

		float f = image_size / 4.0f;
		cv::Point2f center(image_size / 2.0f, image_size / 2.0f);

		for (const auto &point : template_views_)
		{
			float x = point.x();
			float y = point.y();
			float z = point.z();
			if (z == 0.0f)
				continue;

			float u = f * (x / z) + center.x;
			float v = f * (y / z) + center.y;
			VLOG(0) << "u v " << u << " " << v;
			if (u >= 0 && u < image_size && v >= 0 && v < image_size)
			{
				cv::circle(image, cv::Point2f(u, v), 3, cv::Scalar(0, 0, 255), -1);
			}
		}

		cv::imshow(window_name, image);
		cv::waitKey(0);
	}

	int Picker::PickClosetViews(const Eigen::Vector3f &template_view, const std::vector<cv::Matx44f> &body2cam_poses)
	{
		int index = 0;
		int target_index = 0;
		float closest_dot = -1.0f;
		for (auto &body2cam_pose : body2cam_poses)
		{
			if (body2cam_pose(0, 0) == 2.0)
			{
				index++;
				continue;
			}
			Eigen::Matrix4f mat;
			mat << body2cam_pose(0, 0), body2cam_pose(0, 1), body2cam_pose(0, 2), body2cam_pose(0, 3),
				body2cam_pose(1, 0), body2cam_pose(1, 1), body2cam_pose(1, 2), body2cam_pose(1, 3),
				body2cam_pose(2, 0), body2cam_pose(2, 1), body2cam_pose(2, 2), body2cam_pose(2, 3),
				0.0f, 0.0f, 0.0f, 1.0f;
			Eigen::Transform<float, 3, 2> matAffine(mat);
			Eigen::Vector3f orientation{
				matAffine.rotation().inverse() *
				matAffine.translation().matrix().normalized()};

			float dot = orientation.dot(template_view);
			if (dot > closest_dot)
			{
				target_index = index;
				closest_dot = dot;
			}
			index++;
		}
		return target_index;
	}

	int Picker::PickClosetViews(const Eigen::Vector3f &template_view,
								const std::vector<cv::Matx44f> &body2cam_poses,
								const std::vector<int> &targetIds)
	{
		int index = 0;
		int target_index = -1;
		float closest_dot = -1.0f;
		for (auto &body2cam_pose : body2cam_poses)
		{
			if (body2cam_pose(0, 0) == 2.0)
			{
				index++;
				continue;
			}

			if (std::find(targetIds.begin(), targetIds.end(), index) != targetIds.end())
			{
				index++;
				continue;
			}

			Eigen::Matrix4f mat;
			mat << body2cam_pose(0, 0), body2cam_pose(0, 1), body2cam_pose(0, 2), body2cam_pose(0, 3),
				body2cam_pose(1, 0), body2cam_pose(1, 1), body2cam_pose(1, 2), body2cam_pose(1, 3),
				body2cam_pose(2, 0), body2cam_pose(2, 1), body2cam_pose(2, 2), body2cam_pose(2, 3),
				0.0f, 0.0f, 0.0f, 1.0f;
			Eigen::Transform<float, 3, 2> matAffine(mat);
			Eigen::Vector3f orientation{
				matAffine.rotation().inverse() *
				matAffine.translation().matrix().normalized()};

			float dot = orientation.dot(template_view);
			if (dot > closest_dot)
			{
				target_index = index;
				closest_dot = dot;
			}
			index++;
		}
		return target_index;
	}

	InsideCommunicator::InsideCommunicator()
	{
		VLOG(0) << "Instance an InsideCommubicator";
	}

	InsideCommunicator::~InsideCommunicator()
	{
		VLOG(0) << "Destroy an InsideCommubicator";
	}

	std::shared_ptr<InsideCommunicator> InsideCommunicator::Instance()
	{
		if (!m_instance)
			m_instance = std::make_shared<InsideCommunicator>();
		return m_instance;
	}

	bool ProvideRefers(const std::vector<std::string> &sourceDirs,
					   const std::vector<std::string> &targetDirs,
					   const std::vector<int> &filesIds,
					   const int &maxCpNums)
	{

		for (size_t i = 0; i < sourceDirs.size(); i++)
		{
			std::vector<std::string> sourceFilesNames = GetFilesName(sourceDirs[i]);
			for (int index = 0; index < filesIds.size(); index++)
			{
				std::string sourceFile = sourceDirs[i] + sourceFilesNames[filesIds[index]];
				std::string targetFile = targetDirs[i] + sourceFilesNames[filesIds[index]];
				CopyFile(sourceFile, targetFile);
			}
		}
		return true;
	}

	DataPool::DataPool(const int &capacity) : capacity_(capacity)
	{
	}

	DataPool::~DataPool()
	{
	}

	bool DataPool::Clear()
	{
		datas_.clear();
		datas_.shrink_to_fit();
		return true;
	}

	int DataPool::current_size() const
	{
		return datas_.size();
	}

	bool DataPool::SetCapacity(const int &capacity)
	{
		capacity_ = capacity;
		return true;
	}

	bool DataPool::SetMinViewAngleDiffer(const float &angle)
	{
		min_view_angle_differ_ = angle;
		return true;
	}

	Eigen::Vector3f DataPool::CalculateView(const cv::Matx44f &pose) const
	{
		Eigen::Transform<float, 3, Eigen::Affine> body2camera_pose;
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				body2camera_pose.matrix()(i, j) = pose(i, j);
		Eigen::Vector3f orientation{body2camera_pose.rotation().inverse() * body2camera_pose.translation().matrix().normalized()};
		return orientation.normalized();
	}

	bool DataPool::CheckView(const cv::Matx44f &pose) const
	{
		Eigen::Vector3f v1 = CalculateView(pose);
		for (const auto &data : datas_)
		{
			Eigen::Vector3f v2 = data.view;
			float cosTheta = v1.dot(v2) / (v1.norm() * v2.norm());
			float theta = std::acos(cosTheta) * 180.0 / 3.14159265f;
			if (theta > min_view_angle_differ_)
				continue;
			else
				return false;
		}
		VLOG(0) << "view valid";
		return true;
	}

	bool DataPool::PushData(const DataGroup &dataGroup)
	{
		if (datas_.size() < capacity_)
		{
			datas_.push_back(dataGroup);
			return true;
		}
		return false;
	}

	bool DataPool::PrintInfo(const std::string &var) const
	{
		VLOG(0) << var << ":";
		if (var == "views" || var == "view")
			for (const auto &data : datas_)
				VLOG(0) << data.view;
		return true;
	}

} // ns ar3dv