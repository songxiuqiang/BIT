#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <math.h>

#include "util.hh"
#include "mbt/object3d.hh"
namespace summer{

class PoseReader {
public:
	virtual bool IsLostGT(const cv::Matx44f& gtm, const cv::Matx44f& tkm, const Object3D* object) = 0;
	virtual void Read(const std::string& pose_file, std::vector<std::vector<cv::Matx44f> >& poses) = 0;
	virtual void Read(const std::string& pose_file, std::vector<int>& fids, std::vector<std::vector<cv::Matx44f> >& poses) {
		CHECK(false) << "ERROR: NOT implement";
	}
};

class PoseReaderOPT : public PoseReader {
public:
	static cv::Matx44f ParseMat(const std::string& pose_str) {
		std::vector<std::string> strs = tk::split(pose_str, ' ');
		if (strs.size() < 9)
			strs = tk::split(pose_str, '\t');

		if (strs.size() < 9)
			CHECK(false) << "ERROR: Invalid pose file";
	
		float m00, m01, m02, m03;
		float m10, m11, m12, m13;
		float m20, m21, m22, m23;

		if (strs.size() == 13) {
			m00 = (float)std::atof(strs[1].c_str());
			m01 = (float)std::atof(strs[2].c_str());
			m02 = (float)std::atof(strs[3].c_str());
			m10 = (float)std::atof(strs[4].c_str());
			m11 = (float)std::atof(strs[5].c_str());
			m12 = (float)std::atof(strs[6].c_str());
			m20 = (float)std::atof(strs[7].c_str());
			m21 = (float)std::atof(strs[8].c_str());
			m22 = (float)std::atof(strs[9].c_str());
			m03 = (float)std::atof(strs[10].c_str());
			m13 = (float)std::atof(strs[11].c_str());
			m23 = (float)std::atof(strs[12].c_str());
		}

		if (strs.size() == 12) {
			m00 = (float)std::atof(strs[0].c_str());
			m01 = (float)std::atof(strs[1].c_str());
			m02 = (float)std::atof(strs[2].c_str());
			m10 = (float)std::atof(strs[3].c_str());
			m11 = (float)std::atof(strs[4].c_str());
			m12 = (float)std::atof(strs[5].c_str());
			m20 = (float)std::atof(strs[6].c_str());
			m21 = (float)std::atof(strs[7].c_str());
			m22 = (float)std::atof(strs[8].c_str());
			m03 = (float)std::atof(strs[9].c_str());
			m13 = (float)std::atof(strs[10].c_str());
			m23 = (float)std::atof(strs[11].c_str());
		}

		cv::Matx44f mat(
			m00, m01, m02, m03,
			m10, m11, m12, m13,
			m20, m21, m22, m23,
			0, 0, 0, 1);

		return mat;
	}

	virtual bool IsLostGT(const cv::Matx44f& gtm, const cv::Matx44f& tkm, const Object3D* object) override {
		return false;
	}

	virtual void Read(const std::string& pose_file, std::vector<std::vector<cv::Matx44f> >& poses) override {
		std::ifstream fs;
		fs.open(pose_file);
		if (!fs.is_open()) {
			std::cerr << "failed to open pose file, press any key to exit" << std::endl;
			getchar();
			exit(-1);
		}
		
		poses.resize(1);

		std::string line;
		while (true) {
			if (fs.eof()) break;
			std::getline(fs, line);
			if (line.size() < 4) {
				std::cout << "pose file to short" << std::endl;
				break;
			}

			poses[0].push_back(ParseMat(line));
		}
	}
};

class PoseReaderRBOT: public PoseReader {
public:
	static float get_errorR(cv::Matx33f R_gt, cv::Matx33f R_pred) {
		cv::Matx33f tmp = R_pred.t() * R_gt;
		float trace = tmp(0, 0) + tmp(1, 1) + tmp(2, 2);
		return acos((trace - 1) / 2.0f) * 180.0f / 3.14159265f;
	}

	static float get_errort(cv::Vec3f t_gt, cv::Vec3f t_pred) {
		float et0 = t_gt[0] - t_pred[0];
		float et1 = t_gt[1] - t_pred[1];
		float et2 = t_gt[2] - t_pred[2];
		return sqrt(et0 * et0 + et1 * et1 + et2 * et2);
	}      

	virtual bool IsLostGT(const cv::Matx44f& gtm, const cv::Matx44f& tkm, const Object3D* object) override {
		float rot_err = 5;
		float pos_err = 50;
		
		// std::cout << "gt " << gtm(0, 3) << " " << gtm(1, 3) << " " << gtm(2, 3) << "\n";
		// std::cout << "tk " << tkm(0, 3) << " " << tkm(1, 3) << " " << tkm(2, 3) << "\n";

		for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			if (std::isnan(tkm(i, j)))
				return true;

		cv::Matx33f gtR = gtm.get_minor<3, 3>(0, 0);
		cv::Vec3f gtT(gtm(0,3), gtm(1,3), gtm(2,3));
		cv::Matx33f tkR = tkm.get_minor<3, 3>(0, 0);
		cv::Vec3f tkT(tkm(0,3), tkm(1,3), tkm(2,3));
		//std::cout << "errorR: "<<get_errorR(gtR, tkR) << "  errorT: " << get_errort(gtT, tkT) << std::endl;
		return (get_errorR(gtR, tkR) > rot_err || get_errort(gtT, tkT) > pos_err);
	}

	static void ParseMat(const std::string& pose_str, int& fid, cv::Matx44f& pose) {
		std::vector<std::string> strs = tk::split(pose_str, ' ');
		if (strs.size() < 9)
			strs = tk::split(pose_str, '\t');

		if (strs.size() < 9)
			CHECK(false) << "ERROR: Invalid pose file";
	
		float m00(0.0f), m01(0.0f), m02(0.0f), m03(0.0f);
		float m10(0.0f), m11(0.0f), m12(0.0f), m13(0.0f);
		float m20(0.0f), m21(0.0f), m22(0.0f), m23(0.0f);
		
		

		if (strs.size() == 13) {
			fid = std::atoi(strs[0].c_str());

			m00 = (float)std::atof(strs[1].c_str());
			m01 = (float)std::atof(strs[2].c_str());
			m02 = (float)std::atof(strs[3].c_str());
			m10 = (float)std::atof(strs[4].c_str());
			m11 = (float)std::atof(strs[5].c_str());
			m12 = (float)std::atof(strs[6].c_str());
			m20 = (float)std::atof(strs[7].c_str());
			m21 = (float)std::atof(strs[8].c_str());
			m22 = (float)std::atof(strs[9].c_str());
			m03 = (float)std::atof(strs[10].c_str());
			m13 = (float)std::atof(strs[11].c_str());
			m23 = (float)std::atof(strs[12].c_str());
		}

		if (strs.size() == 12) {
			m00 = (float)std::atof(strs[0].c_str());
			m01 = (float)std::atof(strs[1].c_str());
			m02 = (float)std::atof(strs[2].c_str());
			m10 = (float)std::atof(strs[3].c_str());
			m11 = (float)std::atof(strs[4].c_str());
			m12 = (float)std::atof(strs[5].c_str());
			m20 = (float)std::atof(strs[6].c_str());
			m21 = (float)std::atof(strs[7].c_str());
			m22 = (float)std::atof(strs[8].c_str());
			m03 = (float)std::atof(strs[9].c_str());
			m13 = (float)std::atof(strs[10].c_str());
			m23 = (float)std::atof(strs[11].c_str());
		}

		cv::Matx44f mat(
			m00, m01, m02, m03,
			m10, m11, m12, m13,
			m20, m21, m22, m23,
			0, 0, 0, 1);

		pose = mat;
	}

	virtual void Read(const std::string& pose_file, std::vector<std::vector<cv::Matx44f> >& poses) override {
		std::ifstream fs;
		fs.open(pose_file);
		if (!fs.is_open()) {
			std::cerr << "failed to open pose file, press any key to exit" << std::endl;
			getchar();
			exit(-1);
		}

		std::string line;

		std::getline(fs, line);
		std::getline(fs, line);
		int obj_size = atoi(line.c_str());

		poses.resize(obj_size);

		while (true) {
			if (fs.eof()) break;

			for (int i = 0; i < obj_size; ++i) {
				std::getline(fs, line);
				if (line.size() < 4) {
					//std::cout << "pose file to short" << std::endl;
					break;
				}

				cv::Matx44f pose;
				int fid;
				ParseMat(line, fid, pose);
				
				poses[i].push_back(pose);
			}
		}
	}

	virtual void Read(const std::string& pose_file, std::vector<int>& fids, std::vector<std::vector<cv::Matx44f> >& poses) override {
		std::ifstream fs;
		fs.open(pose_file);
		if (!fs.is_open()) {
			std::cerr << "failed to open pose file, press any key to exit" << std::endl;
			getchar();
			exit(-1);
		}

		std::string line;

		std::getline(fs, line);
		std::getline(fs, line);
		int obj_size = atoi(line.c_str());

		poses.resize(obj_size);
		fids.clear();

		while (true) {
			if (fs.eof()) break;
			
			int fid;
			for (int i = 0; i < obj_size; ++i) {
				std::getline(fs, line);
				if (line.size() < 4) {
					//std::cout << "pose file to short" << std::endl;
					break;
				}

				cv::Matx44f pose;
				ParseMat(line, fid, pose);

				poses[i].push_back(pose);
			}
			fids.push_back(fid);
		}
	}

	void Read(const std::string& pose_file, std::string& config_file, std::vector<int>& fids, std::vector<cv::Matx44f>& poses) {
		std::ifstream fs;
		fs.open(pose_file);
		if (!fs.is_open()) {
			std::cerr << "failed to open pose file, press any key to exit" << std::endl;
			getchar();
			exit(-1);
		}

		std::string line;
		std::getline(fs, line);

		config_file = line;

		while (true) {
			if (fs.eof()) break;
			std::getline(fs, line);
			if (line.size() < 4) {
				std::cout << "pose file to short" << std::endl;
				break;
			}

			cv::Matx44f pose;
			int fid;
			ParseMat(line, fid, pose);

			fids.push_back(fid);
			poses.push_back(pose);
		}
	}
};

class PoseReaderKarl : public PoseReader {
public:
	virtual bool IsLostGT(const cv::Matx44f& gtm, const cv::Matx44f& tkm, const Object3D* object) override {
		float max_dist = 0.0f;
		for (auto x : object->vertices) {
			cv::Vec4f dx = tkm * cv::Vec4f(x(0), x(1), x(2), 1) - gtm * cv::Vec4f(x(0), x(1), x(2), 1);
			float norm2 = dx(0) * dx(0) + dx(1) * dx(1) + dx(2) * dx(2);
			
			if (norm2 > max_dist)
				max_dist = norm2;
		}

		return max_dist > 100.0f;
	}

	static void expm(float r[3], float rm[9]) {
		cv::Matx33f H(
			0, -r[2], r[1],
			r[2], 0, -r[0],
			-r[1], r[0], 0);

		float angle2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
		float angle = sqrt(angle2);

		if (angle < 1E-14) {
			rm[0] = 1, rm[1] = 0, rm[2] = 0;
			rm[3] = 0, rm[4] = 1, rm[5] = 0;
			rm[6] = 0, rm[7] = 0, rm[8] = 1;
		}	else {
			float ef = sin(angle) / angle;
			float gee = (1.0f - cos(angle)) / angle2;
			cv::Matx33f mr = H * H* gee + H * ef + cv::Matx33f::eye();

			rm[0] = mr(0,0), rm[1] = mr(0,1), rm[2] = mr(0,2);
			rm[3] = mr(1,0), rm[4] = mr(1,1), rm[5] = mr(1,2);
			rm[6] = mr(2,0), rm[7] = mr(2,1), rm[8] = mr(2,2);
		}
	}

	static cv::Matx44f ParseExpVecKarl(const std::string& pose_str) {
		std::vector<std::string> strs = tk::split(pose_str, ' ');

		CHECK(6 == strs.size()) << "ERROR: Invalid pose file";

		float rv[3];
		rv[0] = (float)std::atof(strs[3].c_str());
		rv[1] = (float)std::atof(strs[4].c_str());
		rv[2] = (float)std::atof(strs[5].c_str());

		float rm[9];
		//rvec2rotmat(rv, rm);
		expm(rv, rm);

		float t0 = (float)std::atof(strs[0].c_str());
		float t1 = (float)std::atof(strs[1].c_str());
		float t2 = (float)std::atof(strs[2].c_str());

		cv::Matx44f mat(
			rm[0], rm[1], -rm[2], t0,
			-rm[3], -rm[4], rm[5], -t1,
			rm[6], rm[7], -rm[8], t2,
			0, 0, 0, 1);

		return mat;
	}

	virtual void Read(const std::string& pose_file, std::vector<std::vector<cv::Matx44f> >& poses) override {
		std::ifstream fs;
		fs.open(pose_file);
		if (!fs.is_open()) {
			std::cerr << "failed to open pose file, press any key to exit" << std::endl;
			getchar();
			exit(-1);
		}

		poses.resize(1);

		std::string line;
		while (true) {
			if (fs.eof()) break;
			std::getline(fs, line);
			if (line.size() < 4) {
				std::cout << "pose file to short" << std::endl;
				break;
			}
			poses[0].push_back(ParseExpVecKarl(line));
		}
	}
};

class PoseWriter {
public:
	PoseWriter(std::string file, const std::string& config_file, int objs_size) {
		ofs_pose.open(file);
		CHECK(ofs_pose.is_open()) << "Cannot write the pose";
		ofs_pose << config_file << std::endl;
		ofs_pose << objs_size << std::endl;
	}

	~PoseWriter() {
		ofs_pose.close();
	}

	void Record(cv::Matx44f& m, int fid) {
		ofs_pose << fid << ' '
			<< m(0, 0) << ' ' << m(0, 1) << ' ' << m(0, 2) << ' '
			<< m(1, 0) << ' ' << m(1, 1) << ' ' << m(1, 2) << ' '
			<< m(2, 0) << ' ' << m(2, 1) << ' ' << m(2, 2) << ' '
			<< m(0, 3) << ' ' << m(1, 3) << ' ' << m(2, 3) << std::endl;
	}

	std::ofstream ofs_pose;
};

class LostWriter {
public:
	LostWriter(std::string lost_file) {
		ofs_pose.open(lost_file);
		CHECK(ofs_pose.is_open()) << "Cannot write the pose";
	}

	~LostWriter() {
		ofs_pose.close();
	}

	void Record(int fid) {
		ofs_pose << fid << std::endl;
	}

	std::ofstream ofs_pose;
};

class FpsWriter {
public:
	FpsWriter(std::string file) {
		ofs_pose.open(file);
		CHECK(ofs_pose.is_open()) << "Cannot write the pose";
	}

	~FpsWriter() {
		ofs_pose.close();
	}

	void Record(float fps) {
		ofs_pose << fps << std::endl;
	}

	std::ofstream ofs_pose;
};

class ReportWriter {
public:
	ReportWriter(const std::string config_file, const std::string& report_file) {
		conf_file = config_file;

		ofs_pose.open(report_file, std::ios::app);
		CHECK(ofs_pose.is_open()) << "Cannot write the pose";
	}

	~ReportWriter() {
		ofs_pose.close();
	}

	void Record(float score, float avg_rt) {
		ofs_pose.setf(std::ios_base::fixed, std::ios_base::floatfield);
		ofs_pose.setf(std::ios_base::showpoint);
		ofs_pose.precision(1);
		//sstr << "#" << std::setw(3) << std::setfill('0') << fid << ": " << fps;
		ofs_pose << conf_file << " " << score << " " << avg_rt << std::endl;
	}

	std::ofstream ofs_pose;
	std::string conf_file;
};

}