#include "pose_converter.h"

#include "glog/logging.h"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Geometry"
#include <opencv2/core/eigen.hpp>

#include "pose_converter_eigen.h"

using namespace std;
using namespace Eigen;

namespace ar3dv
{

	cv::Matx33f SVDOrthogonal(const cv::Matx33f &R)
	{
		cv::Matx33d Rd = Matx33f2d(R);
		Eigen::Matrix3d Reigen = cvteigen::SVDOrthogonal(CV2Eigen33d(Rd));
		return Matx33d2f(Eigen2cv33d(Reigen));
	}

	cv::Matx44f SVDOrthogonal(const cv::Matx44f &T)
	{
		cv::Matx44d Td = Matx44f2d(T);
		Eigen::Matrix4d Teigen = cvteigen::SVDOrthogonal(CV2Eigen44d(Td));
		return Matx44d2f(Eigen2cv44d(Teigen));
	}

	cv::Vec3f R2so3(const cv::Matx33f &R)
	{
		Eigen::Matrix3d REigen = CV2Eigen33d(Matx33f2d(R));
		Eigen::Vector3d so3 = cvteigen::R2so3(REigen);
		return cv::Vec3f((float)so3[0], (float)so3[1], (float)so3[2]);
	}

	cv::Matx33f So32R(const cv::Vec3f &so3)
	{
		Eigen::Vector3d so3Eigen;
		so3Eigen << (double)so3[0], (double)so3[1], (double)so3[2];
		Eigen::Matrix3d R = cvteigen::So32R(so3Eigen);
		return Matx33d2f(Eigen2cv33d(R));
	}

	cv::Vec6f T2se3(const cv::Matx44f &T)
	{
		Eigen::VectorXd se3;
		se3 = cvteigen::T2se3(CV2Eigen44d(Matx44f2d(T)));
		return cv::Vec6f((float)se3[0], (float)se3[1], (float)se3[2], (float)se3[3], (float)se3[4], (float)se3[5]);
	}

	cv::Matx44f Se32T(const cv::Vec6f &se3)
	{
		Eigen::VectorXd se3Eigen(6);
		se3Eigen << (double)se3[0], (double)se3[1], (double)se3[2], (double)se3[3], (double)se3[4], (double)se3[5];
		cv::Matx44f T = Eigen2cv44d(cvteigen::Se32T(se3Eigen));
		return Matx44d2f(T);
	}

	cv::Vec3f T2so3(const cv::Matx44f &T)
	{
		Eigen::VectorXd se3;
		se3 = cvteigen::T2se3(CV2Eigen44d(Matx44f2d(T)));
		return cv::Vec3f((float)se3[0], (float)se3[1], (float)se3[2]);
	}

	cv::Matx44f So32T(const cv::Vec3f &so3, const cv::Vec3f &t)
	{
		Eigen::Vector3d so3Eigen;
		so3Eigen << (double)so3[0], (double)so3[1], (double)so3[2];
		Eigen::Matrix3d R = cvteigen::So32R(so3Eigen);
		cv::Matx44f T = cv::Matx44f::eye();
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				T(i, j) = (float)R(i, j);
			}
		}
		T(0, 3) = t[0];
		T(1, 3) = t[1];
		T(2, 3) = t[2];
		return T;
	}

	Eigen::Quaternionf R2Quaternion(const cv::Matx33f &R)
	{
		cv::Matx33d Rd = Matx33f2d(R);
		Eigen::Quaterniond qd = cvteigen::R2Quaternion(CV2Eigen33d(Rd));
		Eigen::Quaternionf qf((float)qd.w(), (float)qd.x(), (float)qd.y(), (float)qd.z());
		return qf;
	}

	cv::Matx33f Quaternion2R(const Eigen::Quaternionf &quaternion)
	{
		Eigen::Quaterniond qd((double)quaternion.w(), (double)quaternion.x(), (double)quaternion.y(), (double)quaternion.z());
		Eigen::Matrix3d R = cvteigen::Quaternion2R(qd);
		return Matx33d2f(Eigen2cv33d(R));
	}

	Eigen::Quaternionf T2Quaternion(const cv::Matx44f &T)
	{
		cv::Matx33f R = T.get_minor<3, 3>(0, 0);
		return R2Quaternion(R);
	}

	cv::Matx44f Quaternion2T(const Eigen::Quaternionf &quaternion, const cv::Vec3f &t)
	{
		cv::Matx33f R = Quaternion2R(quaternion);
		cv::Matx44f res{
			R(0, 0), R(0, 1), R(0, 2), t[0],
			R(1, 0), R(1, 1), R(1, 2), t[1],
			R(2, 0), R(2, 1), R(2, 2), t[2],
			0, 0, 0, 1};
		return res;
	}

	cv::Vec3f R2EulerAngle(const cv::Matx33f &R)
	{
		Eigen::Matrix3d REigen = CV2Eigen33d(Matx33f2d(R));
		Eigen::Vector3d eulerAngle = cvteigen::R2EulerAngle(REigen);
		return cv::Vec3f((float)eulerAngle[0], (float)eulerAngle[1], (float)eulerAngle[2]);
	}

	cv::Matx33f EulerAngle2R(const cv::Vec3f &eulerAngle)
	{
		Eigen::Vector3d ea;
		ea << (double)eulerAngle[0], (double)eulerAngle[1], (double)eulerAngle[2];
		Eigen::Matrix3d R = cvteigen::EulerAngle2R(ea);
		return Matx33d2f(Eigen2cv33d(R));
	}

	cv::Matx33f Matx33d2f(const cv::Matx33d &mat)
	{
		cv::Matx33f res;
		for (size_t i = 0; i < mat.rows; ++i)
			for (size_t j = 0; j < mat.rows; ++j)
				res(i, j) = static_cast<float>(mat(i, j));
		return res;
	}

	cv::Matx33d Matx33f2d(const cv::Matx33f &mat)
	{
		cv::Matx33d res;
		for (size_t i = 0; i < mat.rows; ++i)
			for (size_t j = 0; j < mat.rows; ++j)
				res(i, j) = static_cast<double>(mat(i, j));
		return res;
	}

	cv::Matx44f Matx44d2f(const cv::Matx44d &mat)
	{
		cv::Matx44f res;
		for (size_t i = 0; i < mat.rows; ++i)
			for (size_t j = 0; j < mat.rows; ++j)
				res(i, j) = static_cast<float>(mat(i, j));
		return res;
	}

	cv::Matx44d Matx44f2d(const cv::Matx44f &mat)
	{
		cv::Matx44f res;
		for (size_t i = 0; i < mat.rows; ++i)
			for (size_t j = 0; j < mat.rows; ++j)
				res(i, j) = static_cast<double>(mat(i, j));
		return res;
	}

	cv::Matx33f Eigen2cv33f(const Eigen::Matrix3f &matrix)
	{
		cv::Matx33f res{
			matrix(0, 0), matrix(0, 1), matrix(0, 2),
			matrix(1, 0), matrix(1, 1), matrix(1, 2),
			matrix(2, 0), matrix(2, 1), matrix(2, 2)};
		return res;
	}

	cv::Matx33d Eigen2cv33d(const Eigen::Matrix3d &matrix)
	{
		cv::Matx33d res{
			matrix(0, 0), matrix(0, 1), matrix(0, 2),
			matrix(1, 0), matrix(1, 1), matrix(1, 2),
			matrix(2, 0), matrix(2, 1), matrix(2, 2)};
		return res;
	}

	cv::Matx44f Eigen2cv44f(const Eigen::Matrix4f &matrix)
	{
		cv::Matx44f res{
			matrix(0, 0), matrix(0, 1), matrix(0, 2), matrix(0, 3),
			matrix(1, 0), matrix(1, 1), matrix(1, 2), matrix(1, 3),
			matrix(2, 0), matrix(2, 1), matrix(2, 2), matrix(2, 3),
			matrix(3, 0), matrix(3, 1), matrix(3, 2), matrix(3, 3)};
		return res;
	}

	cv::Matx44d Eigen2cv44d(const Eigen::Matrix4d &matrix)
	{
		cv::Matx44d res{
			matrix(0, 0), matrix(0, 1), matrix(0, 2), matrix(0, 3),
			matrix(1, 0), matrix(1, 1), matrix(1, 2), matrix(1, 3),
			matrix(2, 0), matrix(2, 1), matrix(2, 2), matrix(2, 3),
			matrix(3, 0), matrix(3, 1), matrix(3, 2), matrix(3, 3)};
		return res;
	}

	Eigen::Matrix3f CV2Eigen33f(const cv::Matx33f &mat)
	{
		Eigen::Matrix3f res;
		res << mat(0, 0), mat(0, 1), mat(0, 2),
			mat(1, 0), mat(1, 1), mat(1, 2),
			mat(2, 0), mat(2, 1), mat(2, 2);
		return res;
	}

	Eigen::Matrix3d CV2Eigen33d(const cv::Matx33d &mat)
	{
		Eigen::Matrix3d res;
		res << mat(0, 0), mat(0, 1), mat(0, 2),
			mat(1, 0), mat(1, 1), mat(1, 2),
			mat(2, 0), mat(2, 1), mat(2, 2);
		return res;
	}

	Eigen::Matrix4f CV2Eigen44f(const cv::Matx44f &mat)
	{
		Eigen::Matrix4f res;
		res << mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3),
			mat(1, 0), mat(1, 1), mat(1, 2), mat(1, 3),
			mat(2, 0), mat(2, 1), mat(2, 2), mat(2, 3),
			mat(3, 0), mat(3, 1), mat(3, 2), mat(3, 3);
		return res;
	}

	Eigen::Matrix4d CV2Eigen44d(const cv::Matx44d &mat)
	{
		Eigen::Matrix4d res;
		res << mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3),
			mat(1, 0), mat(1, 1), mat(1, 2), mat(1, 3),
			mat(2, 0), mat(2, 1), mat(2, 2), mat(2, 3),
			mat(3, 0), mat(3, 1), mat(3, 2), mat(3, 3);
		return res;
	}
}
