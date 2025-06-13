#pragma once

#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"

namespace ar3dv
{

	/**
	 * 旋转矩阵R正则化, 并返回正则化后的R
	 * @param R 旋转矩阵
	 */
	cv::Matx33f SVDOrthogonal(const cv::Matx33f &R);

	/**
	 * 正则化位姿矩阵T=[R|t]中的R, 并返回正则化后的位姿T
	 * @param T 位姿矩阵
	 */
	cv::Matx44f SVDOrthogonal(const cv::Matx44f &T);

	/**
	 * 旋转矩阵R转so3并返回
	 * @param R 旋转矩阵
	 */
	cv::Vec3f R2so3(const cv::Matx33f &R);

	/**
	 * so3转旋转矩阵R并返回
	 * @param so3 李代数so3表示下的旋转
	 */
	cv::Matx33f So32R(const cv::Vec3f &so3);

	/**
	 * 位姿矩阵T转se3并返回(旋转在前，平移在后)
	 * @param T 位姿矩阵
	 */
	cv::Vec6f T2se3(const cv::Matx44f &T);

	/**
	 * se3=[t|so3] 转位姿矩阵T并返回
	 * @param se3 李代数se3表示下的位姿, 平移在前, 旋转在后
	 */
	cv::Matx44f Se32T(const cv::Vec6f &se3);

	cv::Vec3f T2so3(const cv::Matx44f &T);

	cv::Matx44f So32T(const cv::Vec3f &so3, const cv::Vec3f &t);

	/**
	 * 旋转矩阵R转四元数并返回
	 * @param R 旋转矩阵
	 */
	Eigen::Quaternionf R2Quaternion(const cv::Matx33f &R);

	/**
	 * 四元数转旋转矩阵R并返回
	 * @param quaternion 四元数
	 */
	cv::Matx33f Quaternion2R(const Eigen::Quaternionf &quaternion);

	/**
	 * @brief 根据T中的R计算四元数并返回
	 * @param T 位姿矩阵
	 */
	Eigen::Quaternionf T2Quaternion(const cv::Matx44f &T);

	/**
	 * @brief 根据四数以及平移向量t计算位姿矩阵T并返回
	 * @param quaternion 表示旋转的四元数
	 * @param t 平移向量
	 */
	cv::Matx44f Quaternion2T(const Eigen::Quaternionf &quaternion, const cv::Vec3f &t);

	/**
	 * 旋转矩阵转欧拉角(zyx)并返回
	 * @param R 旋转矩阵
	 */
	cv::Vec3f R2EulerAngle(const cv::Matx33f &R);

	/**
	 * 欧拉角转旋转矩阵R并返回
	 * @param eulerAngle 欧拉角(zyx)
	 */
	cv::Matx33f EulerAngle2R(const cv::Vec3f &eulerAngle);

	// Just use for format convert of float/double, eigen/cv
	cv::Matx33d Matx33f2d(const cv::Matx33f &mat);
	cv::Matx44d Matx44f2d(const cv::Matx44f &mat);
	cv::Matx33f Matx33d2f(const cv::Matx33d &mat);
	cv::Matx44f Matx44d2f(const cv::Matx44d &mat);

	cv::Matx33f Eigen2cv33f(const Eigen::Matrix3f &matrix);
	cv::Matx33d Eigen2cv33d(const Eigen::Matrix3d &matrix);
	cv::Matx44f Eigen2cv44f(const Eigen::Matrix4f &matrix);
	cv::Matx44d Eigen2cv44d(const Eigen::Matrix4d &matrix);

	Eigen::Matrix3f CV2Eigen33f(const cv::Matx33f &mat);
	Eigen::Matrix3d CV2Eigen33d(const cv::Matx33d &mat);
	Eigen::Matrix4f CV2Eigen44f(const cv::Matx44f &mat);
	Eigen::Matrix4d CV2Eigen44d(const cv::Matx44d &mat);

} // namespace ar3dv