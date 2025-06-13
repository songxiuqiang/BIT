// This file provides conversions between various pose representations and matrix representations,
// and poses are represented by types defined in the Eigen library.

#pragma once

#include <cmath>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Eigen>
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"

namespace cvteigen
{ // converter of eigen

	/**
	 * 旋转矩阵R正则化, 并返回正则化后的R
	 * @param R 旋转矩阵
	 */
	Eigen::Matrix3d SVDOrthogonal(const Eigen::Matrix3d &R);

	/**
	 * 正则化位姿矩阵T=[R|t]中的R, 并返回正则化后的位姿T
	 * @param T 位姿矩阵
	 */
	Eigen::Matrix4d SVDOrthogonal(const Eigen::Matrix4d &T);

	/**
	 * 旋转矩阵R转so3并返回
	 * @param R 旋转矩阵
	 */
	Eigen::Vector3d R2so3(const Eigen::Matrix3d &R);

	/**
	 * so3转旋转矩阵R并返回
	 * @param so3 李代数so3表示下的旋转
	 */
	Eigen::Matrix3d So32R(const Eigen::Vector3d &so3);

	/**
	 * 位姿矩阵T转se3=[t|so3]并返回(平移在前，旋转在后)
	 * @param T 位姿矩阵
	 */
	Eigen::VectorXd T2se3(const Eigen::Matrix4d &T);

	/**
	 * se3=[t|so3] 转位姿矩阵T并返回
	 * @param se3 李代数se3表示下的位姿, 平移在前, 旋转在后
	 */
	Eigen::Matrix4d Se32T(const Eigen::VectorXd &se3);

	/**
	 * 旋转矩阵R转四元数并返回
	 * @param R 旋转矩阵
	 */
	Eigen::Quaterniond R2Quaternion(const Eigen::Matrix3d &R);

	/**
	 * 四元数转旋转矩阵R并返回
	 * @param quaternion 四元数
	 */
	Eigen::Matrix3d Quaternion2R(const Eigen::Quaterniond &quaternion);

	/**
	 * 旋转矩阵转欧拉角(zyx)并返回
	 * @param R 旋转矩阵
	 */
	Eigen::Vector3d R2EulerAngle(const Eigen::Matrix3d &R);

	/**
	 * 欧拉角转旋转矩阵R并返回
	 * @param eulerAngle 欧拉角(zyx)
	 */
	Eigen::Matrix3d EulerAngle2R(const Eigen::Vector3d &eulerAngle);

} // namespace cvteigen