#include "pose_converter_eigen.h"

Eigen::Matrix3d cvteigen::SVDOrthogonal(const Eigen::Matrix3d &R)
{
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(R, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Matrix3d V = svd.matrixV(), U = svd.matrixU();
	Eigen::Matrix3d RO = U * V.transpose();
	return RO;
}

Eigen::Matrix4d cvteigen::SVDOrthogonal(const Eigen::Matrix4d &T)
{
	Eigen::Matrix4d TO(T);
	Eigen::Matrix3d R = T.block<3, 3>(0, 0);
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(R, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Matrix3d V = svd.matrixV(), U = svd.matrixU();
	Eigen::Matrix3d RO = U * V.transpose();
	TO.block<3, 3>(0, 0) = RO;
	return TO;
}

Eigen::Vector3d cvteigen::R2so3(const Eigen::Matrix3d &R)
{
	Eigen::Matrix3d RO = SVDOrthogonal(R);
	Sophus::SO3<double> SO3_R(RO);
	Eigen::Vector3d so3 = SO3_R.log();
	return so3;
}

Eigen::Matrix3d cvteigen::So32R(const Eigen::Vector3d &so3)
{
	Eigen::AngleAxisd so3Tmp(so3.norm(), so3.normalized());
	Eigen::Matrix3d rotationMatrix = Eigen::Matrix3d::Identity();
	rotationMatrix = so3Tmp.toRotationMatrix();
	return rotationMatrix;
}

Eigen::VectorXd cvteigen::T2se3(const Eigen::Matrix4d &T)
{
	Eigen::Matrix4d TO(T);
	Eigen::Matrix3d R = T.block<3, 3>(0, 0);
	R = cvteigen::SVDOrthogonal(R);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			TO(i, j) = R(i, j);

	Sophus::SE3d SE3(TO);
	Eigen::VectorXd se3 = SE3.log();
	Eigen::VectorXd se3In_rt_Format(6);
	se3In_rt_Format << se3[3], se3[4], se3[5], se3[0], se3[1], se3[2];
	return se3In_rt_Format;
}

Eigen::Matrix4d cvteigen::Se32T(const Eigen::VectorXd &se3)
{
	Eigen::VectorXd se3In_tr_Format(6);
	se3In_tr_Format << se3[3], se3[4], se3[5], se3[0], se3[1], se3[2];
	Sophus::SE3d SE3 = Sophus::SE3d::exp(se3In_tr_Format);
	return SE3.matrix();
}

Eigen::Quaterniond cvteigen::R2Quaternion(const Eigen::Matrix3d &R)
{
	return Eigen::Quaterniond(R);
}

Eigen::Matrix3d cvteigen::Quaternion2R(const Eigen::Quaterniond &quaternion)
{
	quaternion.normalized();
	Eigen::Matrix3d R = quaternion.toRotationMatrix();
	return R;
}

Eigen::Vector3d cvteigen::R2EulerAngle(const Eigen::Matrix3d &R)
{
	Eigen::Vector3d eulerAngle = R.eulerAngles(0, 1, 2);
	return eulerAngle;
}

Eigen::Matrix3d cvteigen::EulerAngle2R(const Eigen::Vector3d &eulerAngle)
{
	Eigen::Matrix3d rotationMatrix;
	rotationMatrix = Eigen::AngleAxisd(eulerAngle[0], Eigen::Vector3d::UnitX()) *
					 Eigen::AngleAxisd(eulerAngle[1], Eigen::Vector3d::UnitY()) *
					 Eigen::AngleAxisd(eulerAngle[2], Eigen::Vector3d::UnitZ());

	return rotationMatrix;
}