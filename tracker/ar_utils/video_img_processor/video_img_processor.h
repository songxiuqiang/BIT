#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <random>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "gflags/gflags.h"

#include "../data_io/data_loader.h"

using namespace std;

namespace ar3dv
{

	/**
	 * 拍摄视频并存储
	 * 自动模式：按a开始/停止拍摄
	 * 手动模式：按s拍摄一帧
	 * @param savePath 视频帧存储路径
	 * @param camID 相机id
	 * @param autoCapture falue=手动拍摄，true=自动拍摄
	 */
	void CaptureImgs(const std::string &savePath, const int &camID = 0, const bool &autoCapture = false);

	cv::Mat ComCanny(const cv::Mat &img, const cv::Rect &roi, const double &thresh1 = 50, const double &thresh2 = 100);

	void ComCannyInFiles(const std::vector<std::string> &input, const std::string &output, const std::string &roiPath,
						 const double &thresh1 = 50, const double &thresh2 = 100);

	/**
	 * 将一张图片复制为一个序列
	 * @param frame 需要复制的图片
	 * @param numCopies 需复制的数量
	 * @param sequenceSavePath 序列存储路径
	 */
	void CopyOneImg2Sequence(const cv::Mat &frame, const int &numCopies, const std::string &sequenceSavePath);

	cv::Mat CutoffImg(const cv::Mat &img, const int &left, const int &right, const int &up, const int &down);

	void CutoffImgsInFile(const std::string &imgsFilePath, const std::string &savePath,
						  const int &left, const int &right, const int &up, const int &down);

	/**
	 * @brief 将图像向外扩充，用白色填补
	 * @param img
	 * @param distance
	 * @param direction (x,y) 取值为{-1,0,1} -1表示沿x/y负方向(即左/上)进行扩展，+1反之，0表示不再此方向上扩展.
	 * @return cv::Mat
	 */
	cv::Mat ExtendImg(const cv::Mat &img, const int &distance, int *direction);

	/**
	 * @brief 将图像向外扩充，用白色填补
	 * @param imgsFilePath 图片根路径
	 * @param distance 向外扩展的像素
	 * @param direction (x,y) 取值为{-1,0,1} -1表示沿x/y负方向(即左/上)进行扩展，+1反之，0表示不再此方向上扩展.
	 * @param savePath 扩展后的图像的存储路径
	 */
	void ExtendImgsInFile(const std::string &imgsFilePath, const int &distance, int *direction, const std::string &savePath);

	std::vector<int> GenRandomIndex(const int &num, const int &benginIndex, const int &endIndex);

	/**
	 * @brief 获得文件夹下指定后缀的图片文件
	 * @param rootPath 文件夹路径，以/结尾
	 * @param suffix 图片格式，默认为".png"
	 * @return std::vector<cv::String> 所有图片的路径
	 */
	std::vector<cv::String> GetImagesPaths(const std::string &rootPath, const std::string &suffix = ".*");

	/**
	 * @brief  视频离散为图片序列
	 * @param videoInput	视频路径
	 * @param imgsOutput	图像保存的文件夹路径
	 */
	void Video2Imgs(const std::string &videoInput, const std::string &imgsOutput);

	/**
	 * @brief 图片序列合成视频
	 * @param imgsInput		图片的文件夹路径
	 * @param videoOutput 	视频保存路径
	 */
	void Imgs2Video(const std::string &imgsInput, const std::string &videoOutput, const std::string &suffix = ".png", const double &fps = 60);

	/**
	 * @brief 多张图片拼接成一张更高的图片
	 * @param imgs 输入的图片vector
	 * @return cv::Mat 拼接的图片
	 */
	cv::Mat MutiImgs2HigherImg(const std::vector<cv::Mat> &imgs);

	/**
	 * @brief 多张图片拼接成一张更宽的图片
	 * @param imgs 输入的图片vector
	 * @return cv::Mat 拼接的图片
	 */
	cv::Mat MutiImgs2WiderImg(const std::vector<cv::Mat> &imgs);

	void MultiImgSequence2HigherVideo(const std::vector<std::string> &imgSequences,
									  const std::string &videoOutput,
									  const float &scale = 1);

	/**
	 * @brief 多个图片序列合成为一个宽视频
	 * @author sonxgiuqiang
	 * @param imgSequences 图片序列文件夹目录
	 * @param videoOutput 视频保存路径
	 * @param scale 视频的缩放比例
	 * @param suffix 图片类型/后缀，默认选取所有类型/后缀的图片
	 * @param truncationIdx 截取序列的前truncationIdx帧进行视频合成，默认为INT_MAX
	 */
	void MultiImgSequence2WiderVideo(const std::vector<std::string> &imgSequences,
									 const std::string &videoOutput,
									 const float &scale = 1,
									 const std::string &suffix = ".*",
									 const int &truncationIdx = INT_MAX);

	/**
	 * @brief 多个图片序列合成为一个宽序列
	 * @author sonxgiuqiang
	 * @param imgSequences 图片序列文件夹目录
	 * @param imgsOutput 结果保存目录
	 * @param scale 图片的缩放比例
	 * @param suffix 图片类型/后缀，默认选取所有类型/后缀的图片
	 * @param truncationIdx 截取序列的前truncationIdx帧，默认为INT_MAX
	 */
	void MultiImgSequence2WiderImgs(const std::vector<std::string> &imgSequences,
									const std::string &imgsOutput,
									const float &scale = 1,
									const std::string &suffix = ".*",
									const int &truncationIdx = INT_MAX);
	/**
	 * @brief 多个图片序列合成为一个高序列
	 * @author sonxgiuqiang
	 * @param imgSequences 图片序列文件夹目录
	 * @param imgsOutput 结果保存目录
	 * @param scale 图片的缩放比例
	 * @param suffix 图片类型/后缀，默认选取所有类型/后缀的图片
	 * @param truncationIdx 截取序列的前truncationIdx帧，默认为INT_MAX
	 */
	void MultiImgSequence2HigherImgs(const std::vector<std::string> &imgSequences,
									 const std::string &imgsOutput,
									 const float &scale = 1,
									 const std::string &suffix = ".*",
									 const int &truncationIdx = INT_MAX);

	/**
	 * 多个出片序列合成为长视频
	 * @param imgSequences
	 * @param videoOutput
	 */
	void MultiImgsSequence2LongVideo(const std::vector<std::string> &imgSequences,
									 const std::string &videoOutput);

	void PutTextOnVideo(const std::string &inputVideo, const std::string &outputVideo);

	/**
	 * 批量resize图片
	 * @param inputRootPath
	 * @param outputRootPath
	 * @param size
	 * @param suffix
	 */
	void ImgsBatchResize(const std::string &inputRootPath,
						 const std::string &outputRootPath,
						 const cv::Size &size,
						 const std::string &suffix = ".png");

	void ImgsSlice(const std::string &imgsRoot, const std::string &roiPath, const std::string &saveRoot,
				   const std::string &prefix, const std::string &suffix = ".png", const bool &zeroPadding = false);

	/**
	 * 根据roi裁剪图片
	 * @param img
	 * @param roi
	 * @return 裁剪后的图片
	 */
	cv::Mat ImgSlice(const cv::Mat &img, const cv::Rect &roi);

	/**
	 * 数字补零
	 * @param index
	 * @param number 补零的位数，例如4位 0006
	 * @return 补零后的index
	 */
	std::string ZeroPaddingIndex(const int &index, const int &number);

	void ImgsRecover(const std::string &roiImgsRoot, const std::string &roiPath,
					 const std::string &saveRoot,
					 const std::string &bgImgsRoot,
					 const std::string &prefix, const std::string &suffix = ".png",
					 const bool &zeroPadding = true);

	cv::Mat ImgRecover(const cv::Mat &roiImg, const cv::Rect &roi, const cv::Mat &bg, const bool &useBg);

	/**
	 * 图像增强
	 * @param img
	 * @return 增强后的图像
	 */
	cv::Mat ImgEnhance(cv::Mat img);

	cv::Mat ImgDeleteBg(cv::Mat img, cv::Scalar scalar, int thresh = 100);

} // namespace ar3dv