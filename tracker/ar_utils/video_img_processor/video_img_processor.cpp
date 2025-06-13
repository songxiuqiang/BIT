#include "video_img_processor.h"

using namespace std;
namespace ar3dv
{

	void CaptureImgs(const std::string &savePath, const int &camID, const bool &autoCapture)
	{
		cv::VideoCapture vc;
		vc.open(camID);
		if (!vc.isOpened())
		{
			VLOG(0) << "Can not open camera " << camID;
			return;
		}

		cv::Mat frame;
		int key = 0;
		int index = 0;
		int saveIndex = 0;
		bool startCap = false;
		while (1)
		{
			vc >> frame;
			cv::imshow("frame", frame);
			key = cv::waitKey(1);

			if (autoCapture && startCap)
			{
				cv::imwrite(savePath + std::to_string(saveIndex) + ".png", frame);
				saveIndex++;
				VLOG(0) << "save " << saveIndex;
			}

			if (key == 's')
			{
				cv::imwrite(savePath + std::to_string(saveIndex) + ".png", frame);
				saveIndex++;
				VLOG(0) << "save " << index;
			}

			if (key == 'a')
				startCap = !startCap;

			if (key == 'q')
			{
				vc.release();
				break;
			}
		}
	}

	cv::Mat ComCanny(const cv::Mat &img, const cv::Rect &roi, const double &thresh1, const double &thresh2)
	{
		cv::Mat res = cv::Mat::zeros(img.size(), CV_8UC1);
		cv::Mat imgRoi = img(roi);
		cv::Mat gray, edge;
		cv::cvtColor(imgRoi, gray, cv::COLOR_BGR2GRAY);
		cv::blur(gray, gray, cv::Size(3, 3));
		cv::Canny(gray, edge, thresh1, thresh2);
		edge.copyTo(res(roi));
		return res;
	}

	void ComCannyInFiles(const std::vector<std::string> &input, const std::string &output, const std::string &roiPath,
						 const double &thresh1, const double &thresh2)
	{
		std::vector<std::vector<float>> roiInfo = Loadtxt<float>(roiPath, 0, ' ');
		std::vector<cv::Rect> rois;
		for (auto &info : roiInfo)
		{
			cv::Rect roi(info[0], info[1], info[2], info[3]);
			rois.push_back(roi);
		}

		for (size_t index = 0; index < rois.size(); index++)
		{
			cv::Mat img = cv::imread(input[index]);
			CHECK(!img.empty()) << "img empty in path " << input[index];
			cv::Mat cannyImg = ComCanny(img, rois[index], thresh1, thresh2);
			std::string savePath = output + std::to_string(index) + ".png";
			cv::imwrite(savePath, cannyImg);
		}
	}

	void CopyOneImg2Sequence(const cv::Mat &frame, const int &numCopies, const std::string &sequenceSavePath)
	{
		for (int i = 0; i < numCopies; i++)
		{
			std::string imgSavePath = sequenceSavePath + std::to_string(i) + ".png";
			cv::imwrite(imgSavePath, frame);
		}
	}

	cv::Mat ExtendImg(const cv::Mat &img, const int &distance, int *direction)
	{
		int width = img.cols;
		int height = img.rows;

		// 扩展后的长宽
		int eWidth = width + std::abs(direction[0]) * distance;
		int eHeight = height + std::abs(direction[1]) * distance;

		cv::Mat dst(eHeight, eWidth, img.type(), cv::Scalar(255, 255, 255));

		int up = 0, left = 0;
		if (direction[0] == -1)
			left += distance;
		if (direction[1] == -1)
			up += distance;

		cv::Rect roi(left, up, width, height);
		img.copyTo(dst(roi));
		// VLOG(0) << "dst.cols " << dst.cols;
		// cv::imshow("dst",dst);
		// cv::waitKey(0);

		return dst;
	}

	cv::Mat CutoffImg(const cv::Mat &img, const int &left, const int &right, const int &up, const int &down)
	{

		cv::Mat dst = img(cv::Range(up, down), cv::Range(left, right));
		return dst;
	}

	void ExtendImgsInFile(const std::string &imgsFilePath, const int &distance, int *direction, const std::string &savePath)
	{
		cv::Mat frame;
		cv::Mat dst;
		for (int i = 0;; i++)
		{
			std::string imgPath = imgsFilePath + std::to_string(i) + ".png";
			frame = cv::imread(imgPath);
			if (frame.empty())
			{
				VLOG(0) << "Extend imgs numbers: " << i;
				break;
			}
			dst = ar3dv::ExtendImg(frame, distance, direction);
			VLOG(0) << "img index " << i;
			std::string imgSavePath = savePath + std::to_string(i) + ".png";
			cv::imwrite(imgSavePath, dst);
		}
	}

	void CutoffImgsInFile(const std::string &imgsFilePath, const std::string &savePath,
						  const int &left, const int &right, const int &up, const int &down)
	{
		cv::Mat frame;
		cv::Mat dst;
		for (int i = 0;; i++)
		{
			std::string imgPath = imgsFilePath + std::to_string(i) + ".png";
			// std::string imgPath = imgsFilePath + ZeroPadding(i,4) + ".png";
			// std::string imgPath = imgsFilePath + std::to_string(i) + ".jpg";
			frame = cv::imread(imgPath);
			if (frame.empty())
			{
				VLOG(0) << "Extend imgs numbers: " << i;
				break;
			}
			dst = ar3dv::CutoffImg(frame, left, right, up, down);
			VLOG(0) << "img index " << i;
			std::string imgSavePath = savePath + std::to_string(i) + ".png";
			cv::imwrite(imgSavePath, dst);
		}
	}

	std::vector<int> GenRandomIndex(const int &num, const int &benginIndex, const int &endIndex)
	{
		// 设置随机数生成器
		std::random_device rd;
		std::mt19937 gen(rd());

		// 生成范围为[0, m]的随机整数
		std::uniform_int_distribution<int> dis(benginIndex, endIndex);

		// 生成n个随机整数
		std::vector<int> randomIntegers;
		while (randomIntegers.size() < num)
		{
			int randomNum = dis(gen);
			if (std::find(randomIntegers.begin(), randomIntegers.end(), randomNum) == randomIntegers.end())
			{
				randomIntegers.push_back(randomNum);
			}
		}

		return randomIntegers;
	}

	std::vector<cv::String> GetImagesPaths(const std::string &rootPath, const std::string &suffix)
	{
		std::string imgs = rootPath + "*" + suffix;
		std::vector<cv::String> imgPaths;
		cv::glob(imgs, imgPaths);
		if (imgPaths.size() == 0)
		{
			LOG(WARNING) << "Empty imgs in file " << rootPath << " with suffix \"" << suffix << "\"";
			return imgPaths;
		}
		std::sort(imgPaths.begin(), imgPaths.end());
		return imgPaths;
	}

	void Video2Imgs(const std::string &videoInput, const std::string &imgsOutput)
	{
		cv::VideoCapture cap;
		cap.open(videoInput);
		if (!cap.isOpened())
		{
			LOG(WARNING) << videoInput << " is not openped.";
		}

		cv::Mat frame;
		int frmIdx = 0;
		while (1)
		{
			cap >> frame;
			if (frame.empty())
			{
				break;
			}

			// cv::resize(frame,frame,frame.size()/2);
			cv::imshow("frame", frame);
			cv::imwrite(imgsOutput + std::to_string(frmIdx) + ".png", frame);
			cv::waitKey(1);

			frmIdx++;
		}
		cap.release();
	}

	void Imgs2Video(const std::string &imgsInput, const std::string &videoOutput, const std::string &suffix, const double &fps)
	{
		std::vector<cv::String> image_files = ar3dv::GetImagesPaths(imgsInput, suffix);
		// std::string pattern_png = imgsInput + "*" + suffix;
		// std::vector<cv::String> image_files;
		// cv::glob(pattern_png, image_files);

		int frmIdx = 0;
		std::string firstImgPath = imgsInput + std::to_string(frmIdx) + suffix;
		// std::string firstImgPath = image_files[0];
		cv::Mat frame = cv::imread(firstImgPath);
		VLOG(0) << " firstFramePath " << firstImgPath;
		cv::VideoWriter writer;
		// CV_FOURCC('M', 'P', 'E', 'G'),  //编码格式 MPEG-4

		writer.open(
			videoOutput,								 // 输出文件名
			cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), // 编码格式 MPEG-4
			fps,										 // 帧率（FPS），double型
			cv::Size(frame.cols, frame.rows),			 // 单帧分辨率
			true										 // 只输入彩色图
		);

		while (true)
		{
			// if(frmIdx == 1700) frmIdx = 1560;
			std::string frmPath = imgsInput + std::to_string(frmIdx) + suffix;
			// std::string frmPath = image_files[frmIdx];
			frame = cv::imread(frmPath);
			if (frame.empty())
				break;
			cv::imshow("frame", frame);
			cv::waitKey(1);
			writer << frame;
			VLOG(0) << "frmIdx " << frmIdx;
			frmIdx++;
			// frmIdx--;
			// if(frmIdx==1000) break;
		}

		writer.release();
	}

	cv::Mat MutiImgs2HigherImg(const std::vector<cv::Mat> &imgs)
	{
		int w = 0, h = 0;
		for (int i = 0; i < imgs.size(); i++)
		{
			h += imgs[i].rows;
			w = w > imgs[i].cols ? w : imgs[i].cols;
		}

		int up = 0;
		cv::Mat dst(h, w, imgs[0].type()), frame;
		for (int i = 0; i < imgs.size(); i++)
		{
			cv::Rect roi = cv::Rect(0, up, imgs[i].cols, imgs[i].rows);
			up += imgs[i].rows;
			imgs[i].copyTo(dst(roi));
		}
		// cv::imshow("dst",dst);
		// cv::waitKey(0);
		return dst;
	}

	cv::Mat MutiImgs2WiderImg(const std::vector<cv::Mat> &imgs)
	{
		int w = 0, h = 0;
		for (int i = 0; i < imgs.size(); i++)
		{
			w += imgs[i].cols;
			h = h > imgs[i].rows ? h : imgs[i].rows;
		}

		int left = 0;
		cv::Mat dst(h, w, imgs[0].type()), frame;
		for (int i = 0; i < imgs.size(); i++)
		{
			cv::Rect roi = cv::Rect(left, 0, imgs[i].cols, imgs[i].rows);
			left += imgs[i].cols;
			imgs[i].copyTo(dst(roi));
			// if(i==300) break;
		}
		return dst;
	}

	void MultiImgsSequence2LongVideo(const std::vector<std::string> &imgSequences,
									 const std::string &videoOutput)
	{
		int numImgSequences = imgSequences.size();
		int h = 0, w = 0; // 要合成的视频的长宽

		std::string firstFramePath = imgSequences[0] + "0.png";
		cv::Mat firstFrame = cv::imread(firstFramePath);
		h = firstFrame.rows;
		w = firstFrame.cols;
		VLOG(0) << "Video size " << w << " " << h;

		for (size_t i = 1; i < imgSequences.size(); i++)
		{
			std::string firstImgPath = imgSequences[i] + "0.png";
			cv::Mat firstImg = cv::imread(firstImgPath);
			if (h != firstImg.rows || w != firstImg.cols)
			{
				VLOG(0) << "Img size not equal, sequence: " << i;
				return;
			}
		}

		cv::VideoWriter writer;
		writer.open(
			videoOutput,
			cv::VideoWriter::fourcc('M', 'P', 'E', 'G'),
			30.0,			// 帧率（FPS），double型
			cv::Size(w, h), // 单帧分辨率
			true			// 只输入彩色图
		);

		int frmIdx = 0;
		cv::Mat frame;
		std::vector<cv::Mat> imgs;

		for (size_t i = 0; i < imgSequences.size(); i++)
		{
			for (size_t index = 0; index < 10000; index++)
			{
				std::string ip = imgSequences[i] + std::to_string(index) + ".png";
				frame = cv::imread(ip);
				if (frame.empty())
				{
					VLOG(0) << "frame empty in index " << index;
					break;
				}
				cv::imshow("frame", frame);
				cv::waitKey(1);
				writer << frame;
			}
		}
		writer.release();
	}

	void MultiImgSequence2HigherVideo(const std::vector<std::string> &imgSequences, const std::string &videoOutput, const float &scale)
	{
		int numImgSequences = imgSequences.size();
		int h = 0, w = 0; // 要合成的视频的长宽

		// 获取视频的长宽
		for (int i = 0; i < numImgSequences; i++)
		{
			std::string firstFramePath = imgSequences[i] + "0.png";
			cv::Mat firstFrame = cv::imread(firstFramePath);
			h += firstFrame.rows;
			w = w > firstFrame.cols ? w : firstFrame.cols;
		}
		w *= scale;
		h *= scale;
		cv::Size scaledSize(w, h);

		VLOG(0) << "w h " << w << " " << h;
		cv::VideoWriter writer;
		writer.open(
			videoOutput,
			cv::VideoWriter::fourcc('M', 'P', 'E', 'G'),
			25.0,			// 帧率（FPS），double型
			cv::Size(w, h), // 单帧分辨率
			true			// 只输入彩色图
		);

		int frmIdx = 0;
		cv::Mat frame;
		std::vector<cv::Mat> imgs;

		while (true)
		{
			for (int i = 0; i < numImgSequences; i++)
			{
				std::string frmPath = imgSequences[i] + std::to_string(frmIdx) + ".png";
				frame = cv::imread(frmPath);
				if (frame.empty())
					break;
				imgs.push_back(frame);
			}
			if (imgs.size() == 0)
				break;
			cv::Mat widerImg = ar3dv::MutiImgs2HigherImg(imgs);
			if (scale != 1)
			{
				cv::resize(widerImg, widerImg, scaledSize);
			}
			cv::imshow("wider img", widerImg);
			cv::waitKey(1);
			writer << widerImg;
			imgs.clear();
			VLOG(0) << "frmIdx " << frmIdx;
			frmIdx++;
			// if(frmIdx== 200) break;
		}

		writer.release();
	}

	void MultiImgSequence2WiderVideo(const std::vector<std::string> &imgSequences,
									 const std::string &videoOutput,
									 const float &scale,
									 const std::string &suffix,
									 const int &truncationIdx)
	{
		int numImgSequences = imgSequences.size();
		int h = 0, w = 0; // 要合成的视频的长宽

		// vector<vector<cv::String> > imgSequencesFiles;
		// for(int i = 0; i < numImgSequences; i++)
		// {
		// vector<cv::String> imgsFiles = GetImagesPaths(imgSequences[i],suffix);
		// imgSequencesFiles.push_back(imgsFiles);
		// }

		// 获取视频的长宽,为各个视频之和
		for (int i = 0; i < numImgSequences; i++)
		{
			std::string firstFramePath = imgSequences[i] + "0.png";
			cv::Mat firstFrame = cv::imread(firstFramePath);
			w += firstFrame.cols;
			h = h > firstFrame.rows ? h : firstFrame.rows;
		}
		w *= scale;
		h *= scale;
		cv::Size scaledSize(w, h);

		VLOG(0) << "w h " << w << " " << h;
		cv::VideoWriter writer;
		writer.open(
			videoOutput,
			cv::VideoWriter::fourcc('M', 'P', 'E', 'G'),
			25.0,			// 帧率（FPS），double型
			cv::Size(w, h), // 单帧分辨率
			true			// 只输入彩色图
		);

		int frmIdx = 0;
		cv::Mat frame;
		std::vector<cv::Mat> imgs;

		while (true)
		{
			for (int i = 0; i < numImgSequences; i++)
			{
				// std::string frmPath = imgSequencesFiles[i][frmIdx];
				std::string frmPath = imgSequences[i] + std::to_string(frmIdx) + ".png";
				frame = cv::imread(frmPath);
				if (frame.empty())
					break;
				imgs.push_back(frame);
			}
			if (imgs.size() == 0)
				break;
			cv::Mat widerImg = ar3dv::MutiImgs2WiderImg(imgs);
			if (scale != 1)
			{
				cv::resize(widerImg, widerImg, scaledSize);
			}
			cv::imshow("wider img", widerImg);
			cv::waitKey(1);
			writer << widerImg;
			imgs.clear();
			VLOG(0) << "frmIdx " << frmIdx;
			frmIdx++;
			// if(frmIdx < 150) continue;
			// if(frmIdx == truncationIdx) break;
		}

		writer.release();
	}

	void MultiImgSequence2WiderImgs(const std::vector<std::string> &imgSequences,
									const std::string &imgsOutput,
									const float &scale,
									const std::string &suffix,
									const int &truncationIdx)
	{
		int numImgSequences = imgSequences.size();
		int h = 0, w = 0; // 要合成的视频的长宽

		// 获取视频的长宽,为各个视频之和
		for (int i = 0; i < numImgSequences; i++)
		{
			std::string firstFramePath = imgSequences[i] + "0.png";
			cv::Mat firstFrame = cv::imread(firstFramePath);
			w += firstFrame.cols;
			h = h > firstFrame.rows ? h : firstFrame.rows;
		}
		w *= scale;
		h *= scale;
		cv::Size scaledSize(w, h);
		VLOG(0) << "w h " << w << " " << h;

		int frmIdx = 0;
		cv::Mat frame;
		std::vector<cv::Mat> imgs;

		while (true)
		{
			for (int i = 0; i < numImgSequences; i++)
			{
				// std::string frmPath = imgSequencesFiles[i][frmIdx];
				std::string frmPath = imgSequences[i] + std::to_string(frmIdx) + ".png";
				frame = cv::imread(frmPath);
				if (frame.empty())
					break;
				imgs.push_back(frame);
			}
			if (imgs.size() == 0)
				break;
			cv::Mat widerImg = ar3dv::MutiImgs2WiderImg(imgs);
			if (scale != 1)
			{
				cv::resize(widerImg, widerImg, scaledSize);
			}
			cv::imshow("wider img", widerImg);
			cv::waitKey(1);

			std::string savePath = imgsOutput + std::to_string(frmIdx) + ".png";
			cv::imwrite(savePath, widerImg);
			imgs.clear();
			VLOG(0) << "frmIdx " << frmIdx;
			frmIdx++;
			if (frmIdx == truncationIdx)
				break;
		}
	}

	void MultiImgSequence2HigherImgs(const std::vector<std::string> &imgSequences,
									 const std::string &imgsOutput,
									 const float &scale,
									 const std::string &suffix,
									 const int &truncationIdx)
	{
		int numImgSequences = imgSequences.size();
		int h = 0, w = 0; // 要合成的视频的长宽

		// 获取视频的长宽,为各个视频之和
		for (int i = 0; i < numImgSequences; i++)
		{
			std::string firstFramePath = imgSequences[i] + "0.png";
			cv::Mat firstFrame = cv::imread(firstFramePath);
			h += firstFrame.rows;
			w = w > firstFrame.cols ? w : firstFrame.cols;
		}
		w *= scale;
		h *= scale;
		cv::Size scaledSize(w, h);

		int frmIdx = 0;
		cv::Mat frame;
		std::vector<cv::Mat> imgs;

		while (true)
		{
			for (int i = 0; i < numImgSequences; i++)
			{
				// std::string frmPath = imgSequencesFiles[i][frmIdx];
				std::string frmPath = imgSequences[i] + std::to_string(frmIdx) + ".png";
				frame = cv::imread(frmPath);
				if (frame.empty())
					break;
				imgs.push_back(frame);
			}
			if (imgs.size() == 0)
				break;
			cv::Mat widerImg = ar3dv::MutiImgs2HigherImg(imgs);
			if (scale != 1)
			{
				cv::resize(widerImg, widerImg, scaledSize);
			}
			cv::imshow("wider img", widerImg);
			cv::waitKey(1);

			std::string savePath = imgsOutput + std::to_string(frmIdx) + ".png";
			cv::imwrite(savePath, widerImg);
			imgs.clear();
			VLOG(0) << "frmIdx " << frmIdx;
			frmIdx++;
			if (frmIdx == truncationIdx)
				break;
		}
	}

	void PutTextOnVideo(const std::string &inputVideo, const std::string &outputVideo)
	{
		cv::VideoCapture vc(inputVideo);
		if (!vc.isOpened())
			LOG(ERROR) << " Can not open video " << inputVideo;

		double fps = vc.get(cv::CAP_PROP_FPS);
	}

	void ImgsBatchResize(const std::string &inputRootPath,
						 const std::string &outputRootPath,
						 const cv::Size &size,
						 const std::string &suffix)
	{
		std::vector<cv::String> imgsPaths = ar3dv::GetImagesPaths(inputRootPath, suffix);
		for (int i = 0; i < imgsPaths.size(); i++)
		{
			std::string path = inputRootPath + std::to_string(i) + suffix;
			// cv::Mat img = cv::imread(imgsPaths[i]);
			cv::Mat img = cv::imread(path);
			cv::resize(img, img, size);

			std::string savePath = outputRootPath + std::to_string(i) + ".png";
			cv::imwrite(savePath, img);
		}
	}

	void ImgsSlice(const std::string &imgsRoot, const std::string &roiPath, const std::string &saveRoot,
				   const std::string &prefix, const std::string &suffix, const bool &zeroPadding)
	{
		int index, x, y, width, height;
		std::ifstream ifs(roiPath);

		std::vector<cv::String> ips = ar3dv::GetImagesPaths(imgsRoot, suffix);

		for (int i = 0; i < ips.size(); i++)
		{
			std::string imgPath;
			if (zeroPadding)
				imgPath = imgsRoot + prefix + ZeroPaddingIndex(i, 4) + suffix;
			else
				imgPath = imgsRoot + prefix + std::to_string(i) + suffix;
			std::string savePath = saveRoot + std::to_string(i) + suffix;

			cv::Mat img = cv::imread(imgPath);
			if (img.empty())
				break;

			ifs >> index >> x >> y >> width >> height;
			if (index != i)
			{
				VLOG(0) << "Index between img and roi not align";
				break;
			}
			cv::Rect roi(x, y, width, height);
			cv::Mat roiImg = ImgSlice(img, roi);
			cv::imwrite(savePath, roiImg);
			VLOG(0) << index << " " << imgPath;
		}
		return;
	}

	cv::Mat ImgSlice(const cv::Mat &img, const cv::Rect &roi)
	{
		cv::Mat roiImg;
		img(roi).copyTo(roiImg);
		cv::imshow("roiImg", roiImg);
		cv::waitKey(1);
		return roiImg;
	}

	void ImgsRecover(const std::string &roiImgsRoot, const std::string &roiPath,
					 const std::string &saveRoot,
					 const std::string &bgImgsRoot,
					 const std::string &prefix, const std::string &suffix,
					 const bool &zeroPadding)
	{
		std::ifstream ifs(roiPath);
		int index, x, y, w, h;

		for (int i = 0;; i++)
		{
			std::string rip = roiImgsRoot + std::to_string(i) + suffix; // roi img path
			std::string bip;
			if (zeroPadding)
				bip = bgImgsRoot + prefix + ZeroPaddingIndex(i, 4) + suffix;
			else
				bip = bgImgsRoot + prefix + std::to_string(i) + suffix;
			std::string sip = saveRoot + std::to_string(i) + suffix;
			cv::Mat roiImg = cv::imread(rip);
			cv::Mat bgImg = cv::imread(bip);
			if (roiImg.empty())
			{
				VLOG(0) << "roi img empty " << rip;
				break;
			}
			if (bgImg.empty())
			{
				VLOG(0) << "backgound img empty";
				break;
			}

			ifs >> index >> x >> y >> w >> h;
			if (index != i)
			{
				VLOG(0) << "Index between img and roi not align: " << index << " " << i << "\n"
						<< "img is: " << bip << "\n"
						<< "roi is: " << rip;

				break;
			}
			cv::Rect roi(x, y, w, h);
			cv::Mat res = ImgRecover(roiImg, roi, bgImg, false);
			cv::imshow("res", res);
			cv::waitKey(1);
			cv::imwrite(sip, res);
		}
		ifs.close();
		return;
	}

	cv::Mat ImgRecover(const cv::Mat &roiImg, const cv::Rect &roi, const cv::Mat &bg, const bool &useBg)
	{
		cv::Mat res;
		if (useBg)
			res = bg.clone();
		else
			res = cv::Mat::zeros(bg.size(), bg.type());
		cv::imshow("roiImg", roiImg);
		// cv::waitKey(0);
		roiImg.copyTo(res(roi));
		return res;
	}

	std::string ZeroPaddingIndex(const int &index, const int &number)
	{
		std::stringstream ss;
		ss << std::setw(number) << std::setfill('0') << std::to_string(index);
		return ss.str();
	}

	cv::Mat ImgEnhance(cv::Mat img)
	{
		cv::Mat res[3];
		cv::split(img, res);

		cv::equalizeHist(res[0], res[0]);
		cv::equalizeHist(res[1], res[1]);
		cv::equalizeHist(res[2], res[2]);

		cv::Mat enhance;
		cv::merge(res, 3, enhance);
		return enhance;
	}

	cv::Mat ImgDeleteBg(cv::Mat img, cv::Scalar scalar, int thresh)
	{
		cv::Mat res = img.clone();
		cv::Mat spl[3];
		cv::split(img, spl);

		int w = img.cols;
		int h = img.rows;

		int level;
		if (scalar[0])
			level = 0;
		if (scalar[1])
			level = 1;
		if (scalar[2])
			level = 2;
		VLOG(0) << level;
		for (int i = 0; i < h; i++)
		{
			uchar *ps = spl[level].ptr<uchar>(i);
			// cv::Vec3b *pr = res.ptr<cv::Vec3b>(i);
			for (int j = 0; j < w; j++)
			{
				if ((int)ps[j] < thresh)
				{
					res.at<cv::Vec3b>(i, j)[0] = 0;
					res.at<cv::Vec3b>(i, j)[1] = 0;
					res.at<cv::Vec3b>(i, j)[2] = 0;
				}
			}
		}
		cv::imshow("res", res);
		return res;
	}

} // namespace ar3dv