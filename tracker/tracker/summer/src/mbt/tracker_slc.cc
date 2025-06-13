#include <iostream>

#include <fstream>
#include <sstream>
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <glog/logging.h>

#include "base/global_param.hh"
#include "mbt/m_func.hh"
#include "mbt/bundel_hist.hh"
#include "mbt/search_line.hh"
#include "mbt/tracker_slc.hh"

// #define SHOW_SLC_DEBUG
//  #define SHOW_SLC_PROB_MAP
// #define SHOW_SLC_SEARCH_LINE
// #define SHOW_SLC_BUNDLE
// #define SHOW_SLC_CONTOUR_POINTS
// #define SHOW_SLC_MODELLED_OCCLUSION
// #define SHOW_SLC_SEARCH_LINE_WEIGHT
// #define SHOW_RUNTIME

namespace summer
{

	enum
	{
		RUN_TRACK = 0,
		RUN_DEBUG = 1,
	};

	void Depth2Mask(const cv::Mat &depth, cv::Mat &mask,
					const cv::Rect &roi, const int &maskValue)
	{
		mask = cv::Mat(depth.size(), CV_8UC1, cv::Scalar(0));
		uchar depthType = depth.type() & CV_MAT_DEPTH_MASK;
		int yRange = roi.y + roi.height;
		int xRange = roi.x + roi.width;

		if (depthType == CV_32F)
		{
			for (int r = roi.y; r < yRange; r++)
			{
				const float *psrc = depth.ptr<float>(r);
				uchar *pdst = mask.ptr<uchar>(r);
				for (int c = roi.x; c < xRange; c++)
				{
					if (psrc[c] != 0)
					{
						pdst[c] = maskValue;
					}
				}
			}
		}
		else
		{
			LOG(ERROR) << "WRONG IMAGE TYPE";
		}
	}
	cv::Mat DrawResultOverlay2(const std::vector<Object3D *> &objects, const cv::Mat &frame, const int &level = 0)
	{
		// render the models with phong shading
		Renderer::Instance()->setLevel(level);
		std::vector<cv::Point3f> colors;
		colors.push_back(cv::Point3f(1.0, 0.5, 0.0));
		Renderer::Instance()->RenderShaded(std::vector<Model *>(objects.begin(), objects.end()), GL_FILL, colors, true);

		// download the rendering to the CPU
		cv::Mat rendering = Renderer::Instance()->DownloadFrame(Renderer::RGB);

		// download the depth buffer to the CPU
		cv::Mat depth = Renderer::Instance()->DownloadFrame(Renderer::DEPTH);

		// compose the rendering with the current camera image for demo purposes (can be done more efficiently directly in OpenGL)
		cv::Mat result = frame.clone();
		for (int y = 0; y < frame.rows; y++)
		{
			for (int x = 0; x < frame.cols; x++)
			{
				cv::Vec3b color = rendering.at<cv::Vec3b>(y, x);
				if (depth.at<float>(y, x) != 0.0f)
				{
					result.at<cv::Vec3b>(y, x)[0] = 0.5 * color[2] + 0.5 * result.at<cv::Vec3b>(y, x)[0];
					result.at<cv::Vec3b>(y, x)[1] = 0.5 * color[1] + 0.5 * result.at<cv::Vec3b>(y, x)[1];
					result.at<cv::Vec3b>(y, x)[2] = 0.5 * color[0] + 0.5 * result.at<cv::Vec3b>(y, x)[2];
				}
			}
		}
		return result;
	}

	SLCTracker::SLCTracker(const cv::Matx33f &K, std::vector<Object3D *> &objects) : Tracker(K, objects)
	{
		tk::GlobalParam *gp = tk::GlobalParam::Instance();
		search_line = new SearchLine();

		float zn = renderer_->zn();
		float zf = renderer_->zf();
	}

	SLCTracker::~SLCTracker()
	{
		if (search_line != nullptr)
			delete search_line;
	}

	bool SLCTracker::UseLargerImage(int &numInitialized, int &level, const int &width, const int &height, const bool &use)
	{
		if (!use)
			return false;
		bool res{false};
		for (int o = 0; o < objects_.size(); o++)
		{
			if (!objects_[o]->isInitialized())
				continue;

			numInitialized++;

			cv::Rect roi = Compute2DROI(objects_[o], cv::Size(width / pow(2, level), height / pow(2, level)), 8);
			if (roi.area() == 0)
				continue;
			while (roi.area() < 3000 && level > 0)
			{
				res = true;
				level--;
				renderer_->setLevel(level);
				roi = Compute2DROI(objects_[o], cv::Size(width / pow(2, level), height / pow(2, level)), 8);
			}
		}
		return res;
	}

	cv::Mat SLCTracker::prob_map(const cv::Mat &frame, const int &objIndex)
	{
		hists->GetRegionProb(frame, 0, prob_map_);
		return prob_map_;
	}

	cv::Mat SLCTracker::SetMask(const int &numInitialized, const cv::Mat &depth)
	{
		if (numInitialized > 1)
			return renderer_->DownloadFrame(Renderer::MASK);
		else
			return depth;
	}

	void SLCTracker::Track(std::vector<cv::Mat> &imagePyramid, std::vector<Object3D *> &objects, int runs)
	{
		RunIteration(objects, imagePyramid, 2, 12, 2, 8.0, 1.2);
		RunIteration(objects, imagePyramid, 2, 12, 2, 8.0, 1.2);
		RunIteration(objects, imagePyramid, 2, 12, 2, 8.0, 1.2);
		RunIteration(objects, imagePyramid, 2, 12, 2, 8.0, 1.2);
		RunIteration(objects, imagePyramid, 1, 10, 2, 6.0, 1.0);
		RunIteration(objects, imagePyramid, 1, 10, 2, 6.0, 1.0);
		RunIteration(objects, imagePyramid, 0, 8, 2, 4.0, 0.8);
	}

	bool IsOccluded(int oid, int pixel_idx, int contour_idx, uchar *mask_data, float *depth_data)
	{
		uchar oidc = mask_data[pixel_idx];
		if (oidc != 0 && oidc != oid && depth_data[contour_idx] < depth_data[pixel_idx])
		{
			return true;
		}
		return false;
	}

	inline float ColorWeight(float c, float x)
	{
		return exp(1.2 * (x - 1));
		return 1 - (x * x - 1);
		return (1 - c * (x - 1) * c * (x - 1)) * (1 - c * (x - 1) * c * (x - 1));
	}

	inline float DistanceWeight(float lambda, float x)
	{
		return exp(lambda * x);
	}

	void SLCTracker::ComputeJac(
		Object3D *object,
		int m_id,
		const cv::Mat &frame,
		const cv::Mat &mask_map,
		const cv::Mat &masks_map,
		const cv::Mat &depth_map,
		const cv::Mat &depth_inv_map,
		SearchLine *search_line,
		cv::Matx66f &wJTJM, cv::Matx61f &JTM,
		float band_width,
		float ss)
	{
		float *depth_data = (float *)depth_map.ptr<float>();
		float *depth_inv_data = (float *)depth_inv_map.ptr<float>();
		uchar *frame_data = frame.data;
		uchar *mask_data = mask_map.data;
		uchar *masks_data = masks_map.data;
		const std::vector<std::vector<cv::Point>> &search_points = search_line->search_points;
		const std::vector<std::vector<cv::Point2f>> &bundle_prob = search_line->bundle_prob;

		JTM = cv::Matx61f::zeros();
		wJTJM = cv::Matx66f::zeros();
		float *JT = JTM.val;
		float *wJTJ = wJTJM.val;

		float zf = renderer_->zf();
		float zn = renderer_->zn();
		cv::Matx33f K = renderer_->K44s()[renderer_->level()].get_minor<3, 3>(0, 0);
		cv::Matx33f K_inv = K.inv();
		float *K_inv_data = K_inv.val;
		float fx = K(0, 0);
		float fy = K(1, 1);

		for (int r = 0; r < search_points.size(); r++)
		{
			if (!search_line->actives[r])
				continue;

			int mid = search_points[r][search_points[r].size() - 1].x;
			int eid = search_points[r][search_points[r].size() - 1].y;

			float nx = search_line->norms[r].x;
			float ny = search_line->norms[r].y;

			// if (eid < 0)
			//	continue;

			float lambda1 = -1.2f;
			float we = eid < 0 ? 0.1f : ColorWeight(lambda1, scores[r]);

			int mx = search_points[r][mid].x;
			int my = search_points[r][mid].y;
			int zidx = my * depth_map.cols + mx;

			for (int c = 0; c < search_points[r].size() - 1; c++)
			{
				int pidx = search_points[r][c].y * frame.cols + search_points[r][c].x;
				if ((c < mid && mask_data[pidx]) || (c > mid && !mask_data[pidx]))
					continue;

				float pyf = bundle_prob[r][c].x;
				float pyb = bundle_prob[r][c].y;
				if (eid > 0 && (c < eid && pyb < pyf || c > eid && pyf < pyb))
					continue;

				float dist = GetDistance(search_points[r][c], search_points[r][mid]);
				if (dist > band_width)
					continue;

				if (c > mid)
					dist = -dist;

				float lambda2 = -0.25f;
				float wd = eid < 0 ? DistanceWeight(lambda2, 8.0f) : DistanceWeight(lambda2, GetDistance(search_points[r][c], search_points[r][eid]));

				float wa = we * wd;

				float s = ss;
				float s2 = s * s;

				float heaviside = 1.0f / float(CV_PI) * (-atan(dist * s)) + 0.5f;
				float e = heaviside * (pyf - pyb) + pyb + 0.000001;
				float dirac = (1.0f / float(CV_PI)) * (s / (dist * s2 * dist + 1.0f));
				float DlogeDe = -(pyf - pyb) / e;
				float constant_deriv = DlogeDe * dirac;
				float c2 = constant_deriv * constant_deriv;
				float w = -1.0f / log(e) * wa;

				float depth = 1.0f - depth_data[zidx];
				float D = 2.0f * zn * zf / (zf + zn - (2.0f * depth - 1.0) * (zf - zn));
				float Xc = D * (K_inv_data[0] * mx + K_inv_data[2]);
				float Yc = D * (K_inv_data[4] * my + K_inv_data[5]);
				float Zc = D;

				float J[6];

				float Zc2 = Zc * Zc;
				J[0] = nx * (-Xc * fx * Yc / Zc2) + ny * (-fy - Yc * Yc * fy / Zc2);
				J[1] = nx * (fx + Xc * Xc * fx / Zc2) + ny * (Xc * Yc * fy / Zc2);
				J[2] = nx * (-fx * Yc / Zc) + ny * (Xc * fy / Zc);
				J[3] = nx * (fx / Zc);
				J[4] = ny * (fy / Zc);
				J[5] = nx * (-Xc * fx / Zc2) + ny * (-Yc * fy / Zc2);

				for (int n = 0; n < 6; n++)
				{
					JT[n] += constant_deriv * J[n] * wa;
				}

				for (int n = 0; n < 6; n++)
					for (int m = n; m < 6; m++)
					{
						wJTJ[n * 6 + m] += w * J[n] * c2 * J[m];
					}

				depth = 1.0f - depth_inv_data[zidx];
				D = 2.0f * zn * zf / (zf + zn - (2.0f * depth - 1.0) * (zf - zn));
				Xc = D * (K_inv_data[0] * mx + K_inv_data[2]);
				Yc = D * (K_inv_data[4] * my + K_inv_data[5]);
				Zc = D;

				Zc2 = Zc * Zc;
				J[0] = nx * (-Xc * fx * Yc / Zc2) + ny * (-fy - Yc * Yc * fy / Zc2);
				J[1] = nx * (fx + Xc * Xc * fx / Zc2) + ny * (Xc * Yc * fy / Zc2);
				J[2] = nx * (-fx * Yc / Zc) + ny * (Xc * fy / Zc);
				J[3] = nx * (fx / Zc);
				J[4] = ny * (fy / Zc);
				J[5] = nx * (-Xc * fx / Zc2) + ny * (-Yc * fy / Zc2);

				for (int n = 0; n < 6; n++)
				{
					JT[n] += constant_deriv * J[n] * wa;
				}

				for (int n = 0; n < 6; n++)
					for (int m = n; m < 6; m++)
					{
						wJTJ[n * 6 + m] += w * J[n] * c2 * J[m];
					}
			}
		}

		for (int i = 0; i < wJTJM.rows; i++)
			for (int j = i + 1; j < wJTJM.cols; j++)
			{
				wJTJM(j, i) = wJTJM(i, j);
			}
	}

	void ShowContourPoints(
		SearchLine *search_line, const cv::Mat &frame, std::vector<float> &scores,
		cv::Mat &all_pt_img, cv::Mat &pc_img, cv::Mat &filter_pt_img, cv::Mat &ctr_pt_img)
	{
		std::vector<std::vector<cv::Point>> &search_points = search_line->search_points;
		std::vector<std::vector<cv::Point2f>> &bundle_prob = search_line->bundle_prob;

		all_pt_img = cv::Mat(frame.size(), CV_8UC1);
		all_pt_img = 0;

		pc_img = cv::Mat(frame.size(), CV_8UC1);
		pc_img = 0;

		filter_pt_img = cv::Mat(frame.size(), CV_8UC1);
		filter_pt_img = 255;

		for (int r = 0; r < bundle_prob.size(); ++r)
		{
			float score_min = 100000000;

			float nx = search_line->norms[r].x;
			float ny = search_line->norms[r].y;

			float prob_max = 0.0f;
			search_points[r][search_points[r].size() - 1].y = -1;
			scores[r] = 0.0f;

			int mid = search_points[r][search_points[r].size() - 1].x;

			for (int c = 3; c < bundle_prob[r].size() - 3; ++c)
			{
				if (fabs(bundle_prob[r][c + 1].x - bundle_prob[r][c - 1].x) > 0.2f)
				{
					float prbf =
						bundle_prob[r][c - 3].x *
						bundle_prob[r][c - 2].x *
						bundle_prob[r][c - 1].x;
					float prbb =
						bundle_prob[r][c - 3].y *
						bundle_prob[r][c - 2].y *
						bundle_prob[r][c - 1].y;
					float prff =
						bundle_prob[r][c + 1].x *
						bundle_prob[r][c + 2].x *
						bundle_prob[r][c + 3].x;
					float prfb =
						bundle_prob[r][c + 1].y *
						bundle_prob[r][c + 2].y *
						bundle_prob[r][c + 3].y;

					float pr_C = prbb * prff;
					float pr_F = prbf * prff;
					float pr_B = prbb * prfb;

					float prn_C = pr_C / (pr_C + pr_F + pr_B);

					all_pt_img.at<uchar>(search_points[r][c]) = 255;

					uchar pcv = 255 * prn_C;
					pcv = pcv > 0 ? pcv : 1;
					pc_img.at<uchar>(search_points[r][c]) = pcv;

					if (prn_C > 0.5f)
					{

						filter_pt_img.at<uchar>(search_points[r][c]) = 0;

						float ex = nx * (search_points[r][mid].x - search_points[r][c].x) + ny * (search_points[r][mid].y - search_points[r][c].y);

						float score = -log(prn_C);
						if (score < score_min)
						{
							score_min = score;
							scores[r] = pr_C;
							search_points[r][search_points[r].size() - 1].y = c;
						}
					}
				}
			}
		}

		ctr_pt_img = cv::Mat(frame.size(), CV_8UC1, cv::Scalar(255));
		for (int r = 0; r < bundle_prob.size(); ++r)
		{
			int eid = search_points[r][search_points[r].size() - 1].y;
			if (eid < 0)
				continue;
			ctr_pt_img.at<uchar>(search_points[r][eid]) = 0;
		}
	}

	void SLCTracker::FindMatchPoint(SearchLine *search_line, float diff)
	{
		std::vector<std::vector<cv::Point>> &search_points = search_line->search_points;
		const std::vector<std::vector<cv::Point2f>> &bundle_prob = search_line->bundle_prob;
		scores.resize(search_points.size());

		for (int r = 0; r < bundle_prob.size(); ++r)
		{
			float score_min = 1000000;
			search_points[r][search_points[r].size() - 1].y = -1;
			scores[r] = 0;
			for (int c = 3; c < bundle_prob[r].size() - 3; ++c)
			{
				if (fabs(bundle_prob[r][c + 1].x - bundle_prob[r][c - 1].x) > 0.5f)
				{
					float prbf =
						bundle_prob[r][c - 3].x *
						bundle_prob[r][c - 2].x *
						bundle_prob[r][c - 1].x;
					float prbb =
						bundle_prob[r][c - 3].y *
						bundle_prob[r][c - 2].y *
						bundle_prob[r][c - 1].y;
					float prff =
						bundle_prob[r][c + 1].x *
						bundle_prob[r][c + 2].x *
						bundle_prob[r][c + 3].x;
					float prfb =
						bundle_prob[r][c + 1].y *
						bundle_prob[r][c + 2].y *
						bundle_prob[r][c + 3].y;

					float pr_C = prbb * prff;
					float pr_F = prbf * prff;
					float pr_B = prbb * prfb;

					if ((pr_C > pr_F) && (pr_C > pr_B))
					{
						float score = -log(prbb) - log(prff);
						if (score < score_min)
						{
							score_min = score;
							scores[r] = pr_C;
							search_points[r][search_points[r].size() - 1].y = c;
						}
					}
				}
			}
		}
	}

	static void ConvertMask(const cv::Mat &src_mask, cv::Mat &mask, uchar oid)
	{
		mask = cv::Mat(src_mask.size(), CV_8UC1, cv::Scalar(0));
		uchar depth = src_mask.type() & CV_MAT_DEPTH_MASK;

		if (CV_8U == depth && oid > 0)
		{
			for (int r = 0; r < src_mask.rows; ++r)
				for (int c = 0; c < src_mask.cols; ++c)
				{
					if (oid == src_mask.at<uchar>(r, c))
						mask.at<uchar>(r, c) = 255;
				}
		}
		else if (CV_32F == depth)
		{
			for (int r = 0; r < src_mask.rows; ++r)
				for (int c = 0; c < src_mask.cols; ++c)
				{
					if (src_mask.at<float>(r, c))
						mask.at<uchar>(r, c) = 255;
				}
		}
		else
		{
			LOG(ERROR) << "WRONG IMAGE TYPE";
		}
	}

	void ShowModelledOccludedMaskContour(int m_id, const cv::Mat &masks_map, const cv::Mat &depth_map, SearchLine *sl, cv::Mat &contour_map)
	{
		contour_map = cv::Mat(masks_map.size(), CV_8UC3);
		contour_map = 0;

		for (int r = 0; r < contour_map.rows; ++r)
			for (int c = 0; c < contour_map.cols; ++c)
			{
				if (1 == masks_map.at<uchar>(r, c))
				{
					contour_map.at<cv::Vec3b>(r, c)[0] = 255;
					contour_map.at<cv::Vec3b>(r, c)[1] = 255;
					contour_map.at<cv::Vec3b>(r, c)[2] = 255;
				}
				else if (2 == masks_map.at<uchar>(r, c))
				{
					contour_map.at<cv::Vec3b>(r, c)[0] = 128;
					contour_map.at<cv::Vec3b>(r, c)[1] = 128;
					contour_map.at<cv::Vec3b>(r, c)[2] = 128;
				}
			}

		for (int oid = 1; oid <= 2; oid++)
		{
			cv::Mat mask_map;
			ConvertMask(masks_map, mask_map, oid);

			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(mask_map, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			cv::Vec3b color;
			if (1 == oid)
				color = cv::Vec3b(0, 255, 0);
			if (2 == oid)
				color = cv::Vec3b(203, 192, 255);

			for (auto contour : contours)
				for (auto pt : contour)
				{
					contour_map.at<cv::Vec3b>(pt) = color;
				}
		}

		const std::vector<std::vector<cv::Point>> &search_points = sl->search_points;

		for (int r = 0; r < search_points.size(); ++r)
		{
			int mid = search_points[r][search_points[r].size() - 1].x;

			cv::Point ptc = search_points[r][mid];
			cv::Point ptb = search_points[r][mid - 1];

			uchar oidc = masks_map.at<uchar>(ptc);
			uchar oidb = masks_map.at<uchar>(ptb);

			if (oidb != 0 && oidb != oidc && depth_map.at<float>(ptc) < depth_map.at<float>(ptb))
			{
				contour_map.at<cv::Vec3b>(ptc) = cv::Vec3b(0, 0, 255);
			}
		}
	}

	void ShowModelledOccludedSearchLine(int m_id, const cv::Mat &mask_map, const cv::Mat &depth_map, SearchLine *sl, cv::Mat &sl_map)
	{
		const std::vector<std::vector<cv::Point>> &search_points = sl->search_points;
		float *depth_data = (float *)depth_map.ptr<float>();
		uchar *buf_data = sl_map.data;
		uchar *mask_data = mask_map.data;

		for (int r = 0; r < search_points.size(); r++)
		{
			int mid = search_points[r][search_points[r].size() - 1].x;
			int mx = search_points[r][mid].x;
			int my = search_points[r][mid].y;
			int zidx = my * depth_map.cols + mx;

			// occlude type 1
			if (0 == sl->actives[r])
			{
				for (int c = 0; c < search_points[r].size() - 1; c++)
				{
					int pidx = search_points[r][c].y * sl_map.cols + search_points[r][c].x;

					buf_data[3 * pidx] = 0;
					buf_data[3 * pidx + 1] = 0;
					buf_data[3 * pidx + 2] = 255;
				}
				continue;
			}

			// occlude type 2
			for (int c = 0; c < search_points[r].size() - 1; c++)
			{
				int pidx = search_points[r][c].y * sl_map.cols + search_points[r][c].x;

				if (m_id > 0 && c < mid && IsOccluded(m_id, pidx, zidx, mask_data, depth_data))
				{
					buf_data[3 * pidx] = 0;
					buf_data[3 * pidx + 1] = 0;
					buf_data[3 * pidx + 2] = 255;
				}
				else
				{
					buf_data[3 * pidx] = 255;
					buf_data[3 * pidx + 1] = 255;
					buf_data[3 * pidx + 2] = 255;
				}
			}
		}
	}

	void ShowBundle(SearchLine *search_line, int len, const cv::Mat &frame, const cv::Mat &prob, cv::Mat &brgb, cv::Mat &bpr, cv::Mat &mrgb)
	{
		const std::vector<std::vector<cv::Point>> &search_points = search_line->search_points;
		const std::vector<std::vector<cv::Point2f>> &bundle_prob = search_line->bundle_prob;

		int width = len * 2 + 1;
		brgb = cv::Mat(search_points.size(), width, CV_8UC3);
		bpr = cv::Mat(search_points.size(), width, CV_8UC1);

		for (int r = 0; r < search_points.size(); r++)
		{
			int i = 0;
			for (; i < width - search_points[r].size() + 1; ++i)
			{
				brgb.at<cv::Vec3b>(r, i) = cv::Vec3b(0, 0, 0);
				bpr.at<uchar>(r, i) = 0;
			}

			for (int c = 0; c < search_points[r].size() - 1; c++)
			{
				brgb.at<cv::Vec3b>(r, i + c) = frame.at<cv::Vec3b>(search_points[r][c]);
				bpr.at<uchar>(r, i + c) = prob.at<uchar>(search_points[r][c]);
			}
		}

		mrgb = brgb.clone();

		for (int r = 0; r < bpr.rows; ++r)
		{
			float score_min = 1000000;
			cv::Point pt(-1, -1);
			for (int c = 3; c < bpr.cols - 3; ++c)
			{
				// SLE = 0.2; SLC = fabs(0.5)
				int diff = 0.2f * 255;
				float deno = 1.0f / (255.f * 255.f * 255.f);
				if (bpr.at<uchar>(r, c + 1) - bpr.at<uchar>(r, c - 1) > diff)
				{
					mrgb.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);

					float prbf =
						bpr.at<uchar>(r, c - 3) *
						bpr.at<uchar>(r, c - 2) *
						bpr.at<uchar>(r, c - 1) *
						deno;

					float prbb =
						(255 - bpr.at<uchar>(r, c - 3)) *
						(255 - bpr.at<uchar>(r, c - 2)) *
						(255 - bpr.at<uchar>(r, c - 1)) *
						deno;

					float prff =
						bpr.at<uchar>(r, c + 1) *
						bpr.at<uchar>(r, c + 2) *
						bpr.at<uchar>(r, c + 3) *
						deno;

					float prfb =
						(255 - bpr.at<uchar>(r, c + 1)) *
						(255 - bpr.at<uchar>(r, c + 2)) *
						(255 - bpr.at<uchar>(r, c + 3)) *
						deno;

					float pr_C = prbb * prff;
					float pr_F = prbf * prff;
					float pr_B = prbb * prfb;

					if ((pr_C > pr_F) && (pr_C > pr_B))
					{
						float score = -log(prbb) - log(prff);
						if (score < score_min)
						{
							score_min = score;
							pt.y = r;
							pt.x = c;
						}
					}
				}
			}
			if (-1 != pt.x)
				mrgb.at<cv::Vec3b>(pt) = cv::Vec3b(0, 255, 0);
		}
	}

	void ShowSearchLineWeight(SearchLine *search_line, const cv::Mat &frame, std::vector<float> &scores, cv::Mat &slw)
	{
		std::vector<std::vector<cv::Point>> &search_points = search_line->search_points;
		std::vector<std::vector<float>> ws(search_points.size());

		for (int r = 0; r < search_points.size(); r++)
		{
			int mid = search_points[r][search_points[r].size() - 1].x;
			int eid = search_points[r][search_points[r].size() - 1].y;

			float lambda1 = -1.2f;
			float we = eid < 0 ? 0.1f : ColorWeight(lambda1, scores[r]);

			ws[r].resize(search_points[r].size());
			for (int c = 0; c < search_points[r].size() - 1; c++)
			{
				float lambda2 = -0.25f;
				float wd = eid < 0 ? DistanceWeight(lambda2, 8.0f) : DistanceWeight(lambda2, GetDistance(search_points[r][c], search_points[r][eid]));
				ws[r][c] = we * wd;
			}
		}

		slw = cv::Mat(frame.size(), CV_8UC1, cv::Scalar(0));
		for (int r = 0; r < search_points.size(); r++)
		{
			for (int c = 0; c < search_points[r].size() - 1; c++)
			{
				int v = ws[r][c] * 255;
				if (v < 1)
					v = 1;
				slw.at<uchar>(search_points[r][c]) = v;
			}
		}
	}

	void SLCTracker::FilterOccludedPoint(const cv::Mat &mask, const cv::Mat &depth, SearchLine *sl)
	{
		const std::vector<std::vector<cv::Point>> &search_points = sl->search_points;

		for (int r = 0; r < search_points.size(); ++r)
		{
			int mid = search_points[r][search_points[r].size() - 1].x;

			cv::Point ptc = search_points[r][mid];
			cv::Point ptb = search_points[r][mid - 1];

			uchar oidc = mask.at<uchar>(ptc);
			uchar oidb = mask.at<uchar>(ptb);

			if (oidb != 0 && oidb != oidc && depth.at<float>(ptc) < depth.at<float>(ptb))
			{
				sl->actives[r] = 0;
			}
		}
	}

	cv::Mat ReplacePixelValueWithGLDepth(const cv::Mat &img1, const cv::Mat &img2)
	{
		cv::Mat res = cv::Mat::zeros(img2.size(), img2.type());
		for (size_t i = 0; i < img1.rows; ++i)
		{
			const float *p1 = img1.ptr<float>(i);
			const float *p2 = img2.ptr<float>(i);
			float *pres = res.ptr<float>(i);
			for (size_t j = 0; j < img1.cols; ++j)
			{
				if (p1[j] != 0.0f)
					pres[j] = p2[j];
			}
		}
		return res;
	}

	void SLCTracker::RunIteration(std::vector<Object3D *> &objects, const std::vector<cv::Mat> &imagePyramid,
								  int level, int sl_len, int sl_seg, float band_width, float ss, int run_type)
	{
		int width = renderer_->full_width();
		int height = renderer_->full_height();
		renderer_->setLevel(level);
		int numInitialized = 0;
		UseLargerImage(numInitialized, level, width, height, true);

		cv::Mat fineDepth;
		cv::Mat depth_map, masks_map;
		renderer_->RenderSilhouette(std::vector<Model *>(objects.begin(), objects.end()), GL_FILL, false);

		depth_map = renderer_->DownloadFrameFromROI(Renderer::DEPTH, objects[0]);

		masks_map = SetMask(numInitialized, depth_map);

		for (int o = 0; o < objects.size(); o++)
		{
			if (!objects[o]->isInitialized())
				continue;

			cv::Rect roi = Compute2DROI(objects[o], cv::Size(width / pow(2, level), height / pow(2, level)), 8);
			if (roi.area() == 0)
				continue;

			int m_id = (numInitialized <= 1) ? -1 : objects[o]->getModelID();
			cv::Mat mask_map;
			ConvertMask(masks_map, m_id, mask_map);

			search_line->FindContours(mask_map, sl_seg, true);
			search_line->FindSearchLine(mask_map, imagePyramid[level], sl_len, sl_seg, true);

			if (numInitialized > 1)
			{
				FilterOccludedPoint(masks_map, depth_map, search_line);
			}

			hists->GetBundleProb(search_line, imagePyramid[level], o);

			FindMatchPoint(search_line, 0.5);

			renderer_->RenderSilhouette(objects[o], GL_FILL, true);
			cv::Mat depth_inv_map = renderer_->DownloadFrameFromROI(Renderer::DEPTH, objects[0]);

			cv::Matx66f wJTJ = cv::Matx66f::zeros(), ipJTJ = cv::Matx66f::zeros();
			cv::Matx61f JT = cv::Matx61f::zeros(), iphJT = cv::Matx61f::zeros();

			ComputeJac(objects[o], m_id, imagePyramid[level], mask_map, masks_map, depth_map, depth_inv_map, search_line, wJTJ, JT, band_width, ss);
			cv::Matx44f T_cm = Transformations::exp(-wJTJ.inv(cv::DECOMP_CHOLESKY) * JT) * objects[o]->getPose();
			objects[o]->setPose(T_cm);
		}
	}

	cv::Rect SLCTracker::compute2DROI(Object3D *object, const cv::Size &maxSize, int offset)
	{
		// PROJECT THE 3D BOUNDING BOX AS 2D ROI
		cv::Rect boundingRect;
		std::vector<cv::Point2f> projections;

		renderer_->ProjectBoundingBox(object, projections, boundingRect);

		if (boundingRect.x >= maxSize.width || boundingRect.y >= maxSize.height || boundingRect.x + boundingRect.width <= 0 || boundingRect.y + boundingRect.height <= 0)
		{
			return cv::Rect(0, 0, 0, 0);
		}

		// CROP THE ROI AROUND THE SILHOUETTE
		cv::Rect roi = cv::Rect(boundingRect.x - offset, boundingRect.y - offset, boundingRect.width + 2 * offset, boundingRect.height + 2 * offset);

		if (roi.x < 0)
		{
			roi.width += roi.x;
			roi.x = 0;
		}

		if (roi.y < 0)
		{
			roi.height += roi.y;
			roi.y = 0;
		}

		if (roi.x + roi.width > maxSize.width)
			roi.width = maxSize.width - roi.x;
		if (roi.y + roi.height > maxSize.height)
			roi.height = maxSize.height - roi.y;

		return roi;
	}

	void SLCTracker::UpdateHist(cv::Mat &frame)
	{
		float afg = 0.1, abg = 0.2;
		if (initialized_)
		{
			renderer_->setLevel(0);
			renderer_->RenderSilhouette(std::vector<Model *>(objects_.begin(), objects_.end()), GL_FILL);
			cv::Mat masks_map = renderer_->DownloadFrame(Renderer::MASK);
			cv::Mat depth_map = renderer_->DownloadFrame(Renderer::DEPTH);
			cv::Vec2f center = renderer_->Get2DCenter(objects_[0]);

			for (int oid = 0; oid < objects_.size(); oid++)
			{
				hists->Update(frame, masks_map, depth_map, oid, afg, abg);
			}
		}
	}

} // ns summer