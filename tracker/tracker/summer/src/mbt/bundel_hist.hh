#pragma once
#include <vector>
#include "mbt/object3d.hh"
#include "mbt/tclc_histograms.hh"
#include "../base/compatible.h"

namespace summer
{

	class Renderer;
	class SearchLine;

	class BundleHist
	{
	public:
		BundleHist(const std::vector<Object3D *> &objs);

		virtual ~BundleHist() = 0;

		virtual void Update(cv::Mat &frame, cv::Mat &mask_map, cv::Mat &depth_map, int oid, float afg, float abg) = 0;
		virtual void GetPixelProb(uchar rc, uchar gc, uchar bc, int x, int y, int oid, float &ppf, float &ppb) {}
		virtual void GetBundleProb(SearchLine *search_line, const cv::Mat &frame, int oid) = 0;
		virtual void GetRegionProb(const cv::Mat &frame, int oid, cv::Mat &prob_map) = 0;

	protected:
		Renderer *renderer_;
		std::vector<Object3D *> objs_;
	};

	class RBOTHist : public BundleHist
	{
	public:
		RBOTHist(const std::vector<Object3D *> &objects);

		virtual void Update(cv::Mat &frame, cv::Mat &mask_map, cv::Mat &depth_map, int oid, float afg, float abg) override;
		virtual void GetPixelProb(uchar rc, uchar gc, uchar bc, int x, int y, int oid, float &ppf, float &ppb) override;
		virtual void GetBundleProb(SearchLine *search_line, const cv::Mat &frame, int oid) override;
		virtual void GetRegionProb(const cv::Mat &frame, int oid, cv::Mat &prob_map) override;
	};

} // ns sumemr