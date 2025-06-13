#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "signed_distance_transform2d.hh"

#include "../base/compatible.h"
namespace summer
{

	class Model;

	/**
	 *  This class implements an statistical image segmentation model based on temporary
	 *  consistent, local color histograms (tclc-histograms). Here, each histogram corresponds
	 *  to a 3D vertex of a given 3D model.
	 */
	class TCLCHistograms
	{
	public:
		/**
		 *  Constructor that allocates both normalized and not normalized foreground
		 *  and background histograms for each vertex of the given 3D model.
		 *
		 *  @param  model The 3D model for which the histograms are being created.
		 *  @param  numBins The number of bins per color channel.
		 *  @param  radius The radius of the local image region in pixels used for updating the histograms.
		 *  @param  offset The minimum distance between two projected histogram centers in pixels during an update.
		 */
		TCLCHistograms(Model *model, int numBins, int radius, float offset);

		virtual ~TCLCHistograms();

		/**
		 *  Updates the histograms from a given camera frame by projecting all histogram
		 *  centers into the image and selecting those close or on the object's contour.
		 *
		 *  @param  frame The color frame to be used for updating the histograms.
		 *  @param  mask The corresponding binary shilhouette mask of the object.
		 *  @param  depth The per pixel depth map of the object used to filter histograms on the back of the object,
		 *  @param  K The camera's instrinsic matrix.
		 *  @param  zNear The near plane used to render the depth map.
		 *  @param  zFar The far plane used to render the depth map.
		 */
		virtual void update(const cv::Mat &frame, const cv::Mat &mask, const cv::Mat &depth, cv::Matx33f &K, float zNear, float zFar, float afg, float abg);

		/**
		 *  Computes updated center locations and IDs of all histograms that project onto or close
		 *  to the contour based on the current object pose at a specified image pyramid level.
		 *
		 *  @param  mask The binary shilhouette mask of the object.
		 *  @param  depth The per pixel depth map of the object used to filter histograms on the back of the object,
		 *  @param  K The camera's instrinsic matrix.
		 *  @param  zNear The near plane used to render the depth map.
		 *  @param  zFar The far plane used to render the depth map.
		 *  @param  level The image pyramid level to be used for the update.
		 */
		void updateCentersAndIds(const cv::Mat &mask, const cv::Mat &depth, const cv::Matx33f &K, float zNear, float zFar, int level);

		/**
		 *  Returns all normalized forground histograms in their current state.
		 *
		 *  @return The normalized foreground histograms.
		 */
		cv::Mat getLocalForegroundHistograms();

		/**
		 *  Returns all normalized background histograms in their current state.
		 *
		 *  @return The normalized background histograms.
		 */
		cv::Mat getLocalBackgroundHistograms();

		/**
		 *  Returns the locations and IDs of all histogram centers that where used for the last
		 *  update() or updateCentersAndIds() call.
		 *
		 *  @return The list of all current center locations on or close to the contour and their corresponding IDs [(x_0, y_0, id_0), (x_1, y_1, id_1), ...].
		 */
		std::vector<cv::Point3i> getCentersAndIDs();

		/**
		 *  Returns a 1D binary mask of all histograms where a '1' means that the histograms
		 *  corresponding to the index has been intialized before.
		 *
		 *  @return A 1D binary mask telling wheter each histogram has been initialized or not.
		 */
		cv::Mat getInitialized();

		/**
		 *  Returns the number of histogram bin per image channel as specified in the constructor.
		 *
		 *  @return The number of histogram bins per channel.
		 */
		int getNumBins();

		/**
		 *  Returns the number of histograms, i.e. verticies of the corresponding 3D model.
		 *
		 *  @return The number of histograms.
		 */
		int getNumHistograms();

		/**
		 *  Returns the radius of the local image region in pixels used for updating the
		 *  histograms as specified in the constructor.
		 *
		 *  @return The minumum distance between two projected histogram centers in pixels.
		 */
		int getRadius();

		/**
		 *  Returns the minumum distance between two projected histogram centers during an update
		 *  as specified in the constructor.
		 *
		 *  @return The minumum distance between two projected histogram centers in pixels.
		 */
		float getOffset();

		/**
		 *  Clears all histograms by resetting them to zero and setting their status to
		 *  uninitialized
		 */
		void clear();

		cv::Mat sumsFB;

		void TestLine(uchar *frameRow, uchar *maskRow, int xl, int xr, float *localHistogramFG, float *localHistogramBG, float &sum_err, float &sum_all);
		float ComputeWeight(const cv::Mat &frame, const cv::Mat &mask, cv::Point3i &center, int radius);
		std::vector<float> wes;

	protected:
		int numBins_;

		int numHistograms_;

		int radius_;

		float offset_;

		cv::Mat notNormalizedFG_;
		cv::Mat notNormalizedBG_;

		cv::Mat normalizedFG_;
		cv::Mat normalizedBG_;

		cv::Mat initialized;

		Model *model_;

		std::vector<cv::Point3i> centersIDs_;

		std::vector<cv::Point3i> computeLocalHistogramCenters(const cv::Mat &mask);

		std::vector<cv::Point3i> parallelComputeLocalHistogramCenters(const cv::Mat &mask, const cv::Mat &depth, const cv::Matx33f &K, float zNear, float zFar, int level);

		void filterHistogramCenters(int numHistograms, float offset);
	};

} // ns summer