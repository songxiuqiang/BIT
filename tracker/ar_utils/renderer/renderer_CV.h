#ifndef AR3DVENGINE_RENDERER_CV_H
#define AR3DVENGINE_RENDERER_CV_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "model/model.h"

/**
 * This class implements OpenCV-based rendering.
 */
namespace ar3dv {

    class RenderingEngineCV {
    public:

        RenderingEngineCV() {};

        ~RenderingEngineCV() {};

		static RenderingEngineCV *Instance(void)
        {
            if (m_instance == NULL) m_instance = new RenderingEngineCV();
            return m_instance;
        }
        /**
         * Init the GL renderer. Pointer model paras to render paras.
         *
         * @param K intrinsic paras
         * @param width img width
         * @param height img height
         * @param zNear near plane
         * @param zFar far plane
         * @param model the model to be rendered
         */
        void init(const cv::Matx33f &K, int width, int height, float zNear, float zFar,
                  Model *model);

        /**
         * Render the object.
         * For now, only render the foreground mask and the depth, need to be replenish
         *
         * @param model the model to be rendered
         */
        void render(Model *model);

        /**
         * Get the foreground mask of the object
         *
         * @return the object foreground mask
         */
        cv::Mat getMask();

        /**
         * Get the depth map of the object
         *
         * @return the object depth map
         */
        cv::Mat getDepth();

        /**
         * Compute the visible faces of the object at the current pose
         *
         * @param model the model to be renderd
         * @param R the rotation matrix
         * @param t the translation vector
         * @param visibleFacesId the visible face ID to be computed
         */
        void getVisibleFaces(Model *model, cv::Matx33f &R, cv::Vec3f &t, std::vector<size_t> &visibleFacesId);

        /**
         * Unproject 2D points in image to 3D points in object coordinate frame
         *
         * @param p2d points in 2D image
         * @param faceID the points's face ID
         * @param K intrinsic paras
         * @param R rotation matrix
         * @param t translation vector
         * @param p3d the 3D points in the object coordinate frame
         */
        void unproject(cv::Point &p2d, int faceID, cv::Matx33f &K, cv::Matx33f &R,
                       cv::Vec3f &t, cv::Vec3f &p3d);

    private:
		static RenderingEngineCV* m_instance;

        cv::Matx33f K;

        std::vector<cv::Vec3f> vertices;
        std::vector<unsigned int> indices;

        std::vector<cv::Vec3i> faces;
        std::vector<cv::Vec3f> faceNormals;
        std::vector<cv::Vec3f> faceCenters;
        std::vector<float> minusd;

        cv::Mat fgMask;
        cv::Mat depth;
    };

}

#endif //AR3DVENGINE_RENDERER_CV_H
