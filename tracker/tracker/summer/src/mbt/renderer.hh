// #pragma once
#ifndef __SUMMER_VIEW__
#define __SUMMER_VIEW__

#include <iostream>
#include <vector>

#include <QOpenGLContext>
#include <QOffscreenSurface>

#include <QGLFramebufferObject>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_3_3_Core>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "transformations.hh"
#include "model.hh"

#include "../../../../ar_utils/timer/timer.h"

#include "../base/global_params.h"
#include "../base/compatible.h"
namespace summer
{

	class Renderer : public QOpenGLFunctions_3_3_Core
	{
	public:
		enum FrameType
		{
			MASK,
			RGB,
			RGB_32F,
			DEPTH
		};

		Renderer(void);

		~Renderer(void);

		static Renderer *Instance(void)
		{
			if (instance == NULL)
				instance = new Renderer();
			return instance;
		}

		bool IsInFrame(cv::Point2f &pt)
		{
			return (pt.x >= 0 && pt.x < width_ && pt.y >= 0 && pt.y < height_);
		}

		void init(const cv::Matx33f &K, int width, int height, float zNear, float zFar, int numLevels);

		void doneCurrent();

		void destroy();

		void setLevel(int level);

		void ChangeK44s(const cv::Matx33f &K, const float &zn = 0.1f, const float &zf = 2000.0f);
		void ChangeK44s(std::vector<cv::Matx44f> Ks);
		void OptimizeProjectionMatrix(const cv::Matx44f &pose, const float &bbxDiagLen);

		// Rendering
		void Project(const cv::Matx44f &mv_mat, std::vector<cv::Vec3f> &model_points, std::vector<cv::Vec2f> &image_points);
		void RenderCV(Model *model, cv::Mat &frame, cv::Scalar color = cv::Scalar(1, 255, 1));

		void RenderSilhouette(Model *model, GLenum polyonMode, bool invertDepth = false, float r = 1.0f, float g = 1.0f, float b = 1.0f, bool drawAll = false);
		void RenderSilhouette(std::vector<Model *> models, GLenum polyonMode, bool invertDepth = false, const std::vector<cv::Point3f> &colors = std::vector<cv::Point3f>(), bool drawAll = false);

		void RenderShaded(Model *model, GLenum polyonMode, float r = 1.0f, float g = 0.5f, float b = 0.0f, bool drawAll = false);
		void RenderShaded(std::vector<Model *> models, GLenum polyonMode, const std::vector<cv::Point3f> &colors = std::vector<cv::Point3f>(), bool drawAll = false);

		void RenderNormals(Model *model, GLenum polyonMode, bool drawAll = false);
		void RenderNormals(std::vector<Model *> models, GLenum polyonMode, bool drawAll = false);

		// Downloader
		cv::Mat DownloadFrame(Renderer::FrameType type);
		cv::Mat DownloadMask();
		cv::Mat DownloadFrameFromROI(Renderer::FrameType type, Model *model);
		cv::Mat DownloadFrameSafety(Renderer::FrameType type, Model *model);

		void ConvertMask(const cv::Mat &src_mask, cv::Mat &mask, uchar oid);

		cv::Rect Compute2DROI(Model *model, const cv::Size &maxSize, int offset);
		cv::Rect RoiByMask();
		cv::Rect RoiByMask(const cv::Mat &mask);
		void MakeSureRoiSafety(cv::Rect &roi);

		// Projecter
		void ProjectBoundingBox(Model *model, std::vector<cv::Point2f> &projections, cv::Matx44f &pose, cv::Rect &boundingRect);
		void ProjectBoundingBox(Model *model, std::vector<cv::Point2f> &projections, cv::Rect &boundingRect);
		void ProjectPoints(const std::vector<cv::Point3f> &pts3d, const cv::Matx44f &pose, std::vector<cv::Point2f> &pts);
		void BackProjectPoints(std::vector<cv::Point> &pts, const cv::Mat &depth_map, const cv::Matx44f &pose, std::vector<cv::Point3f> &pts3d);
		std::vector<cv::Vec2f> Project3DPoints(Model *model, const std::vector<cv::Vec3f> &points3D);

		// Draw
		bool DrawBBX(cv::Mat &img, Model *model);
		bool DrawCenter(cv::Mat &frame, Model *model);
		bool DrawLines(const cv::Mat &img, const cv::Point2f &p1, const cv::Point2f &p2, const cv::Scalar &scalar, const bool &arrow = false);
		cv::Mat DrawResultOverlay(const std::vector<Model *> &objects, const cv::Mat &frame);
		cv::Mat DrawResultOverlay(const std::vector<Model *> &objects, const cv::Mat &frame, const bool &success);
		cv::Mat DrawResultOverlayProjector(const std::vector<Model *> &objects, const cv::Mat &frame, const bool &success);
		cv::Mat DrawMeshOverlay(const std::vector<Model *> &objects, const cv::Mat &frame);
		cv::Mat DrawContourOverlay(const std::vector<Model *> &objects, const cv::Mat &frame);
		cv::Mat DrawMask(const std::vector<Model *> &objects);

		// Getters
		int max_levels() const { return max_levels_; }
		int level() const { return level_; }
		int full_width() { return full_width_; }
		int full_height() { return full_height_; }
		float zn() const { return zn_; }
		float zf() const { return zf_; }
		std::vector<cv::Matx44f> K44s() const { return K44s_; };
		cv::Matx33f K33(const int &level = 0) const;
		cv::Vec2f Get2DCenter(Model *model);

	protected:
		QOpenGLContext *getContext();
		GLuint getFrameBufferID();
		GLuint getColorTextureID();
		GLuint getDepthTextureID();

		void makeCurrent();
		// void doneCurrent();

	private:
		static Renderer *instance;

		std::vector<cv::Point2f> m_2D_bbx;
		cv::Vec3f m_center_3d;

		int max_levels_;
		int level_;

		int width_;
		int height_;

		int full_width_;
		int full_height_;

		float zn_;
		float zf_;

		std::vector<cv::Matx44f> K44s_;
		cv::Matx44f projection_matrix_;
		cv::Matx44f look_at_matrix_;

		QOffscreenSurface *surface;
		QOpenGLContext *glContext;

		GLuint frameBufferID;
		GLuint colorTextureID;
		GLuint depthTextureID;

		int angle;

		cv::Vec3f lightPosition;

		QString shaderFolder;
		QOpenGLShaderProgram *silhouetteShaderProgram;
		QOpenGLShaderProgram *phongblinnShaderProgram;
		QOpenGLShaderProgram *normalsShaderProgram;

		bool initRenderingBuffers();
		bool initShaderProgramFromCode(QOpenGLShaderProgram *program, char *vertex_shader, char *fragment_shader);
	};

} // ns slot

#endif