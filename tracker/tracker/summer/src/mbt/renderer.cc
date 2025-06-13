#include <iostream>

#include <glog/logging.h>
#include "renderer.hh"

using namespace std;
using namespace cv;
using namespace summer;

Renderer *Renderer::instance;

Renderer::Renderer(void)
{
	QSurfaceFormat glFormat;
	glFormat.setVersion(3, 3);
	glFormat.setProfile(QSurfaceFormat::CoreProfile);
	glFormat.setRenderableType(QSurfaceFormat::OpenGL);

	surface = new QOffscreenSurface();
	surface->setFormat(glFormat);
	surface->create();

	glContext = new QOpenGLContext();
	glContext->setFormat(surface->requestedFormat());
	glContext->create();

	silhouetteShaderProgram = new QOpenGLShaderProgram();
	phongblinnShaderProgram = new QOpenGLShaderProgram();
	normalsShaderProgram = new QOpenGLShaderProgram();

	K44s_.push_back(Matx44f::eye());

	projection_matrix_ = Transformations::perspectiveMatrix(40, 4.0f / 3.0f, 0.1, 1000.0);

	look_at_matrix_ = Transformations::lookAtMatrix(0, 0, 0, 0, 0, 1, 0, -1, 0);

	level_ = 0;
}

Renderer::~Renderer(void)
{
	glDeleteTextures(1, &colorTextureID);
	glDeleteTextures(1, &depthTextureID);
	glDeleteFramebuffers(1, &frameBufferID);

	delete phongblinnShaderProgram;
	delete normalsShaderProgram;
	delete silhouetteShaderProgram;
	delete surface;
}

void Renderer::destroy()
{
	doneCurrent();

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	delete instance;
	instance = NULL;
}

void Renderer::makeCurrent()
{
	glContext->makeCurrent(surface);
}

void Renderer::doneCurrent()
{
	glContext->doneCurrent();
}

QOpenGLContext *Renderer::getContext()
{
	return glContext;
}

GLuint Renderer::getFrameBufferID()
{
	return frameBufferID;
}

GLuint Renderer::getColorTextureID()
{
	return colorTextureID;
}

GLuint Renderer::getDepthTextureID()
{
	return depthTextureID;
}

cv::Matx33f Renderer::K33(const int &level) const
{
	cv::Matx44f K44 = K44s_[level];
	cv::Matx33f K33(K44(0, 0), K44(0, 1), K44(0, 2),
					K44(1, 0), K44(1, 1), K44(1, 2),
					K44(2, 0), K44(2, 1), K44(2, 2));
	return K33;
}

#include "shader/shaders.hh"
#include <opencv2/highgui.hpp>

void Renderer::init(const Matx33f &K, int width, int height, float zNear, float zFar, int numLevels)
{
	this->width_ = width;
	this->height_ = height;

	full_width_ = width;
	full_height_ = height;

	zn_ = zNear;
	zf_ = zFar;

	max_levels_ = numLevels;

	projection_matrix_ = Transformations::perspectiveMatrix(K, width, height, zNear, zFar, true);

	makeCurrent();

	initializeOpenGLFunctions();

	// FIX FOR NEW OPENGL
	uint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	K44s_.clear();

	for (int i = 0; i < max_levels_; i++)
	{
		float s = pow(2, i);

		Matx44f K_l = Matx44f::eye();
		K_l(0, 0) = K(0, 0) / s;
		K_l(1, 1) = K(1, 1) / s;
		K_l(0, 2) = K(0, 2) / s;
		K_l(1, 2) = K(1, 2) / s;

		K44s_.push_back(K_l);
	}

	// cout << "GL Version " << glGetString(GL_VERSION) << endl << "GLSL Version " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

	glEnable(GL_DEPTH);
	glEnable(GL_DEPTH_TEST);

	// INVERT DEPTH BUFFER
	glDepthRange(1, 0);
	glClearDepth(0.0f);
	glDepthFunc(GL_GREATER);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glClearColor(0.0, 0.0, 0.0, 1.0);

	initRenderingBuffers();

	shaderFolder = "src/";

	initShaderProgramFromCode(silhouetteShaderProgram, silhouette_vertex_shader, silhouette_fragment_shader);
	initShaderProgramFromCode(phongblinnShaderProgram, phongblinn_vertex_shader, phongblinn_fragment_shader);
	initShaderProgramFromCode(normalsShaderProgram, normals_vertex_shader, normals_fragment_shader);

	angle = 0;

	lightPosition = cv::Vec3f(0, 0, 0);

	// doneCurrent();
}

void Renderer::setLevel(int level)
{
	level_ = level;
	int s = pow(2, level_);
	width_ = full_width_ / s;
	height_ = full_height_ / s;
	width_ += width_ % 4;
	height_ += height_ % 4;
}

bool Renderer::initRenderingBuffers()
{
	glGenTextures(1, &colorTextureID);
	glBindTexture(GL_TEXTURE_2D, colorTextureID);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glGenTextures(1, &depthTextureID);
	glBindTexture(GL_TEXTURE_2D, depthTextureID);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width_, height_, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glGenFramebuffers(1, &frameBufferID);
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTextureID, 0);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTextureID, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		cout << "error creating rendering buffers" << endl;
		return false;
	}
	return true;
}

bool Renderer::initShaderProgramFromCode(QOpenGLShaderProgram *program, char *vertex_shader, char *fragment_shader)
{
	if (!program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertex_shader))
	{
		cout << "error adding vertex shader from source file" << endl;
		return false;
	}
	if (!program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragment_shader))
	{
		cout << "error adding fragment shader from source file" << endl;
		return false;
	}

	if (!program->link())
	{
		cout << "error linking shaders" << endl;
		return false;
	}
	return true;
}

void Renderer::OptimizeProjectionMatrix(const cv::Matx44f &pose, const float &bbxDiagLen)
{
	cv::Vec3f object_center(pose(0, 3), pose(1, 3), pose(2, 3));

	float distance_to_camera = cv::norm(object_center);

	float half_diag_len = bbxDiagLen / 2.0f;

	float lambda = 2.0f;
	zn_ = std::max(0.1f, distance_to_camera - half_diag_len * lambda);
	zf_ = distance_to_camera + half_diag_len * lambda;

	projection_matrix_ = Transformations::perspectiveMatrix(K33(level_), width_, height_, zn_, zf_, true);
	return;
}

void Renderer::ChangeK44s(std::vector<cv::Matx44f> Ks)
{
	K44s_ = std::vector<cv::Matx44f>(Ks);
	;
	return;
}

void Renderer::ChangeK44s(const cv::Matx33f &K, const float &zn, const float &zf)
{
	projection_matrix_ = Transformations::perspectiveMatrix(K, width_, height_, zn, zf, true);
	VLOG(30) << "width_ height_ " << width_ << " " << height_ << " " << " " << zn << " " << zf << " " << this->projection_matrix_;
	K44s_.clear();
	cv::Matx44f K44 = cv::Matx44f::eye();
	for (size_t i = 0; i < 4; i++)
	{
		K44(0, 0) = K(0, 0) / pow(2.0, float(i));
		K44(0, 2) = K(0, 2) / pow(2.0, float(i));
		K44(1, 1) = K(1, 1) / pow(2.0, float(i));
		K44(1, 2) = K(1, 2) / pow(2.0, float(i));
		K44s_.push_back(K44);
	}
	return;
}

void Renderer::Project(const cv::Matx44f &mv_mat, std::vector<cv::Vec3f> &model_points, std::vector<cv::Vec2f> &image_points)
{
	for (int i = 0; i < model_points.size(); ++i)
	{
		cv::Matx44f kmat = Renderer::K44s()[level_];
		cv::Vec4f ptv = kmat * mv_mat * cv::Vec4f(model_points[i](0), model_points[i](1), model_points[i](2), 1);

		float dz = 1.0f / ptv(2);
		image_points[i](0) = ptv(0) * dz;
		image_points[i](1) = ptv(1) * dz;
	}
}

static bool PtInFrame(const cv::Vec2f &pt, int width, int height)
{
	return (pt(0) < width && pt(1) < height && pt(0) >= 0 && pt(1) >= 0);
}

void Renderer::RenderCV(Model *model, cv::Mat &frame, cv::Scalar color)
{
	std::vector<cv::Vec3f> &model_points = model->vertices;
	std::vector<GLuint> &indices = model->indices;

	std::vector<cv::Vec2f> image_points(model_points.size());
	Project(model->getPose(), model_points, image_points);

	cv::Mat buf(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < indices.size(); i += 3)
	{
		cv::Vec2f pt0 = image_points[indices[i]];
		cv::Vec2f pt1 = image_points[indices[i + 1]];
		cv::Vec2f pt2 = image_points[indices[i + 2]];

		cv::line(buf, cv::Point(pt0(0), pt0(1)), cv::Point(pt1(0), pt1(1)), color, 2);
		cv::line(buf, cv::Point(pt1(0), pt1(1)), cv::Point(pt2(0), pt2(1)), color, 2);
		cv::line(buf, cv::Point(pt2(0), pt2(1)), cv::Point(pt0(0), pt0(1)), color, 2);
	}

	for (int r = 0; r < buf.rows; ++r)
		for (int c = 0; c < buf.cols; ++c)
		{
			float alpha = 0.5f;
			if (buf.at<cv::Vec3b>(r, c)[0])
			{
				cv::Vec3b &vf = frame.at<cv::Vec3b>(r, c);
				cv::Vec3b &vb = buf.at<cv::Vec3b>(r, c);
				frame.at<cv::Vec3b>(r, c) = (1.0f - alpha) * vb + alpha * vf;
			}
		}
}

void Renderer::RenderSilhouette(Model *model, GLenum polyonMode, bool invertDepth, float r, float g, float b, bool drawAll)
{
	vector<Model *> models;
	models.push_back(model);

	vector<Point3f> colors;
	colors.push_back(Point3f(r, g, b));

	RenderSilhouette(models, polyonMode, invertDepth, colors, drawAll);
}

cv::Mat Renderer::DrawResultOverlay(const std::vector<Model *> &objects, const cv::Mat &frame)
{
	// render the models with phong shading

	Renderer::Instance()->setLevel(0);

	std::vector<cv::Point3f> colors;
	colors.push_back(cv::Point3f(1.0, 0.5, 0.0));
	// Renderer::Instance()->RenderSilhouette(std::vector<Model*>(objects.begin(), objects.end()), GL_FILL, false, colors, true);
	Renderer::Instance()->RenderShaded(std::vector<Model *>(objects.begin(), objects.end()), GL_FILL, colors, true);
	// RenderNormals(std::vector<Model*>(objects.begin(), objects.end()), GL_FILL);

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
				result.at<cv::Vec3b>(y, x)[2] = 0.6 * result.at<cv::Vec3b>(y, x)[0] + 0.4 * color[2];
				result.at<cv::Vec3b>(y, x)[1] = 0.6 * result.at<cv::Vec3b>(y, x)[1] + 0.4 * color[1];
				result.at<cv::Vec3b>(y, x)[0] = 0.6 * result.at<cv::Vec3b>(y, x)[2] + 0.4 * color[0];
			}
		}
	}
	return result;
}

cv::Mat Renderer::DrawMeshOverlay(const std::vector<Model *> &objects, const cv::Mat &frame)
{
	setLevel(0);

	std::vector<cv::Point3f> colors;
	colors.push_back(cv::Point3f(1.0, 0.5, 0.0));
	colors.push_back(Point3f(0.2, 0.3, 1.0));
	cv::Mat result = frame.clone();
	RenderCV(objects[0], result);
	return result;
}

cv::Mat Renderer::DrawResultOverlay(const std::vector<Model *> &objects, const cv::Mat &frame, const bool &success)
{
	// render the models with phong shading

	Renderer::Instance()->setLevel(0);

	std::vector<cv::Point3f> colors;
	colors.push_back(cv::Point3f(1.0, 0.5, 0.0));
	Renderer::Instance()->RenderSilhouette(std::vector<Model *>(objects.begin(), objects.end()), GL_LINE, false, colors, true);

#ifdef GLOBAL_DOWNLOAD
	cv::Mat rendering = Renderer::Instance()->DownloadFrame(Renderer::RGB);
	cv::Mat depth = Renderer::Instance()->DownloadFrame(Renderer::DEPTH);
#elif defined(LOCAL_DOWNLOAD)
	cv::Mat rendering = Renderer::Instance()->DownloadFrameFromROI(Renderer::RGB, objects[0]);
	cv::Mat depth = Renderer::Instance()->DownloadFrameFromROI(Renderer::DEPTH, objects[0]);
#elif defined(SAFE_DOWNLOAD)
	cv::Mat rendering = Renderer::Instance()->DownloadFrameSafety(Renderer::RGB, objects[0]);
	cv::Mat depth = Renderer::Instance()->DownloadFrameSafety(Renderer::DEPTH, objects[0]);
#endif

	cv::Rect roi = Renderer::Instance()->Compute2DROI(objects[0], frame.size(), 8);

	// compose the rendering with the current camera image for demo purposes (can be done more efficiently directly in OpenGL)
	cv::Mat result = frame.clone();
	for (int y = roi.y; y < roi.y + roi.height; y++)
	{
		for (int x = roi.x; x < roi.x + roi.width; x++)
		{
			cv::Vec3b color = rendering.at<cv::Vec3b>(y, x);
			if (depth.at<float>(y, x) != 0.0f)
			{
				if (success)
				{
					result.at<cv::Vec3b>(y, x)[0] = 0.5 * color[2] + 0.5 * result.at<cv::Vec3b>(y, x)[0];
					result.at<cv::Vec3b>(y, x)[1] = 0.5 * color[1] + 0.5 * result.at<cv::Vec3b>(y, x)[1];
					result.at<cv::Vec3b>(y, x)[2] = 0.5 * color[0] + 0.5 * result.at<cv::Vec3b>(y, x)[2];
				}
				else
				{
					result.at<cv::Vec3b>(y, x)[0] = color[0];
				}
			}
		}
	}
	return result;
}

cv::Mat Renderer::DrawResultOverlayProjector(const std::vector<Model *> &objects, const cv::Mat &frame, const bool &success)
{
	// render the models with phong shading

	Renderer::Instance()->setLevel(0);

	std::vector<cv::Point3f> colors;
	colors.push_back(cv::Point3f(1.0, 0.5, 0.0));
	colors.push_back(cv::Point3f(0.0, 0.5, 1.0));
	// Renderer::Instance()->RenderSilhouette(std::vector<Model*>(objects.begin(), objects.end()), GL_FILL, false, colors, true);
	Renderer::Instance()->RenderShaded(std::vector<Model *>(objects.begin(), objects.end()), GL_LINE, colors, true);

	// download the rendering to the CPU
	cv::Mat rendering = Renderer::Instance()->DownloadFrame(Renderer::RGB);
	return rendering;
}

cv::Mat Renderer::DrawMask(const std::vector<Model *> &objects)
{
	std::vector<cv::Point3f> colors;
	colors.push_back(cv::Point3f(1.0, 1.0, 1.0));
	Renderer::Instance()->setLevel(0);
	Renderer::Instance()->RenderSilhouette(std::vector<Model *>(objects.begin(), objects.end()), GL_FILL, false, colors, true);
	cv::Mat mask = Renderer::Instance()->DownloadFrame(Renderer::MASK);
	return mask;
}

void Renderer::ConvertMask(const cv::Mat &src_mask, cv::Mat &mask, uchar oid)
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

cv::Mat Renderer::DrawContourOverlay(const std::vector<Model *> &objects, const cv::Mat &frame)
{
	setLevel(0);
	RenderSilhouette(std::vector<Model *>(objects.begin(), objects.end()), GL_FILL);

	cv::Mat depth_map = DownloadFrame(Renderer::DEPTH);
	cv::Mat masks_map;
	if (objects.size() > 1)
	{
		masks_map = DownloadFrame(Renderer::MASK);
	}
	else
	{
		masks_map = depth_map;
	}

	cv::Mat result = frame.clone();

	for (int oid = 0; oid < objects.size(); oid++)
	{
		cv::Mat mask_map;
		ConvertMask(masks_map, mask_map, objects[oid]->getModelID());

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(mask_map, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		cv::Vec3b color;
		if (0 == oid)
			color = cv::Vec3b(0, 255, 0);
		if (1 == oid)
			color = cv::Vec3b(0, 0, 255);

		for (auto contour : contours)
			for (auto pt : contour)
			{
				result.at<cv::Vec3b>(pt) = color;
			}
	}

	return result;
}

void Renderer::RenderShaded(Model *model, GLenum polyonMode, float r, float g, float b, bool drawAll)
{
	vector<Model *> models;
	models.push_back(model);

	vector<Point3f> colors;
	colors.push_back(Point3f(r, g, b));

	RenderShaded(models, polyonMode, colors, drawAll);
}

void Renderer::RenderNormals(Model *model, GLenum polyonMode, bool drawAll)
{
	vector<Model *> models;
	models.push_back(model);

	RenderNormals(models, polyonMode, drawAll);
}

void Renderer::RenderSilhouette(vector<Model *> models, GLenum polyonMode, bool invertDepth, const std::vector<cv::Point3f> &colors, bool drawAll)
{
	glViewport(0, 0, width_, height_);

	if (invertDepth)
	{
		glClearDepth(1.0f);
		glDepthFunc(GL_LESS);
	}

	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	for (int i = 0; i < models.size(); i++)
	{
		Model *model = models[i];

		if (model->isInitialized() || drawAll)
		{
			Matx44f pose = model->getPose();
			Matx44f normalization = model->getNormalization();

			Matx44f modelViewMatrix = look_at_matrix_ * (pose * normalization);

			Matx44f modelViewProjectionMatrix = projection_matrix_ * modelViewMatrix;

			silhouetteShaderProgram->bind();
			silhouetteShaderProgram->setUniformValue("uMVPMatrix", QMatrix4x4(modelViewProjectionMatrix.val));
			silhouetteShaderProgram->setUniformValue("uAlpha", 1.0f);

			Point3f color;
			if (i < colors.size())
			{
				color = colors[i];
			}
			else
			{
				color = Point3f((float)(model->getModelID()) / 255.0f, 0.0f, 0.0f);
			}
			silhouetteShaderProgram->setUniformValue("uColor", QVector3D(color.x, color.y, color.z));

			glPolygonMode(GL_FRONT_AND_BACK, polyonMode);

			model->draw(silhouetteShaderProgram);
		}
	}

	glClearDepth(0.0f);
	glDepthFunc(GL_GREATER);

	glFinish();
}

void Renderer::RenderShaded(vector<Model *> models, GLenum polyonMode, const std::vector<cv::Point3f> &colors, bool drawAll)
{
	glViewport(0, 0, width_, height_);

	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	for (int i = 0; i < models.size(); i++)
	{
		Model *model = models[i];
		if (model->isInitialized() || drawAll)
		{
			Matx44f pose = model->getPose();
			Matx44f normalization = model->getNormalization();

			Matx44f modelViewMatrix = look_at_matrix_ * (pose * normalization);

			Matx33f normalMatrix = modelViewMatrix.get_minor<3, 3>(0, 0).inv().t();

			Matx44f modelViewProjectionMatrix = projection_matrix_ * modelViewMatrix;
			// VLOG(0) << "lookat  " << normalization << std::endl << "project  "<<projection_matrix_ << std::endl;
			phongblinnShaderProgram->bind();
			phongblinnShaderProgram->setUniformValue("uMVMatrix", QMatrix4x4(modelViewMatrix.val));
			phongblinnShaderProgram->setUniformValue("uMVPMatrix", QMatrix4x4(modelViewProjectionMatrix.val));
			phongblinnShaderProgram->setUniformValue("uNormalMatrix", QMatrix3x3(normalMatrix.val));
			phongblinnShaderProgram->setUniformValue("uLightPosition1", QVector3D(0.1, 0.1, -0.02));
			phongblinnShaderProgram->setUniformValue("uLightPosition2", QVector3D(-0.1, 0.1, -0.02));
			phongblinnShaderProgram->setUniformValue("uLightPosition3", QVector3D(0.0, 0.0, 0.1));
			phongblinnShaderProgram->setUniformValue("uShininess", 100.0f);
			phongblinnShaderProgram->setUniformValue("uAlpha", 1.0f);

			Point3f color;
			if (i < colors.size())
			{
				color = colors[i];
			}
			else
			{
				color = Point3f(1.0, 0.5, 0.0);
			}
			phongblinnShaderProgram->setUniformValue("uColor", QVector3D(color.x, color.y, color.z));

			glPolygonMode(GL_FRONT_AND_BACK, polyonMode);

			model->draw(phongblinnShaderProgram);
		}
	}

	glFinish();
}

void Renderer::RenderNormals(vector<Model *> models, GLenum polyonMode, bool drawAll)
{
	glViewport(0, 0, width_, height_);

	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	for (int i = 0; i < models.size(); i++)
	{
		Model *model = models[i];

		if (model->isInitialized() || drawAll)
		{
			Matx44f pose = model->getPose();
			Matx44f normalization = model->getNormalization();

			Matx44f modelViewMatrix = look_at_matrix_ * (pose * normalization);

			Matx33f normalMatrix = modelViewMatrix.get_minor<3, 3>(0, 0).inv().t();

			Matx44f modelViewProjectionMatrix = projection_matrix_ * modelViewMatrix;

			normalsShaderProgram->bind();
			normalsShaderProgram->setUniformValue("uMVMatrix", QMatrix4x4(modelViewMatrix.val));
			normalsShaderProgram->setUniformValue("uMVPMatrix", QMatrix4x4(modelViewProjectionMatrix.val));
			normalsShaderProgram->setUniformValue("uNormalMatrix", QMatrix3x3(normalMatrix.val));
			normalsShaderProgram->setUniformValue("uAlpha", 1.0f);

			glPolygonMode(GL_FRONT_AND_BACK, polyonMode);

			model->draw(normalsShaderProgram);
		}
	}

	glFinish();
}

void Renderer::ProjectBoundingBox(Model *model, std::vector<cv::Point2f> &projections, cv::Matx44f &pose, cv::Rect &boundingRect)
{
	Vec3f lbn = model->getLBN();
	Vec3f rtf = model->getRTF();

	Vec4f Plbn = Vec4f(lbn[0], lbn[1], lbn[2], 1.0);
	Vec4f Prbn = Vec4f(rtf[0], lbn[1], lbn[2], 1.0);
	Vec4f Pltn = Vec4f(lbn[0], rtf[1], lbn[2], 1.0);
	Vec4f Plbf = Vec4f(lbn[0], lbn[1], rtf[2], 1.0);
	Vec4f Pltf = Vec4f(lbn[0], rtf[1], rtf[2], 1.0);
	Vec4f Prtn = Vec4f(rtf[0], rtf[1], lbn[2], 1.0);
	Vec4f Prbf = Vec4f(rtf[0], lbn[1], rtf[2], 1.0);
	Vec4f Prtf = Vec4f(rtf[0], rtf[1], rtf[2], 1.0);

	vector<Vec4f> points3D;
	points3D.push_back(Plbn);
	points3D.push_back(Prbn);
	points3D.push_back(Pltn);
	points3D.push_back(Plbf);
	points3D.push_back(Pltf);
	points3D.push_back(Prtn);
	points3D.push_back(Prbf);
	points3D.push_back(Prtf);

	Matx44f normalization = model->getNormalization();

	Point2f lt(FLT_MAX, FLT_MAX);
	Point2f rb(-FLT_MAX, -FLT_MAX);
	m_2D_bbx.clear();

	for (int i = 0; i < points3D.size(); i++)
	{
		Vec4f p = K44s_[level_] * pose * normalization * points3D[i];

		if (p[2] == 0)
			continue;

		Point2f p2d = Point2f(p[0] / p[2], p[1] / p[2]);
		projections.push_back(p2d);
		m_2D_bbx.push_back(p2d);

		if (p2d.x < lt.x)
			lt.x = p2d.x;
		if (p2d.x > rb.x)
			rb.x = p2d.x;
		if (p2d.y < lt.y)
			lt.y = p2d.y;
		if (p2d.y > rb.y)
			rb.y = p2d.y;
	}

	boundingRect.x = lt.x;
	boundingRect.y = lt.y;
	boundingRect.width = rb.x - lt.x;
	boundingRect.height = rb.y - lt.y;
}

void Renderer::ProjectBoundingBox(Model *model, std::vector<cv::Point2f> &projections, cv::Rect &boundingRect)
{
	Vec3f lbn = model->getLBN();
	Vec3f rtf = model->getRTF();

	Vec4f Plbn = Vec4f(lbn[0], lbn[1], lbn[2], 1.0);
	Vec4f Prbn = Vec4f(rtf[0], lbn[1], lbn[2], 1.0);
	Vec4f Pltn = Vec4f(lbn[0], rtf[1], lbn[2], 1.0);
	Vec4f Plbf = Vec4f(lbn[0], lbn[1], rtf[2], 1.0);
	Vec4f Pltf = Vec4f(lbn[0], rtf[1], rtf[2], 1.0);
	Vec4f Prtn = Vec4f(rtf[0], rtf[1], lbn[2], 1.0);
	Vec4f Prbf = Vec4f(rtf[0], lbn[1], rtf[2], 1.0);
	Vec4f Prtf = Vec4f(rtf[0], rtf[1], rtf[2], 1.0);

	vector<Vec4f> points3D;
	points3D.push_back(Plbn);
	points3D.push_back(Prbn);
	points3D.push_back(Pltn);
	points3D.push_back(Plbf);
	points3D.push_back(Pltf);
	points3D.push_back(Prtn);
	points3D.push_back(Prbf);
	points3D.push_back(Prtf);

	Matx44f pose = model->getPose();
	Matx44f normalization = model->getNormalization();

	Point2f lt(FLT_MAX, FLT_MAX);
	Point2f rb(-FLT_MAX, -FLT_MAX);
	std::vector<cv::Point2f>().swap(m_2D_bbx);

	for (int i = 0; i < points3D.size(); i++)
	{
		Vec4f p = K44s_[level_] * pose * normalization * points3D[i];
		if (p[2] == 0)
			continue;

		Point2f p2d = Point2f(p[0] / p[2], p[1] / p[2]);
		projections.push_back(p2d);
		m_2D_bbx.push_back(p2d);

		if (p2d.x < lt.x)
			lt.x = p2d.x;
		if (p2d.x > rb.x)
			rb.x = p2d.x;
		if (p2d.y < lt.y)
			lt.y = p2d.y;
		if (p2d.y > rb.y)
			rb.y = p2d.y;
	}

	boundingRect.x = lt.x;
	boundingRect.y = lt.y;
	boundingRect.width = rb.x - lt.x;
	boundingRect.height = rb.y - lt.y;
}

bool Renderer::DrawLines(const cv::Mat &img, const cv::Point2f &p1, const cv::Point2f &p2, const cv::Scalar &scalar, const bool &arrow)
{
	if (p1.y > 0 && p1.y < img.rows && p1.x > 0 && p1.x < img.cols && p2.y > 0 && p2.y < img.rows && p2.x > 0 && p2.x < img.cols)
	{
		if (arrow)
			cv::arrowedLine(img, p1, p2, scalar, 4, cv::LINE_AA, 0, 0.2);
		else
			cv::line(img, p1, p2, scalar, 1, cv::LINE_AA);
		return true;
	}
	return false;
}

bool Renderer::DrawBBX(cv::Mat &img, Model *model)
{

	std::vector<cv::Point2f> projections;
	cv::Rect rect;
	Renderer::Instance()->ProjectBoundingBox(model, projections, rect);

	if (m_2D_bbx.size() == 0)
		return false;
	for (int i = 0; i < m_2D_bbx.size(); i++)
	{
		Point2f p = m_2D_bbx[i];
		if (p.x > 0 && p.x < img.rows && p.y > 0 && p.y < img.cols)
		{
			// cv::circle(m_img, p, 3, Scalar(0, 0, 255), 2);
		}
	}

	Point2f p0 = m_2D_bbx[0];
	Point2f p1 = m_2D_bbx[1];
	Point2f p2 = m_2D_bbx[2];
	Point2f p3 = m_2D_bbx[3];
	Point2f p4 = m_2D_bbx[4];
	Point2f p5 = m_2D_bbx[5];
	Point2f p6 = m_2D_bbx[6];
	Point2f p7 = m_2D_bbx[7];

	Scalar scalar_0(0, 0, 255);
	Scalar scalar_1(0, 255, 0);
	Scalar scalar_2(255, 0, 0);
	Scalar scalar_3(255, 255, 0);
	Scalar scalar_4(0, 255, 255);

	DrawLines(img, p0, p1, scalar_4);
	DrawLines(img, p0, p2, scalar_4);
	DrawLines(img, p0, p3, scalar_4);
	DrawLines(img, p1, p5, scalar_4);
	DrawLines(img, p1, p6, scalar_4);
	DrawLines(img, p2, p4, scalar_4);
	DrawLines(img, p2, p5, scalar_4);
	DrawLines(img, p3, p4, scalar_4);
	DrawLines(img, p3, p6, scalar_4);
	DrawLines(img, p7, p4, scalar_4);
	DrawLines(img, p7, p5, scalar_4);
	DrawLines(img, p7, p6, scalar_4);

	return true;
}

bool Renderer::DrawCenter(cv::Mat &frame, Model *model)
{
	cv::Vec2f center2d;
	Matx44f pose = model->getPose();
	Matx44f normalization = model->getNormalization();
	cv::Vec4f center3d(m_center_3d[0], m_center_3d[1], m_center_3d[2], 1.0f);
	Vec4f p = K44s_[level_] * pose * normalization * center3d;
	Point2f p2d = Point2f(p[0] / p[2], p[1] / p[2]);
	cv::circle(frame, p2d, 2, cv::Scalar(0, 0, 255), 2, 8);
	return true;
}

cv::Vec2f Renderer::Get2DCenter(Model *model)
{
	Matx44f pose = model->getPose();
	Matx44f normalization = model->getNormalization();
	cv::Vec4f center3d(m_center_3d[0], m_center_3d[1], m_center_3d[2], 1.0f);
	Vec4f p = K44s_[level_] * pose * normalization * center3d;
	Point2f p2d = Point2f(p[0] / p[2], p[1] / p[2]);
	return cv::Vec2f(p[0] / p[2], p[1] / p[2]);
}

cv::Rect Renderer::RoiByMask()
{
	cv::Mat mask = this->DownloadFrame(Renderer::MASK);
	if (mask.empty())
		LOG(ERROR) << "Dowload Null Mask in Actual2dBBX()";
	cv::Mat bin;
	cv::threshold(mask, bin, 0, 255, cv::THRESH_BINARY);
	int left, right, top, down;

	for (int i = 0; i < bin.rows; i++)
	{
		if (cv::countNonZero(bin.row(i)) != 0)
		{
			top = i;
			break;
		}
	}
	for (int i = bin.rows - 1; i > 0; i--)
	{
		if (cv::countNonZero(bin.row(i)) != 0)
		{
			down = i;
			break;
		}
	}
	for (int i = 0; i < bin.cols; i++)
	{
		if (cv::countNonZero(bin.col(i)) != 0)
		{
			left = i;
			break;
		}
	}
	for (int i = bin.cols - 1; i > 0; i--)
	{
		if (cv::countNonZero(bin.col(i)) != 0)
		{
			right = i;
			break;
		}
	}

	cv::Rect roi(left, top, right - left, down - top);
	return roi;
}

cv::Rect Renderer::RoiByMask(const cv::Mat &mask)
{
	if (mask.empty())
		LOG(ERROR) << "Dowload Null Mask in Actual2dBBX()";
	cv::Mat bin;
	cv::threshold(mask, bin, 0, 255, cv::THRESH_BINARY);
	int left, right, top, down;

	for (int i = 0; i < bin.rows; i++)
	{
		if (cv::countNonZero(bin.row(i)) != 0)
		{
			top = i;
			break;
		}
	}
	for (int i = bin.rows - 1; i > 0; i--)
	{
		if (cv::countNonZero(bin.row(i)) != 0)
		{
			down = i;
			break;
		}
	}
	for (int i = 0; i < bin.cols; i++)
	{
		if (cv::countNonZero(bin.col(i)) != 0)
		{
			left = i;
			break;
		}
	}
	for (int i = bin.cols - 1; i > 0; i--)
	{
		if (cv::countNonZero(bin.col(i)) != 0)
		{
			right = i;
			break;
		}
	}

	cv::Rect roi(left, top, right - left, down - top);
	return roi;
}

cv::Rect Renderer::Compute2DROI(Model *model, const cv::Size &maxSize, int offset)
{
	// PROJECT THE 3D BOUNDING BOX AS 2D ROI
	cv::Rect boundingRect;
	std::vector<cv::Point2f> projections;

	ProjectBoundingBox(model, projections, boundingRect);

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

Mat Renderer::DownloadMask()
{
	int safeX{0}, safeY{0};
	int safeWidth = width_ - width_ % 8;
	int safeHeight = height_ - height_ % 8;
	cv::Rect copyRange(safeX, safeY, safeWidth, safeHeight);

	Mat res, roiImg;
	res = cv::Mat::zeros(height_, width_, CV_8UC1);
	roiImg = cv::Mat::zeros(safeHeight, safeWidth, CV_8UC1);
	glReadPixels(safeX, safeY, safeWidth, safeHeight, GL_RED, GL_UNSIGNED_BYTE, roiImg.data);
	roiImg.copyTo(res(copyRange));
	return res;
}

Mat Renderer::DownloadFrame(Renderer::FrameType type)
{
	int safeX{0}, safeY{0};
	int safeWidth = width_ - width_ % 8;
	int safeHeight = height_ - height_ % 8;
	cv::Rect copyRange(safeX, safeY, safeWidth, safeHeight);

	Mat res, roiImg;
	switch (type)
	{
	case MASK:
	{
		// res = Mat(height_, width_, CV_8UC1);
		// glReadPixels(0, 0, res.cols, res.rows, GL_RED, GL_UNSIGNED_BYTE, res.data);
		// break;
		res = cv::Mat::zeros(height_, width_, CV_8UC1);
		roiImg = cv::Mat::zeros(safeHeight, safeWidth, CV_8UC1);
		glReadPixels(safeX, safeY, safeWidth, safeHeight, GL_RED, GL_UNSIGNED_BYTE, roiImg.data);
		roiImg.copyTo(res(copyRange));
		break;
	}
	case RGB:
		res = Mat(height_, width_, CV_8UC3);
		glReadPixels(0, 0, res.cols, res.rows, GL_RGB, GL_UNSIGNED_BYTE, res.data);
		break;
	case RGB_32F:
		res = Mat(height_, width_, CV_32FC3);
		glReadPixels(0, 0, res.cols, res.rows, GL_RGB, GL_FLOAT, res.data);
		break;
	case DEPTH:
	{
		// res = Mat(height_, width_, CV_32FC1);
		// glReadPixels(0, 0, res.cols, res.rows, GL_DEPTH_COMPONENT, GL_FLOAT, res.data);
		// break;
		res = cv::Mat::zeros(height_, width_, CV_32FC1);
		roiImg = cv::Mat::zeros(safeHeight, safeWidth, CV_32FC1);
		glReadPixels(safeX, safeY, safeWidth, safeHeight, GL_DEPTH_COMPONENT, GL_FLOAT, roiImg.data);
		roiImg.copyTo(res(copyRange));
		break;
	}
	default:
		res = Mat::zeros(height_, width_, CV_8UC1);
		break;
	}

	return res;
}

void Renderer::MakeSureRoiSafety(cv::Rect &roi)
{
	// Adjust ROI to ensure width and height are multiples of 4
	if (roi.width % 4 != 0)
	{
		roi.width -= roi.width % 4;
	}
	if (roi.height % 4 != 0)
	{
		roi.height -= roi.height % 4;
	}

	// Ensure ROI is within image bounds
	roi.width = std::min(roi.width, width_ - roi.x);
	roi.height = std::min(roi.height, height_ - roi.y);

	// If the ROI is still out of bounds, adjust position
	if (roi.x + roi.width > width_)
	{
		roi.x = width_ - roi.width;
	}
	if (roi.y + roi.height > height_)
	{
		roi.y = height_ - roi.height;
	}
}

cv::Mat Renderer::DownloadFrameFromROI(Renderer::FrameType type, Model *model)
{
	cv::Rect roi = Compute2DROI(model, cv::Size(width_, height_), 8);

	cv::Mat res;
	switch (type)
	{
	case MASK:
	{
		res = cv::Mat::zeros(height_, width_, CV_8UC1);
		glReadPixels(0, 0, res.cols, res.rows, GL_RED, GL_UNSIGNED_BYTE, res.data);
		break;
	}
	case RGB:
	{
		res = Mat(height_, width_, CV_8UC3);
		MakeSureRoiSafety(roi);
		cv::Mat roiImg = cv::Mat::zeros(roi.height, roi.width, CV_8UC3);
		glReadPixels(roi.x, roi.y, roi.width, roi.height, GL_RGB, GL_UNSIGNED_BYTE, roiImg.data);
		if (roi.area() != 0)
			roiImg.copyTo(res(roi));
		else
			glReadPixels(0, 0, res.cols, res.rows, GL_RGB, GL_UNSIGNED_BYTE, res.data);
		break;
	}
	case RGB_32F:
		res = Mat(height_, width_, CV_32FC3);
		glReadPixels(0, 0, res.cols, res.rows, GL_RGB, GL_FLOAT, res.data);
		break;
	case DEPTH:
	{
		res = cv::Mat::zeros(height_, width_, CV_32FC1);
		cv::Mat roiImg = cv::Mat::zeros(roi.height, roi.width, CV_32FC1);
		glReadPixels(roi.x, roi.y, roi.width, roi.height, GL_DEPTH_COMPONENT, GL_FLOAT, roiImg.data);
		if (roi.area() != 0)
			roiImg.copyTo(res(roi));
		else
			glReadPixels(0, 0, res.cols, res.rows, GL_DEPTH_COMPONENT, GL_FLOAT, res.data);
		break;
	}
	default:
		res = Mat::zeros(height_, width_, CV_8UC1);
		break;
	}
	return res;
}

cv::Mat Renderer::DownloadFrameSafety(Renderer::FrameType type, Model *model)
{
	int safeX{0}, safeY{0};
	int safeWidth = width_ - width_ % 8;
	int safeHeight = height_ - height_ % 8;

	int frameCenterX = width_ / 2;
	int frameCenterY = height_ / 2;

	cv::Rect roi = Compute2DROI(model, cv::Size(width_, height_), 8);
	int roiCenterX = (roi.x + roi.width) / 2;
	int roiCenterY = (roi.y + roi.height) / 2;
	if (roiCenterX > frameCenterX)
		safeX = width_ % 8;
	if (roiCenterY > frameCenterY)
		safeY = height_ % 8;

	cv::Mat res, roiImg;
	cv::Rect copyRange(safeX, safeY, safeWidth, safeHeight);
	switch (type)
	{
	case MASK:
	{
		res = cv::Mat::zeros(height_, width_, CV_8UC1);
		roiImg = cv::Mat::zeros(safeHeight, safeWidth, CV_8UC1);
		glReadPixels(safeX, safeY, safeWidth, safeHeight, GL_RED, GL_UNSIGNED_BYTE, roiImg.data);
		roiImg.copyTo(res(copyRange));
		break;
	}
	case RGB:
	{
		res = Mat(height_, width_, CV_8UC3);
		roiImg = cv::Mat::zeros(safeHeight, safeWidth, CV_8UC3);
		glReadPixels(safeX, safeY, safeWidth, safeHeight, GL_RGB, GL_UNSIGNED_BYTE, roiImg.data);
		roiImg.copyTo(res(copyRange));
		break;
	}
	case RGB_32F:
		res = Mat(height_, width_, CV_32FC3);
		roiImg = cv::Mat::zeros(safeHeight, safeWidth, CV_32FC3);
		glReadPixels(safeX, safeY, safeWidth, safeHeight, GL_RGB, GL_FLOAT, res.data);
		roiImg.copyTo(res(copyRange));
		break;
	case DEPTH:
	{
		res = cv::Mat::zeros(height_, width_, CV_32FC1);
		roiImg = cv::Mat::zeros(safeHeight, safeWidth, CV_32FC1);
		glReadPixels(safeX, safeY, safeWidth, safeHeight, GL_DEPTH_COMPONENT, GL_FLOAT, roiImg.data);
		roiImg.copyTo(res(copyRange));
		break;
	}
	default:
		res = Mat::zeros(height_, width_, CV_8UC1);
		break;
	}
	return res;
}

void Renderer::ProjectPoints(const std::vector<cv::Point3f> &pts3d, const cv::Matx44f &pose, std::vector<cv::Point2f> &pts)
{
	pts.clear();

	cv::Matx44f K = {
		386.52, 0, 326.51, 0,
		0, 387.32, 237.40, 0,
		0, 0, 1, 0,
		0, 0, 0, 1};
	for (int i = 0; i < pts3d.size(); i++)
	{
		// Vec4f p = K44s_[level_] * pose * cv::Vec4f(pts3d[i].x, pts3d[i].y, pts3d[i].z, 1.0f);
		Vec4f p = K * pose * cv::Vec4f(pts3d[i].x, pts3d[i].y, pts3d[i].z, 1.0f);

		float x = p[0] / p[2];
		float y = p[1] / p[2];

		// if (x >= 0 && x < width_ && y >= 0 && y < height_) {
		pts.push_back(cv::Point2f(x, y));
		// }
	}
}

void Renderer::BackProjectPoints(std::vector<cv::Point> &pts, const cv::Mat &depth_map, const cv::Matx44f &pose, std::vector<cv::Point3f> &pts3d)
{
	float *depthData = (float *)depth_map.ptr<float>();
	cv::Matx44f k44 = K44s()[level_];
	const cv::Matx33f &K = K44s()[level_].get_minor<3, 3>(0, 0);
	float *K_invData = K.inv().val;
	float *pdata = pose.inv().val;

	pts3d.resize(pts.size());

	for (int i = 0; i < pts3d.size(); ++i)
	{
		int zidx = pts[i].y * depth_map.cols + pts[i].x;
		float depth = 1.0f - depthData[zidx];

		float D = 2.0f * zn_ * zf_ / (zf_ + zn_ - (2.0f * depth - 1.0) * (zf_ - zn_));

		float X = D * (K_invData[0] * pts[i].x + K_invData[2]);
		float Y = D * (K_invData[4] * pts[i].y + K_invData[5]);
		float Z = D;

		pts3d[i] = cv::Point3f(
			pdata[0] * X + pdata[1] * Y + pdata[2] * Z + pdata[3],
			pdata[4] * X + pdata[5] * Y + pdata[6] * Z + pdata[7],
			pdata[8] * X + pdata[9] * Y + pdata[10] * Z + pdata[11]);
	}
}

std::vector<cv::Vec2f> Renderer::Project3DPoints(Model *model, const std::vector<cv::Vec3f> &points3D)
{
	// cv::Mat frameCopy = frame.clone();
	Matx44f pose = model->getPose();
	Matx44f normalization = model->getNormalization();
	std::vector<cv::Vec2f> p2ds;

	for (int i = 0; i < points3D.size(); i++)
	{
		Vec4f points3DHomo(points3D[i][0], points3D[i][1], points3D[i][2], 1);
		Vec4f p = K44s_[level_] * pose * normalization * points3DHomo;
		Vec2f p2d(p[0] / p[2], p[1] / p[2]);
		p2ds.push_back(p2d);
	}
	return p2ds;
}

// } // ns summer