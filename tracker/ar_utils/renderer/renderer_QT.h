#ifndef AR3DVENGINE_RENDERER_QT_H
#define AR3DVENGINE_RENDERER_QT_H

#include <iostream>

#include <QApplication>
#include <QOpenGLContext>
#include <QOffscreenSurface>
#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include <QGLFramebufferObject>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLTexture>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "ar_utils/model/transformations.h"
#include "ar_utils/model/model.h"

#include "glog/logging.h"

namespace ar3dv{

/**
 *  This class implements an OpenGL-based offscreen rendering engine for generating
 *  images of projected 3D meshes based on given object poses and camera instrinsics.
 *  It supports one or mutiple objects to be rendered as binary masks, depth maps,
 *  normal maps or phong-shaded. It also allows to perform all renderings according
 *  to a specified image pyramid level at lower resolutions. The class is  implemented
 *  as a singleton.
 */
    class RenderingEngineQT : public QOpenGLFunctions_3_3_Core
    {
    public:
        enum FrameType {
			ALPHA,
            MASK,
            RGB,
			RGBA,
            RGB_32F,
            DEPTH
        };

        RenderingEngineQT(void);

        ~RenderingEngineQT(void);

        static RenderingEngineQT *Instance(void)
        {
            if (instance == NULL) instance = new RenderingEngineQT();
            return instance;
        }

        /**
         *  The 3d data is packed into VOBs and uploaded to the GPU.
         *  Should be called after a valid OpenGL context exists.
         *  Must be called before a model can get rendered!
         */
        void initBuffers(Model* model);

        /**
         *  Initializes the rendering engine instance given a 3x3 float
         *  intrinsic camera matrix
         *  K = [fx 0 cx]
         *      [0 fy cy]
         *      [0  0  1],
         *  a desired image resolution, a near and a far plane as well as the number
         *  of pyramidf level supported for the renderings.
         *
         *  @param  K The intrinsic camera matrix.
         *  @param  width The width in pixels of the rendered images at level 0.
         *  @param  height The height in pixels of the rendered images at level 0.
         *  @param  zNear The distance of the OpenGL near plane.
         *  @param  zFar The distance of the OpenGL far plane.
         *  @param  numLevels Number of supported pyramid levels with a downscale factor of 2.
         *  @param  glslPath glslPaths.
         */
        void init(const cv::Matx33f &K, int width, int height, float zNear, float zFar,
                  int numLevels, std::string glslPath);

        /**
         *  Load the material of the model.
         */
        void loadMaterials(const std::string texPath);

        /**
         *  Create a texture object.
         *  @param  path The relative path of the texture image.
         */
        QOpenGLTexture* loadTexture(const QString& path);

        /**
         *  Returns the number of supported pyramid levels for rendering.
         *
         *  @return  The number of supported pyramid levels for rendering.
         */
        int getNumLevels();

        /**
         *  Sets a pyramid level to be used for rendering between 0 (full resolution)
         *  and getNumLevels() (the smallest resolution).
         *
         *  @param level The pyramid level to be used for rendering.
         */
        void setLevel(int level);

        /**
         *  Returns the current pyramid level used for rendering.
         *
         *  @return  The current pyramid level used for rendering.
         */
        int getLevel();

        /**
         *  Activates the OpenGL context of the rendering engine.
         */
        void makeCurrent();

        /**
         *  Deactivates the OpenGL context of the rendering engine.
         */
        void doneCurrent();

        /**
         *  Returns the OpenGL context of the rendering engine.
         *
         *  @return  The OpenGL context of the rendering engine.
         */
        QOpenGLContext *getContext();

        /**
         *  Returns the OpenGL ID of the frame buffer object used for offscreen rendering.
         *
         *  @return  The OpenGL ID of the frame buffer object used for offscreen rendering.
         */
        GLuint getFrameBufferID();

        /**
         *  Returns the OpenGL texture ID of the rendered color image.
         *
         *  @return  The OpenGL texture ID of the rendered color image.
         */
        GLuint getColorTextureID();

        /**
         *  Returns the OpenGL texture ID of the rendered depth buffer.
         *
         *  @return  The OpenGL texture ID of the rendered depth buffer.
         */
        GLuint getDepthTextureID();

        /**
         *  Returns the Z-distance of the near plane.
         *
         *  @return  The the Z-distance of the near plane.
         */
        float getZNear();

        /**
         *  Returns the Z-distance of the far plane.
         *
         *  @return  The the Z-distance of the far plane.
         */
        float getZFar();

        /**
         *  Returns a 4x4 float version of the intrinsic camera matrix wrt the current
         *  pyramid level
         *  K_4x4 = [fx/s 0 cx/s 0]
         *          [0 fy/s cy/s 0]
         *          [0  0    1   0]
         *          [0  0    0   1],
         *  with s = 1/2^level.
         *
         *  @return  A 4x4 float version of the intrinsic camera matrix wrt the current
         *  pyramid level.
         */
        cv::Matx44f getCalibrationMatrix();

		cv::Mat RenderBBX(const cv::Mat &img, Model *model);
		
        /**
         *  Renders a single model with a constant color and no shading in order to
         *  obtain a binary silhouette mask of it wrt its current pose.
         *
         *  @param model The model to be rendered.
         *  @param polyonMode The OpenGL polygon mode to be used (e.g. GL_FILL).
         *  @param invertDepth Whether to invert the depth test during rendering (default = false).
         *  @param r The red intensity of the model surface albedo in [0, 1] (default = 1.0).
         *  @param g The green intensity of the model surface albedo in [0, 1] (default = 1.0).
         *  @param b The blue intensity of the model surface albedo in [0, 1] (default = 1.0).
         *  @param drawAll Whether to draw the model even if it has not yet been initlaized for tracking (default = false).
         */
        void renderSilhouette(Model *model, GLenum polyonMode, bool invertDepth = false, float r = 1.0f, float g = 1.0f, float b = 1.0f, bool drawAll = false);

        /**
         *  Renders a single model wrt its current pose using Phong shading.
         *
         *  @param model The model to be rendered.
         *  @param polyonMode The OpenGL polygon mode to be used (e.g. GL_FILL).
         *  @param r The red intensity of the model surface albedo in [0, 1] (default = 1.0).
         *  @param g The green intensity of the model surface albedo in [0, 1] (default = 0.5).
         *  @param b The blue intensity of the model surface albedo in [0, 1] (default = 0.0).
         *  @param drawAll Whether to draw the model even if it has not yet been initlaized for tracking (default = false).
         */
        void renderShaded(Model *model, GLenum polyonMode, float r = 1.0f, float g = 0.5f, float b = 0.0f, bool drawAll = false);
			  
        void renderFeatureLine(Model *model, GLenum polyonMode, float r = 1.0f, float g = 0.5f, float b = 0.0f, bool drawAll = false);

        /**
         *  Renders the per pixel surface normals of single model wrt its current pose where the
         *  normal direction is mapped from (x, y, z) in [-1, 1] to (r, g, b) in [0, 1].
         *
         *  @param model The model to be rendered.
         *  @param polyonMode The OpenGL polygon mode to be used (e.g. GL_FILL).
         *  @param drawAll Whether to draw the model even if it has not yet been initlaized for tracking (default = false).
         */
        void renderNormals(Model *model, GLenum polyonMode, bool drawAll = false);

        /**
         *  Render the texture of the model
         *
         *  @param model The model to be rendered.
         *  @param polyonMode The OpenGL polygon mode to be used (e.g. GL_FILL).
         *  @param drawAll Whether to draw the model even if it has not yet been initlaized for tracking (default = false).
         */
        void renderTexture(Model* model, GLenum polyonMode, bool drawAll = false);

        /**
         *  Renders a multiple models in a common scene with a constant color and no shading
         *  in order to obtain a their binary silhouette masks with correct occlusions
         *  according to their current poses. If no colors are spefified each model will by default
         *  get rendered with a constant color corresponding to their model index in the red channel.
         *
         *  @param model The models to be rendered.
         *  @param polyonMode The OpenGL polygon mode to be used (e.g. GL_FILL).
         *  @param invertDepth Whether to invert the depth test during rendering (default = false).
         *  @param colors A vector of colors to be used for each model (default = empty).
         *  @param drawAll Whether to draw all models even if they been not yet initlaized for tracking (default = false).
         */
        void renderSilhouette(std::vector<Model*> models, GLenum polyonMode, bool invertDepth = false, const std::vector<cv::Point3f> &colors = std::vector<cv::Point3f>(), bool drawAll = false);

        /**
         *  Renders a multiple models in a common scene wrt their current poses using Phong shading.
         *
         *  @param model The models to be rendered.
         *  @param polyonMode The OpenGL polygon mode to be used (e.g. GL_FILL).
         *  @param colors A vector of colors to be used for each model (default = empty).
         *  @param drawAll Whether to draw the model even if it has not yet been initlaized for tracking (default = false).
         */
        void renderShaded(std::vector<Model*> models, GLenum polyonMode, const std::vector<cv::Point3f> &colors = std::vector<cv::Point3f>(), bool drawAll = false);
		
        void renderFeatureLine(std::vector<Model*> models, GLenum polyonMode, const std::vector<cv::Point3f> &colors = std::vector<cv::Point3f>(), bool drawAll = false);


        /**
         *  Renders the per pixel surface normals of multiple models in a common scene wrt their
         *  current poses where the normal direction is mapped from (x, y, z) in [-1, 1] to (r, g, b)
         *  in [0, 1].
         *
         *  @param model The model to be rendered.
         *  @param polyonMode The OpenGL polygon mode to be used (e.g. GL_FILL).
         *  @param drawAll Whether to draw the model even if it has not yet been initlaized for tracking (default = false).
         */
        void renderNormals(std::vector<Model*> models, GLenum polyonMode, bool drawAll = false);

        /**
         *  Render the texture of the model
         *
         *  @param model The model to be rendered.
         *  @param polyonMode The OpenGL polygon mode to be used (e.g. GL_FILL).
         *  @param drawAll Whether to draw the model even if it has not yet been initlaized for tracking (default = false).
         */
        void renderTexture(std::vector<Model*> models, GLenum polyonMode, bool drawAll = false);

        /**
         *  Draws the model with a given shader programm and
         *  a specified OpenGL data primitive type using VBOs.
         *
         *  @param  program    The shader programm to be used.
         *  @param  primitives The primitive type that shall be used for drawing (e.g. GL_POINTS, GL_LINES,...). The default value is set to GL_TRIANGLES.
         */
        void draw(QOpenGLShaderProgram *program, GLint primitives = GL_TRIANGLES);

        /**
         *  Downloads the most recently rendered image from the GPU to the host memory and converts
         *  it to an OpenCV image depending on a given frametype. Use MASK to obtain a silhouette
         *  mask image (single channel, uchar), RGB to obtain a color image (RGB, uchar), RGB_32F
         *  to obtain color image with normalized intensities in [0, 1] (RGB, float) or DEPTH to
         *  obtain the depth buffer.
         *
         *  @param type The frame type to be downloaded and returned (e.g. MASK, RGB, RGB32F or DEPTH).
         *
         *  @return  The most recently rendered image according to the desired frame type.
         */
        cv::Mat downloadFrame(RenderingEngineQT::FrameType type);

        /**
         * Downloads the most recently rendered image from the GPU
         * Only transfer the ROI region
         *
         * @param type The frame type to be downloaded and returned (e.g. MASK, RGB, RGB32F or DEPTH). Only realize depth type
         * @param roi the region to be download
         *
         * @return The most recently rendered image according to the desired frame type in the ROI region.
         * */
        cv::Mat downloadFrameFromROI(RenderingEngineQT::FrameType type,const cv::Rect &roi);

        /**
         *  Destroys and deletes the current rendering engine singleton instance.
         */
        void destroy();

        int width;
        int height;

    private:
        static RenderingEngineQT *instance;

        QOpenGLBuffer vertexBuffer;
        QOpenGLBuffer normalBuffer;
        QOpenGLBuffer indexBuffer;
        QOpenGLBuffer texcoordsBuffer;

        std::vector<GLuint> offsetsRender;

        bool buffersInitialsed;

        int fullWidth;
        int fullHeight;
        float zNear;
        float zFar;

        int numLevels;
        int currentLevel;

        std::vector<cv::Matx44f> calibrationMatrices;
        cv::Matx44f projectionMatrix;
        cv::Matx44f lookAtMatrix;

        QOffscreenSurface *surface;
        QOpenGLContext *glContext;
        QOpenGLTexture* texture;

        GLuint frameBufferID;
        GLuint colorTextureID;
        GLuint depthTextureID;

		// Anti aliasing
		//-----------------------------------------------
		bool m_isMulti{true};		// 抗锯齿开关
		bool m_isBlanking{true};	// 不可见面消隐
		GLuint frameBufferMultiID;
   		GLuint textureColorBufferMultiSampled;
    	GLuint robMulti;
		//-----------------------------------------------

        int angle;

        cv::Vec3f lightPosition;

        QString shaderFolder;
        QOpenGLShaderProgram *silhouetteShaderProgram;
        QOpenGLShaderProgram *phongblinnShaderProgram;
        QOpenGLShaderProgram *normalsShaderProgram;
        QOpenGLShaderProgram *textureShaderProgram;

        bool initRenderingBuffers();

        bool initShaderProgram(QOpenGLShaderProgram *program, QString shaderName);

    };

} // namespace ar3dv;


#endif //AR3DVENGINE_RENDERER_QT_H
