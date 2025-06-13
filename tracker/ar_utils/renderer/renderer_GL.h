#ifndef AR3DVENGINE_RENDERER_GL_H
#define AR3DVENGINE_RENDERER_GL_H

#include <iostream>
#include <opencv2/opencv.hpp>

#include "./shader_GL.h"
// #include "glad/glad.h"
#include <GLFW/glfw3.h>
#include "model/model.h"


#include "glog/logging.h"

namespace ar3dv{

/**
 *  This class implements an OpenGL-based offscreen rendering engine for generating
 *  images of projected 3D meshes based on given object poses and camera instrinsics.
 *  It supports one or mutiple objects to be rendered as binary masks, depth maps,
 *  normal maps or phong-shaded.
 *  This class is based on GLFW3
 */
    class RenderingEngineGL
    {
    public:

        RenderingEngineGL() {};

        ~RenderingEngineGL() {
            glDeleteVertexArrays(1, &VAO);
            glDeleteBuffers(1, &VBO);
            glDeleteBuffers(1, &EBO);
            glfwTerminate();
        };

		static RenderingEngineGL *Instance(void)
        {
            if (m_instance == NULL) m_instance = new RenderingEngineGL();
            return m_instance;
        }

        /**
         * Init the GL renderer, including the shader, buffer, matrix, etc.
         *
         * @param K intrinsic paras
         * @param width img width
         * @param height img height
         * @param zNear near plane
         * @param zFar far plane
         * @param model the model to be rendered
         */
        void init(const cv::Matx33f &K, int width, int height, float zNear, float zFar,
                  Model* model);

        /**
         * Render the object.
         * For now, only render the foreground mask and the depth, need to be replenish
         *
         * @param model the model to be rendered
         */
        void render(Model* model);

        /**
         * Get the foreground mask of the object
         * Can be speed up using ROI
         *
         * @param size the image size for download buffer
         * @return the object foreground mask
         */
        cv::Mat getMask(const cv::Size size);

        /**
         * Get the depth map of the object
         * Can be speed up using ROI
         *
         * @param size the image size for download buffer
         * @return the object depth map
         */
        cv::Mat getDepth(const cv::Size size);

        /**
         * Prepare the vertices of the object for rendering. Pointer the vertices and indices of the model to renderer
         *
         * @param model the model to be rendered
         */
        void prepareVertices(Model* model);

    private:
		static RenderingEngineGL* m_instance;

        int width, height;
        float fx, fy, cx, cy;
        float zn, zf;

        GLFWwindow* window;
        std::vector<cv::Vec3f> vertices;
        std::vector<unsigned int> indices;
        unsigned int VAO, VBO, EBO;
        ar3dv::ShaderGL ourShader;
        glm::mat4 intrinsicPara, extrinsicPara, flipYZ;

        GLubyte* buffer;
        GLfloat* bufferDepth;

        /**
         * Flip the pose matrix. This is because openGL and openCV has different coordinate system
         *
         * @param extrinsic the object pose
         * @return the fliped object pose, used for openGL rendering
         */
        cv::Mat flipExtrinsicYZ(cv::Mat extrinsic);
    };

} // namespace ar3dv;


#endif //AR3DVENGINE_RENDERER_GL_H
