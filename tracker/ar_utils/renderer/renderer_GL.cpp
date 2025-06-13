#include "renderer_GL.h"

namespace ar3dv {

	RenderingEngineGL* RenderingEngineGL::m_instance;

    void framebufferSizeCallback(GLFWwindow *window, int width, int height) {
        // make sure the viewport matches the new window dimensions; note that width and
        // height will be significantly larger than specified on retina displays.
        glViewport(0, 0, width, height);
    }

    void RenderingEngineGL::init(const cv::Matx33f &K, int width, int height, float zNear, float zFar,
                                 Model* model) {

        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window = glfwCreateWindow(width, height, "", NULL, NULL);
        if (window == NULL) {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return;
        }
        glfwMakeContextCurrent(window);
        glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

        // glad: load all OpenGL function pointers
        // ---------------------------------------
        if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
            std::cout << "Failed to initialize GLAD" << std::endl;
            return;
        }


		glEnable(GL_DEPTH);
        glEnable(GL_DEPTH_TEST);

		glDepthRange(1, 0);
        glClearDepth(0.0f);
        glDepthFunc(GL_GREATER);
        // glEnable(GL_CULL_FACE);
        // glEnable(GL_GREATER);
        // glEnable(GL_LEQUAL);

        const char *vsSource =
                "#version 330 core\n"
                "layout (location = 0) in vec3 aPos;\n"
                "uniform mat4 intrinsicPara;\n"
                "uniform mat4 extrinsicPara;\n"
                "void main()\n"
                "{\n"
                "gl_Position = intrinsicPara * extrinsicPara * vec4(aPos, 1.0);\n"
                "}\0";

        const char *fsSource =
                "#version 330 core\n"
                "out vec4 FragColor;\n"
                "void main()\n"
                "{\n"
                "FragColor = vec4(1.0f, 0.0f, 0.0f, 0.5f);\n"
                "}\n\0";

        ourShader.prepare(vsSource, fsSource);

        prepareVertices(model);

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(cv::Vec3f), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
        glEnableVertexAttribArray(0);

        glBindVertexArray(0);

        //set up intrinsic parameters
        fx = K(0, 0);
        fy = K(1, 1);
        cx = K(0, 2);
        cy = K(1, 2);
        this->width = width;
        this->height = height;
        zn = zNear;
        zf = zFar;

        intrinsicPara[0][0] = 2.0f * fx / width;
        intrinsicPara[0][1] = 0.f;
        intrinsicPara[0][2] = (1.0f - 2.0f * cx / width);
        intrinsicPara[0][3] = 0.f;

        intrinsicPara[1][0] = 0.f;
        intrinsicPara[1][1] = 2.0f * fy / height;
        intrinsicPara[1][2] = -(1.0f - 2.0f * cy / height);
        intrinsicPara[1][3] = 0.f;

        intrinsicPara[2][0] = 0.f;
        intrinsicPara[2][1] = 0.f;
        intrinsicPara[2][2] = (zn + zf) / (zn - zf);
        intrinsicPara[2][3] = 2.0f * zf * zn / (zn - zf);

        intrinsicPara[3][0] = 0.f;
        intrinsicPara[3][1] = 0.f;
        intrinsicPara[3][2] = -1.f;
        intrinsicPara[3][3] = 0.f;

        intrinsicPara = glm::transpose(intrinsicPara);

        buffer = (GLubyte *) malloc(width * height * sizeof(GLubyte));
        bufferDepth = (GLfloat *) malloc(width * height * sizeof(GLfloat));
    }

    void RenderingEngineGL::prepareVertices(Model* model) {
        vertices = model->getVertices();
        indices = model->getIndices();
    }

    void RenderingEngineGL::render(Model* model) {

        cv::Matx44f pose = model->getPose();

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ourShader.use();

        cv::Mat extrinsic(pose);
		extrinsic = extrinsic * model->getNormalization();
        cv::Mat flipedExtrinsic = flipExtrinsicYZ(extrinsic);

        extrinsicPara[0][0] = flipedExtrinsic.ptr<float>(0)[0];
        extrinsicPara[1][0] = flipedExtrinsic.ptr<float>(1)[0];
        extrinsicPara[2][0] = flipedExtrinsic.ptr<float>(2)[0];
        extrinsicPara[3][0] = 0.f;

        extrinsicPara[0][1] = flipedExtrinsic.ptr<float>(0)[1];
        extrinsicPara[1][1] = flipedExtrinsic.ptr<float>(1)[1];
        extrinsicPara[2][1] = flipedExtrinsic.ptr<float>(2)[1];
        extrinsicPara[3][1] = 0.f;

        extrinsicPara[0][2] = flipedExtrinsic.ptr<float>(0)[2];
        extrinsicPara[1][2] = flipedExtrinsic.ptr<float>(1)[2];
        extrinsicPara[2][2] = flipedExtrinsic.ptr<float>(2)[2];
        extrinsicPara[3][2] = 0.f;

        extrinsicPara[0][3] = flipedExtrinsic.ptr<float>(0)[3];
        extrinsicPara[1][3] = flipedExtrinsic.ptr<float>(1)[3];
        extrinsicPara[2][3] = flipedExtrinsic.ptr<float>(2)[3];
        extrinsicPara[3][3] = 1.0f;

        extrinsicPara = glm::transpose(extrinsicPara);

        ourShader.setMat4("intrinsicPara", intrinsicPara);
        ourShader.setMat4("extrinsicPara", extrinsicPara);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, model->getIndices().size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    cv::Mat RenderingEngineGL::getMask(const cv::Size size) {
        cv::Mat mask;
        glReadPixels(0, 0, size.width, size.height, GL_RED, GL_UNSIGNED_BYTE, buffer);
        mask = cv::Mat(size.height, size.width, CV_8UC1, (uchar*)buffer);
        cv::flip(mask, mask, 0);
        return mask;
    }

    cv::Mat RenderingEngineGL::getDepth(const cv::Size size) {
        cv::Mat depth;
        glReadPixels(0, 0, size.width, size.height, GL_DEPTH_COMPONENT, GL_FLOAT, bufferDepth);
        depth = cv::Mat(size.height, size.width, CV_32FC1, (float*)bufferDepth);
        cv::flip(depth, depth, 0);
        return depth;
    }

    cv::Mat RenderingEngineGL::flipExtrinsicYZ(cv::Mat extrinsic) {

        cv::Mat flipedExtrinsic = extrinsic.clone();

        flipedExtrinsic.ptr<float>(1)[0] = -extrinsic.ptr<float>(1)[0];
        flipedExtrinsic.ptr<float>(1)[1] = -extrinsic.ptr<float>(1)[1];
        flipedExtrinsic.ptr<float>(1)[2] = -extrinsic.ptr<float>(1)[2];
        flipedExtrinsic.ptr<float>(1)[3] = -extrinsic.ptr<float>(1)[3];

        flipedExtrinsic.ptr<float>(2)[0] = -extrinsic.ptr<float>(2)[0];
        flipedExtrinsic.ptr<float>(2)[1] = -extrinsic.ptr<float>(2)[1];
        flipedExtrinsic.ptr<float>(2)[2] = -extrinsic.ptr<float>(2)[2];
        flipedExtrinsic.ptr<float>(2)[3] = -extrinsic.ptr<float>(2)[3];

        return flipedExtrinsic;
    }

}