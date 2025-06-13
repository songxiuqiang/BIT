#include "renderer_QT.h"

using namespace std;
using namespace cv;

namespace ar3dv{

    RenderingEngineQT* RenderingEngineQT::instance;

    RenderingEngineQT::RenderingEngineQT(void)
    {
        QSurfaceFormat glFormat;
        glFormat.setVersion(3, 3);
        glFormat.setSamples(4); // 多重采样, 抗锯齿
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
        textureShaderProgram = new QOpenGLShaderProgram();

        calibrationMatrices.push_back(Matx44f::eye());

        projectionMatrix = Transformations::perspectiveMatrix(40, 4.0f/3.0f, 0.1, 1000.0);

        lookAtMatrix = Transformations::lookAtMatrix(0, 0, 0, 0, 0, 1, 0, -1, 0);
		// VLOG(0) << "lookAtMatrix " << lookAtMatrix;

        currentLevel = 0;
    }

    RenderingEngineQT::~RenderingEngineQT(void)
    {
        glDeleteTextures(1, &colorTextureID);
        glDeleteTextures(1, &depthTextureID);
        glDeleteFramebuffers(1, &frameBufferID);

        offsetsRender.clear();

        if(buffersInitialsed)
        {
            vertexBuffer.release();
            vertexBuffer.destroy();
            normalBuffer.release();
            normalBuffer.destroy();
            indexBuffer.release();
            indexBuffer.destroy();
            texcoordsBuffer.release();
            texcoordsBuffer.destroy();
        }

        delete phongblinnShaderProgram;
        delete normalsShaderProgram;
        delete silhouetteShaderProgram;
        delete textureShaderProgram;
        delete surface;
    }

    void RenderingEngineQT::destroy()
    {
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        delete instance;
        instance = NULL;
    }

    void RenderingEngineQT::makeCurrent()
    {
        glContext->makeCurrent(surface);
    }


    void RenderingEngineQT::doneCurrent()
    {
        glContext->doneCurrent();
    }

    QOpenGLContext* RenderingEngineQT::getContext()
    {
        return glContext;
    }

    GLuint RenderingEngineQT::getFrameBufferID()
    {
        return frameBufferID;
    }

    GLuint RenderingEngineQT::getColorTextureID()
    {
        return colorTextureID;
    }

    GLuint RenderingEngineQT::getDepthTextureID()
    {
        return depthTextureID;
    }

    float RenderingEngineQT::getZNear()
    {
        return zNear;
    }

    float RenderingEngineQT::getZFar()
    {
        return zFar;
    }

    Matx44f RenderingEngineQT::getCalibrationMatrix()
    {
        return calibrationMatrices[currentLevel];
    }

    void RenderingEngineQT::init(const Matx33f& K, int width, int height, float zNear, float zFar,
                                  int numLevels, std::string glslPath)
    {
        this->width = width;
        this->height = height;

        fullWidth = width;
        fullHeight = height;

        this->zNear = zNear;
        this->zFar = zFar;

        this->numLevels = numLevels;

        projectionMatrix = Transformations::perspectiveMatrix(K, width, height, zNear, zFar, true);

        makeCurrent();

        initializeOpenGLFunctions();

        //FIX FOR NEW OPENGL
        uint vao;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        calibrationMatrices.clear();

        for(int i = 0; i < numLevels; i++)
        {
            float s = pow(2, i);

            Matx44f K_l = Matx44f::eye();
            K_l(0, 0) = K(0, 0)/s;
            K_l(1, 1) = K(1, 1)/s;
            K_l(0, 2) = K(0, 2)/s;
            K_l(1, 2) = K(1, 2)/s;

            calibrationMatrices.push_back(K_l);
        }

        cout << "GL Version " << glGetString(GL_VERSION) << endl << "GLSL Version " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

		// glEnable(GL_ALPHA_TEST);
        glEnable(GL_DEPTH_TEST);   
        glEnable(GL_DEPTH);
        glEnable(GL_MULTISAMPLE);

        glDepthRange(0, 1);
        glClearDepth(1.0f);
        glDepthFunc(GL_LESS);
        // glDepthFunc(GL_GREATER);
		
		// Invert depth
        // glDepthRange(1, 0);
        // glClearDepth(0.0f);
        // glDepthFunc(GL_GREATER);

        // glClearColor(1.0, 1.0, 1.0, 0.0);
        glClearColor(0.0, 0.0, 0.0, 0.0);


        initRenderingBuffers();

        // to your own path
        shaderFolder = glslPath.data();

        initShaderProgram(silhouetteShaderProgram, "silhouette");
        initShaderProgram(phongblinnShaderProgram, "phongblinn");
        initShaderProgram(normalsShaderProgram, "normals");
        initShaderProgram(textureShaderProgram, "texture");

        angle = 0;

        lightPosition = cv::Vec3f(0, 0, 0);

        doneCurrent();
    }

    void RenderingEngineQT::initBuffers(Model* model)
    {
        buffersInitialsed = false;

        offsetsRender.push_back(0);
        offsetsRender.push_back(model->getIndices().size());

        vertexBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
        normalBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
        indexBuffer = QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
        texcoordsBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);

        vertexBuffer.create();
        vertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexBuffer.bind();
        vertexBuffer.allocate(model->getVertices().data(), (int)model->getVertices().size() * sizeof(Vec3f));

        normalBuffer.create();
        normalBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        normalBuffer.bind();
        normalBuffer.allocate(model->getNormals().data(), (int)model->getNormals().size() * sizeof(Vec3f));

        indexBuffer.create();
        indexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        indexBuffer.bind();
        indexBuffer.allocate(model->getIndices().data(), (int)model->getIndices().size() * sizeof(int));

        texcoordsBuffer.create();
        texcoordsBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        texcoordsBuffer.bind();
        texcoordsBuffer.allocate(model->getTexcoords().data(), (int)model->getNormals().size() * sizeof(Vec2f));

        texture = loadTexture(QString::fromStdString(model->getTexPath()));

        buffersInitialsed = true;
    }

    void RenderingEngineQT::loadMaterials(const std::string texPath)
    {
        texture = loadTexture(QString::fromStdString(texPath));
    }

    QOpenGLTexture* RenderingEngineQT::loadTexture(const QString& path)
    {
        QImage image = QImage(path).mirrored(false, false);
        QOpenGLTexture* texture = new QOpenGLTexture(image.mirrored());
        // set the texture wrapping parameters
        texture->setWrapMode(QOpenGLTexture::DirectionS, QOpenGLTexture::Repeat);
        texture->setWrapMode(QOpenGLTexture::DirectionT, QOpenGLTexture::Repeat);
        // set texture filtering parameters
        texture->setMinificationFilter(QOpenGLTexture::Linear);
        texture->setMagnificationFilter(QOpenGLTexture::Linear);

        return texture;
    }

    int RenderingEngineQT::getNumLevels()
    {
        return numLevels;
    }

    void RenderingEngineQT::setLevel(int level)
    {
        currentLevel = level;
        int s = pow(2, currentLevel);
        width = fullWidth/s;
        height = fullHeight/s;

        width += width%4;
        height += height%4;
    }

    int RenderingEngineQT::getLevel()
    {
        return currentLevel;
    }

    bool RenderingEngineQT::initRenderingBuffers()
    {
        glGenTextures(1, &colorTextureID);
        glBindTexture(GL_TEXTURE_2D, colorTextureID);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glGenTextures(1, &depthTextureID);
        glBindTexture(GL_TEXTURE_2D, depthTextureID);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glGenFramebuffers(1, &frameBufferID);
        glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTextureID, 0);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTextureID, 0);

        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            cout << "error creating rendering buffers" << endl;
            return false;
        }

	if (m_isMulti) {
        glGenFramebuffers(1, &frameBufferMultiID);
        glBindFramebuffer(GL_FRAMEBUFFER, frameBufferMultiID);
        // create a multisampled color attachment texture
        glGenTextures(1, &textureColorBufferMultiSampled);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, textureColorBufferMultiSampled);
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 25, GL_RGBA, width, height, GL_TRUE);
        // glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGB, width, height, GL_TRUE);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, textureColorBufferMultiSampled, 0);

        glGenRenderbuffers(1, &robMulti);
        glBindRenderbuffer(GL_RENDERBUFFER, robMulti);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, 25, GL_DEPTH24_STENCIL8, width, height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, robMulti);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            cout << "error creating multiRendering buffers" << endl;
            return false;
        }
    }
        return true;
    }


	bool DrawLinesInImg(const cv::Mat &img, const cv::Point2f &p1,const cv::Point2f &p2, const cv::Scalar &scalar)
	{
    	if( p1.y > 0 && p1.y < img.rows
    	    && p1.x > 0 && p1.x < img.cols
    	    && p2.y > 0 && p2.y < img.rows
    	    && p2.x > 0 && p2.x < img.cols)
    	{
    	    cv::line(img,p1,p2,scalar,1,cv::LINE_AA);
    	    return true;
    	}
    	return false;
	}

	cv::Mat RenderingEngineQT::RenderBBX(const cv::Mat &img, Model *model)
	{
		std::vector<cv::Point2f> projections;
		cv::Rect rect;
		cv::Matx44f K = getCalibrationMatrix();
		model->projectBoundingBox(K,projections,rect);
		std::vector<cv::Point2f> bbx = model->GetBBXPoints();

		if(bbx.size()==0) VLOG(0) << "Empty bbx points";

		Point2f p0 = bbx[0];
		Point2f p1 = bbx[1];
		Point2f p2 = bbx[2];
		Point2f p3 = bbx[3];
		Point2f p4 = bbx[4];
		Point2f p5 = bbx[5];
		Point2f p6 = bbx[6];
		Point2f p7 = bbx[7];

		Scalar color(0,255,255);
		DrawLinesInImg(img,p0,p1,color);
		DrawLinesInImg(img,p0,p2,color);
		DrawLinesInImg(img,p0,p3,color);
		DrawLinesInImg(img,p1,p5,color);
		DrawLinesInImg(img,p1,p6,color);
		DrawLinesInImg(img,p2,p4,color);
		DrawLinesInImg(img,p2,p5,color);
		DrawLinesInImg(img,p3,p4,color);
		DrawLinesInImg(img,p3,p6,color);
		DrawLinesInImg(img,p7,p4,color);
		DrawLinesInImg(img,p7,p5,color);
		DrawLinesInImg(img,p7,p6,color);

		return img;
	}

    bool RenderingEngineQT::initShaderProgram(QOpenGLShaderProgram *program, QString shaderName)
    {
        if (!program->addShaderFromSourceFile(QOpenGLShader::Vertex, shaderFolder + shaderName + "_vertex_shader.glsl")) {
            cout << "error adding vertex shader from source file" << endl;
            return false;
        }
        if (!program->addShaderFromSourceFile(QOpenGLShader::Fragment, shaderFolder + shaderName + "_fragment_shader.glsl")) {
            cout << "error adding fragment shader from source file" << endl;
            return false;
        }

        if (!program->link()) {
            cout << "error linking shaders" << endl;
            return false;
        }
        return true;
    }

    void RenderingEngineQT::renderSilhouette(Model* model, GLenum polyonMode, bool invertDepth, float r, float g, float b, bool drawAll)
    {
        vector<Model*> models;
        models.push_back(model);

        vector<Point3f> colors;
        colors.push_back(Point3f(r, g, b));

        renderSilhouette(models, polyonMode, invertDepth, colors, drawAll);
    }

    void RenderingEngineQT::renderShaded(Model* model, GLenum polyonMode, float r, float g, float b, bool drawAll)
    {
        vector<Model*> models;
        models.push_back(model);

        vector<Point3f> colors;
        colors.push_back(Point3f(r, g, b));

        renderShaded(models, polyonMode, colors, drawAll);
    }

    void RenderingEngineQT::renderFeatureLine(Model* model, GLenum polyonMode, float r, float g, float b, bool drawAll)
    {
        vector<Model*> models;
        models.push_back(model);

        vector<Point3f> colors;
        colors.push_back(Point3f(r, g, b));

        renderFeatureLine(models, polyonMode, colors, drawAll);
    }

    void RenderingEngineQT::renderNormals(Model* model, GLenum polyonMode, bool drawAll)
    {
        vector<Model*> models;
        models.push_back(model);

        renderNormals(models, polyonMode, drawAll);
    }

    void RenderingEngineQT::renderTexture(Model* model, GLenum polyonMode, bool drawAll)
    {
        vector<Model*> models;
        models.push_back(model);

        renderTexture(models, polyonMode, drawAll);
    }

    void RenderingEngineQT::renderSilhouette(vector<Model*> models, GLenum polyonMode, bool invertDepth, const std::vector<cv::Point3f>& colors, bool drawAll)
    {
        glViewport(0, 0, width, height);
    	if (m_isMulti) {
    	    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferMultiID);
    	}

        if(invertDepth)
        {
            glClearDepth(1.0f);
            glDepthFunc(GL_LESS);
        }

        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
		glLineWidth(5.0);


        for(int i = 0; i < models.size(); i++)
        {
            Model* model = models[i];

            if(model->isInitialized() || drawAll)
            {
                Matx44f pose = model->getPose();
                Matx44f normalization = model->getNormalization();

                Matx44f modelViewMatrix = lookAtMatrix*(pose*normalization);

                Matx44f modelViewProjectionMatrix = projectionMatrix*modelViewMatrix;

                silhouetteShaderProgram->bind();
                silhouetteShaderProgram->setUniformValue("uMVPMatrix", QMatrix4x4(modelViewProjectionMatrix.val));
                silhouetteShaderProgram->setUniformValue("uAlpha", 1.0f);

                Point3f color;
                if(i < colors.size())
                {
                    color = colors[i];
                }
                else
                {
                    color = Point3f((float)(model->getModelID())/255.0f, 0.0f, 0.0f);
                }
                silhouetteShaderProgram->setUniformValue("uColor", QVector3D(color.x, color.y, color.z));

				// 半透明
				// if(i == 0){
				// 	glEnable(GL_BLEND);
                // 	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                // 	phongblinnShaderProgram->setUniformValue("uAlpha", 0.3f); // 设置半透明度
				// }else{
				// 	glDisable(GL_BLEND);
				// }

				// 消隐背面三角片
				// 1. 在GL_FILL模式下渲染三角片, 深度值已写入zbuffer
				// 2. 在GL_LINE模式下渲染现况

				if(m_isBlanking){
					glDisable(GL_TEXTURE_2D);
					glColorMask(0, 0, 0, 0);
        			glEnable(GL_DEPTH_TEST);   
        			// glEnable(DEPTH);   
					glDepthFunc(GL_LEQUAL);
                	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
					glPolygonOffset(1.1f, 4.0f);
    				glEnable (GL_POLYGON_OFFSET_FILL);  //启用多边形偏移
                	model->draw(silhouetteShaderProgram);
				}
				
				glDepthFunc(GL_LEQUAL);
				glEnable(GL_ALPHA_TEST);
				glClearColor(0.0,0.0,0.0,0.0);
				// glClearColor(1.0,1.0,1.0,0.0); // 白色背景
    			glDisable (GL_POLYGON_OFFSET_FILL);
				glColorMask(1, 1, 1, 1);
  				glPolygonMode (GL_FRONT_AND_BACK, polyonMode);
  				// glPolygonMode (GL_FRONT_AND_BACK, polyonMode);
                model->draw(silhouetteShaderProgram);
            }
        }

		
        // glClearDepth(0.0f);
        // glDepthFunc(GL_GREATER);

		if (m_isMulti) {
    	    // now blit multisampled buffer(s) to normal colorbuffer of intermediate FBO. Image is stored in screenTexture
    	    glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferMultiID);
    	    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameBufferID);
    	    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    	    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
	
    	    // return to framebuffer
    	    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
    	}

        glFinish();
    }

    void RenderingEngineQT::renderShaded(vector<Model*> models, GLenum polyonMode, const std::vector<cv::Point3f>& colors, bool drawAll)
    {
        glViewport(0, 0, width, height);
    	if (m_isMulti) {
    	    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferMultiID);
    	}

        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        for(int i = 0; i < models.size(); i++)
        {
            Model* model = models[i];	
			
            if(model->isInitialized() || drawAll)
            {
                Matx44f pose = model->getPose();
                Matx44f normalization = model->getNormalization();
				

                Matx44f modelViewMatrix = lookAtMatrix*(pose*normalization);

                Matx33f normalMatrix = modelViewMatrix.get_minor<3, 3>(0, 0).inv().t();

                Matx44f modelViewProjectionMatrix = projectionMatrix*modelViewMatrix;

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
                if(i < colors.size())
                {
                    color = colors[i];
                }
                else
                {
                    color = Point3f(1.0, 0.5, 0.0);
                }
                phongblinnShaderProgram->setUniformValue("uColor", QVector3D(color.x, color.y, color.z));


				bool renderFirstObj = true;
				if(renderFirstObj){
					if(m_isBlanking){
						glDisable(GL_TEXTURE_2D);
						glColorMask(0, 0, 0, 0);
        				glEnable(GL_DEPTH_TEST);   
						glDepthFunc(GL_LEQUAL);
                		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
						glPolygonOffset(0.1f, 100.0f);
    					glEnable (GL_POLYGON_OFFSET_FILL);  //启用多边形偏移
                		model->draw(phongblinnShaderProgram);
                		// draw(phongblinnShaderProgram);
					}

					glDepthFunc(GL_LEQUAL);
					glEnable(GL_ALPHA_TEST);
					glClearColor(0.0,0.0,0.0,0.0);
					// glClearColor(1.0,1.0,1.0,0.0);
    				glDisable (GL_POLYGON_OFFSET_FILL);
					glColorMask(1, 1, 1, 1);
  					glPolygonMode (GL_FRONT_AND_BACK, polyonMode);
					// glPolygonOffset(1.1f, 4.0f);
                	model->draw(phongblinnShaderProgram);
                	// draw(phongblinnShaderProgram);
				}else{
					if (i == 0) {
   	             		glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE); // 禁用颜色写入
   	             		glEnable(GL_DEPTH_TEST);
   	             		glDepthFunc(GL_LESS);
   	             		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   	             		model->draw(phongblinnShaderProgram);
   	             		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE); // 恢复颜色写入
   	        	 	} else if (i == 1) {
   	             		glEnable(GL_DEPTH_TEST);
   	             		glDepthFunc(GL_LEQUAL);
   	             		glPolygonMode(GL_FRONT_AND_BACK, polyonMode);
   	             		model->draw(phongblinnShaderProgram);
   	         		}
				}


            }
        }
		if (m_isMulti) {
        // now blit multisampled buffer(s) to normal colorbuffer of intermediate FBO. Image is stored in screenTexture
        	glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferMultiID);
        	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameBufferID);
        	glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        	glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
        	// glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
        	// glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_LINEAR);

        	// return to framebuffer
        	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
    	}
        glFinish();
    }

	void RenderingEngineQT::renderFeatureLine(vector<Model*> models, GLenum polyonMode, const std::vector<cv::Point3f>& colors, bool drawAll)
    {
        glViewport(0, 0, width, height);
    	if (m_isMulti) {
    	    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferMultiID);
    	}

        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        for(int i = 0; i < models.size(); i++)
        {
            Model* model = models[i];	
			
			if(i == 1) glClear(GL_COLOR_BUFFER_BIT);
            if(model->isInitialized() || drawAll)
            {
                Matx44f pose = model->getPose();
                Matx44f normalization = model->getNormalization();
				

                Matx44f modelViewMatrix = lookAtMatrix*(pose*normalization);

                Matx33f normalMatrix = modelViewMatrix.get_minor<3, 3>(0, 0).inv().t();

                Matx44f modelViewProjectionMatrix = projectionMatrix*modelViewMatrix;

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
                if(i < colors.size())
                {
                    color = colors[i];
                }
                else
                {
                    color = Point3f(1.0, 0.5, 0.0);
                }
                phongblinnShaderProgram->setUniformValue("uColor", QVector3D(color.x, color.y, color.z));


				// 消隐背面三角片
				// 1. 在GL_FILL模式下渲染三角片, 深度值已写入zbuffer
				// 2. 在GL_LINE模式下渲染现况
				if(m_isBlanking){
					glDisable(GL_TEXTURE_2D);
					glColorMask(0, 0, 0, 0);
        			glEnable(GL_DEPTH_TEST);   
					glDepthFunc(GL_LEQUAL);
                	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
					glPolygonOffset(0.1f, 100.0f);
    				glEnable (GL_POLYGON_OFFSET_FILL);  //启用多边形偏移
                	model->draw(phongblinnShaderProgram);
                	// draw(phongblinnShaderProgram);
				}

			if(i == 1){
				glDepthFunc(GL_LEQUAL);
				glEnable(GL_ALPHA_TEST);
				glClearColor(0.0,0.0,0.0,0.0);
    			glDisable (GL_POLYGON_OFFSET_FILL);
				glColorMask(1, 1, 1, 1);
  				glPolygonMode (GL_FRONT_AND_BACK, polyonMode);
				// glPolygonOffset(1.1f, 4.0f);
                model->draw(phongblinnShaderProgram);
                // draw(phongblinnShaderProgram);
			}
            }
        }
		if (m_isMulti) {
        // now blit multisampled buffer(s) to normal colorbuffer of intermediate FBO. Image is stored in screenTexture
        	glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferMultiID);
        	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameBufferID);
        	glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        	glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
        	// glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
        	// glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_LINEAR);

        	// return to framebuffer
        	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
    	}
        glFinish();
    }

    void RenderingEngineQT::renderNormals(vector<Model*> models, GLenum polyonMode, bool drawAll)
    {
        glViewport(0, 0, width, height);

        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        for(int i = 0; i < models.size(); i++)
        {
            Model* model = models[i];

            if(model->isInitialized() || drawAll)
            {
                Matx44f pose = model->getPose();
                Matx44f normalization = model->getNormalization();

                Matx44f modelViewMatrix = lookAtMatrix*(pose*normalization);

                Matx33f normalMatrix = modelViewMatrix.get_minor<3, 3>(0, 0).inv().t();

                Matx44f modelViewProjectionMatrix = projectionMatrix*modelViewMatrix;

                normalsShaderProgram->bind();
                normalsShaderProgram->setUniformValue("uMVMatrix", QMatrix4x4(modelViewMatrix.val));
                normalsShaderProgram->setUniformValue("uMVPMatrix", QMatrix4x4(modelViewProjectionMatrix.val));
                normalsShaderProgram->setUniformValue("uNormalMatrix", QMatrix3x3(normalMatrix.val));
                normalsShaderProgram->setUniformValue("uAlpha", 1.0f);

                glPolygonMode(GL_FRONT_AND_BACK, polyonMode);
                draw(normalsShaderProgram);
            }
        }

		if (m_isMulti) {
        	// now blit multisampled buffer(s) to normal colorbuffer of intermediate FBO. Image is stored in screenTexture
			glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferMultiID);
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameBufferID);
			glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
			glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
			glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
    	}
        glFinish();
    }


    void RenderingEngineQT::renderTexture(vector<Model*> models, GLenum polyonMode, bool drawAll)
    {
        glViewport(0, 0, width, height);

		bool isMulti = 1;
		if (isMulti) {
    	    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferMultiID);
    	}
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
		

        for (int i = 0; i < models.size(); i++)
        {
            Model* model = models[i];

            if (model->isInitialized() || drawAll)
            {
                Matx44f pose = model->getPose();
                Matx44f normalization = model->getNormalization();

                Matx44f modelViewMatrix = lookAtMatrix * (pose * normalization);

                Matx44f modelViewProjectionMatrix = projectionMatrix * modelViewMatrix;

                textureShaderProgram->bind();
                glActiveTexture(GL_TEXTURE0);
                texture->bind();

                textureShaderProgram->setUniformValue("texture_diffuse", 0);
                textureShaderProgram->setUniformValue("uMVPMatrix", QMatrix4x4(modelViewProjectionMatrix.val));

                glPolygonMode(GL_FRONT_AND_BACK, polyonMode);

                draw(textureShaderProgram);
            }
        }

        glClearDepth(0.0f);
        glDepthFunc(GL_GREATER);

		// Anti aliasing

	if (isMulti) {
        // now blit multisampled buffer(s) to normal colorbuffer of intermediate FBO. Image is stored in screenTexture
        glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferMultiID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameBufferID);
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);

        // return to framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
    }
        glFinish();
    }

    void RenderingEngineQT::draw(QOpenGLShaderProgram *program, GLint primitives)
    {
        vertexBuffer.bind();
        program->enableAttributeArray("aPosition");
        program->setAttributeBuffer("aPosition", GL_FLOAT, 0, 3, sizeof(Vec3f));

        normalBuffer.bind();
        program->enableAttributeArray("aNormal");
        program->setAttributeBuffer("aNormal", GL_FLOAT, 0, 3, sizeof(Vec3f));

        texcoordsBuffer.bind();
        program->enableAttributeArray("aTexCoords");
        program->setAttributeBuffer("aTexCoords", GL_FLOAT, 0, 3, sizeof(Vec2f));

        program->enableAttributeArray("aColor");
        program->setAttributeBuffer("aColor", GL_UNSIGNED_BYTE, 0, 3, sizeof(Vec3b));

        indexBuffer.bind();

        for (uint i = 0; i < offsetsRender.size() - 1; i++) {
            GLuint size = offsetsRender.at(i + 1) - offsetsRender.at(i);
            GLuint offset = offsetsRender.at(i);
            glDrawElements(primitives, size, GL_UNSIGNED_INT, (GLvoid*)(offset*sizeof(GLuint)));
        }
    }

    cv::Mat RenderingEngineQT::downloadFrame(RenderingEngineQT::FrameType type)
    {

        Mat res;
        switch (type)
        {
			case ALPHA:
                res = Mat(height, width, CV_32FC1);
                glReadPixels(0, 0, res.cols, res.rows, GL_ALPHA, GL_FLOAT, res.data);
                break;
            case MASK:
                res = Mat(height, width, CV_8UC1);
                glReadPixels(0, 0, res.cols, res.rows, GL_RED, GL_UNSIGNED_BYTE, res.data);
                break;
            case RGB:
                res = Mat(height, width, CV_8UC3);
                glReadPixels(0, 0, res.cols, res.rows, GL_RGB, GL_UNSIGNED_BYTE, res.data);
                break;
            case RGB_32F:
                res = Mat(height, width, CV_32FC3);
                glReadPixels(0, 0, res.cols, res.rows, GL_RGB, GL_FLOAT, res.data);
                break;
            case DEPTH:
                res = Mat(height, width, CV_32FC1);
                glReadPixels(0, 0, res.cols, res.rows, GL_DEPTH_COMPONENT, GL_FLOAT,  res.data);
                break;
            default:
                res = Mat::zeros(height, width, CV_8UC1);
                break;
        }

        return res;
    }

    cv::Mat RenderingEngineQT::downloadFrameFromROI(RenderingEngineQT::FrameType type, const cv::Rect &roi)
    {
        Mat res;
        switch (type)
        {
            case MASK:
                res = Mat(height, width, CV_8UC1);
                glReadPixels(0, 0, res.cols, res.rows, GL_RED, GL_UNSIGNED_BYTE, res.data);
                break;
            case RGB:
                res = Mat(height, width, CV_8UC3);
                glReadPixels(0, 0, res.cols, res.rows, GL_RGB, GL_UNSIGNED_BYTE, res.data);
                break;
            case RGB_32F:
                res = Mat(height, width, CV_32FC3);
                glReadPixels(0, 0, res.cols, res.rows, GL_RGB, GL_FLOAT, res.data);
                break;
            case DEPTH:
            {
                res = cv::Mat::zeros(height, width, CV_32FC1);
                cv::Mat roiImg = cv::Mat::zeros(roi.height, roi.width, CV_32FC1);
                glReadPixels(roi.x, roi.y, roi.width, roi.height, GL_DEPTH_COMPONENT, GL_FLOAT, roiImg.data);
                if(roi.area() != 0) {
                    roiImg.copyTo(res(roi));
                }else{
                    glReadPixels(0, 0, res.cols, res.rows, GL_DEPTH_COMPONENT, GL_FLOAT, res.data);
                }
                break;
            }
            default:
                res = Mat::zeros(height, width, CV_8UC1);
                break;
        }

        return res;
    }

} // namespace ar3dv