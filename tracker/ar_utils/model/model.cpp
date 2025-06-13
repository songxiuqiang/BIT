#include "model.h"

using namespace std;
using namespace cv;
namespace ar3dv
{

	Model::Model(const string modelFilename, float tx, float ty, float tz, float alpha, float beta, float gamma, float scale)
	{
		m_id = 0;
		initialized = false;
		buffersInitialsed = false;

		T_i = Transformations::translationMatrix(tx, ty, tz) * Transformations::rotationMatrix(alpha, Vec3f(1, 0, 0)) * Transformations::rotationMatrix(beta, Vec3f(0, 1, 0)) * Transformations::rotationMatrix(gamma, Vec3f(0, 0, 1)) * Matx44f::eye();

		T_cm = T_i;

		scaling = scale;

		T_n = Matx44f::eye();

		hasNormals = false;

		loadModel(modelFilename);

		vertexBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
		normalBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
		indexBuffer = QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);

		computeFaceNormals();
		computeFaceCenters();
		computeMinusd();
	}

	Model::Model(const string modelFilename, cv::Matx44f pose, float scale)
	{
		m_id = 0;
		initialized = false;

		T_i = pose;
		T_cm = T_i;
		scaling = scale;
		T_n = Matx44f::eye();
		hasNormals = false;

		vertexBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
		normalBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
		indexBuffer = QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);

		loadModel(modelFilename);

		computeFaceNormals();
		computeFaceCenters();
		computeMinusd();
	}

	Model::~Model()
	{
		vertices.clear();
		normals.clear();
		texcoords.clear();
		faceNormals.clear();
		faceCenters.clear();
	}

	void Model::initialize()
	{
		initialized = true;
	}

	bool Model::isInitialized()
	{
		return initialized;
	}

	QOpenGLTexture *loadTexture(const QString &path)
	{
		QImage image = QImage(path).mirrored(false, false);
		QOpenGLTexture *texture = new QOpenGLTexture(image.mirrored());
		// set the texture wrapping parameters
		texture->setWrapMode(QOpenGLTexture::DirectionS, QOpenGLTexture::Repeat);
		texture->setWrapMode(QOpenGLTexture::DirectionT, QOpenGLTexture::Repeat);
		// set texture filtering parameters
		texture->setMinificationFilter(QOpenGLTexture::Linear);
		texture->setMagnificationFilter(QOpenGLTexture::Linear);

		return texture;
	}

	void Model::draw(QOpenGLShaderProgram *program, GLint primitives)
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

		for (uint i = 0; i < offsets.size() - 1; i++)
		{
			GLuint size = offsets.at(i + 1) - offsets.at(i);
			GLuint offset = offsets.at(i);
			glDrawElements(primitives, size, GL_UNSIGNED_INT, (GLvoid *)(offset * sizeof(GLuint)));
		}
	}

	void Model::initBuffers()
	{
		offsetsRender.push_back(0);
		offsetsRender.push_back(indices.size());

		vertexBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
		normalBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
		indexBuffer = QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
		texcoordsBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);

		vertexBuffer.create();
		vertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
		vertexBuffer.bind();
		vertexBuffer.allocate(vertices.data(), (int)vertices.size() * sizeof(Vec3f));

		normalBuffer.create();
		normalBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
		normalBuffer.bind();
		normalBuffer.allocate(normals.data(), (int)normals.size() * sizeof(Vec3f));

		indexBuffer.create();
		indexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
		indexBuffer.bind();
		indexBuffer.allocate(indices.data(), (int)indices.size() * sizeof(int));

		texcoordsBuffer.create();
		texcoordsBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
		texcoordsBuffer.bind();
		texcoordsBuffer.allocate(this->getTexcoords().data(), (int)normals.size() * sizeof(Vec2f));

		texture = loadTexture(QString::fromStdString(this->getTexPath()));

		// vertexBuffer.create();
		// vertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
		// vertexBuffer.bind();
		// vertexBuffer.allocate(vertices.data(), (int)vertices.size() * sizeof(Vec3f));

		// normalBuffer.create();
		// normalBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
		// normalBuffer.bind();
		// normalBuffer.allocate(normals.data(), (int)normals.size() * sizeof(Vec3f));

		// indexBuffer.create();
		// indexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
		// indexBuffer.bind();
		// indexBuffer.allocate(indices.data(), (int)indices.size() * sizeof(int));

		buffersInitialsed = true;
	}

	cv::Vec3f Model::computeFaceNormal(cv::Vec3i face)
	{
		const Vec3f &v0 = vertices[face[0]];
		const Vec3f &v1 = vertices[face[1]];
		const Vec3f &v2 = vertices[face[2]];

		return normalize((v1 - v0).cross(v2 - v0));
	}

	void Model::computeFaceNormals()
	{

		faceNormals.clear();

		int nf = faces.size();
		for (int f = 0; f < nf; ++f)
		{
			cv::Vec3i face = faces[f];
			cv::Vec3f fn = computeFaceNormal(face);
			faceNormals.push_back(fn);
		}
	}

	void Model::computeFaceCenters()
	{

		faceCenters.clear();

		int nf = faces.size();
		for (size_t f = 0; f < nf; ++f)
		{

			cv::Vec3i face = faces[f];
			const Vec3f &v0 = vertices[face[0]];
			const Vec3f &v1 = vertices[face[1]];
			const Vec3f &v2 = vertices[face[2]];

			cv::Vec3f fc = (v0 + v1 + v2) / 2;
			faceCenters.push_back(fc);
		}
	}

	void Model::computeMinusd()
	{
		for (size_t f = 0; f < faces.size(); ++f)
		{
			// cv::Vec3f n = faceNormals[f];
			// cv::Vec3f p = faceCenters[f];
			cv::Vec3f n = faceNormals[f];
			cv::Vec3f p = vertices[faces[f][0]];
			float d = n.dot(p);
			minusd.push_back(d);
		}
	}

	Matx44f Model::getPose()
	{
		return T_cm;
	}

	std::vector<cv::Point2f> Model::GetBBXPoints()
	{
		return m_bbx;
	}

	void Model::setPose(const Matx44f &T_cm)
	{
		this->T_cm = T_cm;
	}

	void Model::setInitialPose(const Matx44f &T_cm)
	{
		T_i = T_cm;
	}

	Matx44f Model::getNormalization()
	{
		return T_n;
	}

	cv::Vec3f Model::getLBN()
	{
		return lbn;
	}

	cv::Vec3f Model::getRTF()
	{
		return rtf;
	}

	float Model::getScaling()
	{

		return scaling;
	}

	vector<Vec3f> Model::getVertices()
	{
		return vertices;
	}

	vector<Vec3f> Model::getNormals()
	{
		return normals;
	}

	vector<Vec2f> Model::getTexcoords()
	{
		return texcoords;
	}

	std::vector<cv::Vec3i> Model::getFaces()
	{
		return faces;
	}

	std::vector<uint> Model::getIndices()
	{
		return indices;
	}

	vector<Vec3f> Model::getFaceNormals()
	{
		return faceNormals;
	}

	vector<Vec3f> Model::getFaceCenters()
	{
		return faceCenters;
	}

	vector<float> Model::getMinusd()
	{
		return minusd;
	}

	int Model::getNumVertices()
	{
		return (int)vertices.size();
	}

	int Model::getModelID()
	{
		return m_id;
	}

	std::string Model::getTexPath()
	{
		return texPath;
	}

	void Model::setModelID(int i)
	{
		m_id = i;
	}

	void Model::reset()
	{
		initialized = false;

		T_cm = T_i;
	}

	void Model::loadModel(const string modelFilename)
	{
		Assimp::Importer importer;

		const aiScene *scene = importer.ReadFile(modelFilename, aiProcessPreset_TargetRealtime_Fast);

		CHECK(scene != nullptr) << "Error::Mesh Model Empty from " << modelFilename;

		aiMesh *mesh = scene->mMeshes[0];

		hasNormals = mesh->HasNormals();
		materialId = mesh->mMaterialIndex;

		float inf = numeric_limits<float>::infinity();
		lbn = Vec3f(inf, inf, inf);
		rtf = Vec3f(-inf, -inf, -inf);

		for (int i = 0; i < mesh->mNumFaces; i++)
		{
			aiFace f = mesh->mFaces[i];

			indices.push_back(f.mIndices[0]);
			indices.push_back(f.mIndices[1]);
			indices.push_back(f.mIndices[2]);

			faces.push_back(cv::Vec3i(f.mIndices[0], f.mIndices[1], f.mIndices[2]));
		}

		for (int i = 0; i < mesh->mNumVertices; i++)
		{
			aiVector3D v = mesh->mVertices[i];

			Vec3f p(v.x, v.y, v.z);

			// compute the 3D bounding box of the model
			if (p[0] < lbn[0])
				lbn[0] = p[0];
			if (p[1] < lbn[1])
				lbn[1] = p[1];
			if (p[2] < lbn[2])
				lbn[2] = p[2];
			if (p[0] > rtf[0])
				rtf[0] = p[0];
			if (p[1] > rtf[1])
				rtf[1] = p[1];
			if (p[2] > rtf[2])
				rtf[2] = p[2];

			vertices.push_back(p);
		}

		if (hasNormals)
		{
			for (int i = 0; i < mesh->mNumVertices; i++)
			{
				aiVector3D n = mesh->mNormals[i];

				Vec3f vn = Vec3f(n.x, n.y, n.z);

				normals.push_back(vn);
			}
		}

		offsets.push_back(0);
		offsets.push_back(mesh->mNumFaces * 3);

		if (mesh->mTextureCoords[0])
		{
			for (int i = 0; i < mesh->mNumVertices; i++)
			{
				aiVector3D k = mesh->mTextureCoords[0][i];
				Vec2f vk = Vec2f(k.x, k.y);
				texcoords.push_back(vk);
			}
		}

		// the center of the 3d bounding box
		Vec3f bbCenter = (rtf + lbn) / 2;

		// compute a normalization transform that moves the object to the center of its bounding box and scales it according to the prescribed factor
		// T_n = Transformations::scaleMatrix(scaling)*Transformations::translationMatrix(-bbCenter[0], -bbCenter[1], -bbCenter[2]);
		T_n = Transformations::scaleMatrix(scaling);

		// get the texture path
		int index = modelFilename.find_last_of("/");
		string rootPath = modelFilename.substr(0, index);

		aiString texturePath;
		scene->mMaterials[materialId]->GetTexture(aiTextureType_DIFFUSE, 0, &texturePath);
		texPath = rootPath + "/" + texturePath.C_Str();
	}

	void Model::projectBoundingBox(cv::Matx44f K, std::vector<cv::Point2f> &projections, cv::Rect &boundingRect)
	{
		Vec3f lbn = getLBN();
		Vec3f rtf = getRTF();

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

		Matx44f pose = getPose();
		Matx44f normalization = getNormalization();

		Point2f lt(FLT_MAX, FLT_MAX);
		Point2f rb(-FLT_MAX, -FLT_MAX);
		m_bbx.clear();

		for (int i = 0; i < points3D.size(); i++)
		{
			Vec4f p = K * pose * normalization * points3D[i];
			if (p[2] == 0)
				continue;

			Point2f p2d = Point2f(p[0] / p[2], p[1] / p[2]);
			projections.push_back(p2d);
			m_bbx.push_back(p2d);

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

	std::vector<cv::Vec2f> Model::project3DPoints(cv::Matx44f K, const std::vector<cv::Vec3f> &points3D)
	{
		// cv::Mat frameCopy = frame.clone();
		Matx44f pose = getPose();
		Matx44f normalization = getNormalization();
		std::vector<cv::Vec2f> p2ds;

		for (int i = 0; i < points3D.size(); i++)
		{
			Vec4f points3DHomo(points3D[i][0], points3D[i][1], points3D[i][2], 1);
			Vec4f p = K * pose * normalization * points3DHomo;
			Vec2f p2d(p[0] / p[2], p[1] / p[2]);
			p2ds.push_back(p2d);
			// cv::circle(frameCopy,cv::Point2f(p2d[0],p2d[1]),1,cv::Scalar(0,255,255),2);
		}
		// cv::imshow("circle",frameCopy);
		return p2ds;
	}

	cv::Point2f Model::projectOne3DPoint(cv::Matx44f K, const cv::Vec3f &point3d)
	{
		Matx44f pose = getPose();
		Matx44f normalization = getNormalization();
		cv::Vec4f points3DHomo(point3d[0], point3d[1], point3d[2], 1);
		cv::Vec4f p = K * pose * normalization * points3DHomo;
		cv::Point2f p2d = Point2f(p[0] / p[2], p[1] / p[2]);
		return p2d;
	}

	cv::Rect Model::getROI(cv::Matx44f K, const cv::Size &maxSize, int offset)
	{
		// PROJECT THE 3D BOUNDING BOX AS 2D ROI
		Rect boundingRect;
		vector<Point2f> projections;

		projectBoundingBox(K, projections, boundingRect);

		if (boundingRect.x >= maxSize.width || boundingRect.y >= maxSize.height || boundingRect.x + boundingRect.width <= 0 || boundingRect.y + boundingRect.height <= 0)
		{
			return Rect(0, 0, 0, 0);
		}

		// CROP THE ROI AROUND THE SILHOUETTE
		Rect roi = Rect(boundingRect.x - offset, boundingRect.y - offset, boundingRect.width + 2 * offset, boundingRect.height + 2 * offset);

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

	void Model::convertMask(const cv::Mat &depth, cv::Mat &mask,
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

}