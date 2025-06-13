#include "model.hh"
#include "tclc_histograms.hh"

#include <limits>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/mesh.h>
#include <assimp/postprocess.h>
#include "base/util.hh"
#include "base/global_param.hh"

using namespace std;
using namespace cv;
namespace summer
{

	Model::Model(const string modelFilename, float tx, float ty, float tz, float alpha, float beta, float gamma, float scale)
	{
		Matx44f Ti = Transformations::translationMatrix(tx, ty, tz) * Transformations::rotationMatrix(alpha, Vec3f(1, 0, 0)) * Transformations::rotationMatrix(beta, Vec3f(0, 1, 0)) * Transformations::rotationMatrix(gamma, Vec3f(0, 0, 1)) * Matx44f::eye();

		Init(modelFilename, Ti, scale);
	}

	Model::Model(const std::string modelFilename, const cv::Matx44f &Ti, float scale)
	{
		Init(modelFilename, Ti, scale);
	}

	void Model::Init(const std::string modelFilename, const cv::Matx44f &Ti, float scale)
	{
		m_id = 0;

		initialized = false;

		buffersInitialsed = false;

		T_i = Ti;

		T_cm = T_i;
		T_pm = T_i;

		scaling = scale;

		T_n = Matx44f::eye();

		hasNormals = false;

		vertexBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
		normalBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
		indexBuffer = QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);

		loadModel(modelFilename);
	}

	Model::~Model()
	{
		vertices.clear();
		normals.clear();

		indices.clear();
		offsets.clear();

		if (buffersInitialsed)
		{
			vertexBuffer.release();
			vertexBuffer.destroy();
			normalBuffer.release();
			normalBuffer.destroy();

			indexBuffer.release();
			indexBuffer.destroy();
		}
	}

	void Model::initBuffers()
	{
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

		buffersInitialsed = true;
	}

	void Model::UnInitialize()
	{
		initialized = false;
	}

	void Model::initialize()
	{
		initialized = true;
	}

	bool Model::isInitialized()
	{
		return initialized;
	}

	void Model::draw(QOpenGLShaderProgram *program, GLint primitives)
	{
		vertexBuffer.bind();
		program->enableAttributeArray("aPosition");
		program->setAttributeBuffer("aPosition", GL_FLOAT, 0, 3, sizeof(Vec3f));

		normalBuffer.bind();
		program->enableAttributeArray("aNormal");
		program->setAttributeBuffer("aNormal", GL_FLOAT, 0, 3, sizeof(Vec3f));

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

	Matx44f Model::getPose()
	{
		return T_cm;
	}

	void Model::setPose(const Matx44f &T_cm)
	{
		this->T_cm = T_cm;
	}

	Matx44f Model::getPrePose()
	{
		return T_pm;
	}

	void Model::setPrePose(const Matx44f &T_cm)
	{
		this->T_pm = T_cm;
	}

	void Model::setInitialPose(const Matx44f &T_cm)
	{
		T_i = T_cm;
	}

	Matx44f Model::getNormalization()
	{
		return T_n;
	}

	Vec3f Model::getLBN()
	{
		return lbn;
	}

	Vec3f Model::getRTF()
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

	vector<Vec3f> Model::getSimpleVertices()
	{
		return svertices;
	}

	int Model::getNumVertices()
	{
		return (int)vertices.size();
	}

	int Model::getNumSimpleVertices()
	{
		return (int)svertices.size();
	}

	int Model::getModelID()
	{
		return m_id;
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

	void IdenInsert(std::vector<aiVector3D> &verts, const aiVector3D &vert)
	{
		for (int i = 0; i < verts.size(); ++i)
		{
			if (verts[i] == vert)
				return;
		}

		verts.push_back(vert);
	}

	void IdentAdd(std::vector<Vec3f> &vertices, aiVector3D *aiv, unsigned int num)
	{
		vertices.clear();

		std::vector<aiVector3D> setv;

		for (int i = 0; i < num; ++i)
		{
			IdenInsert(setv, aiv[i]);
		}

		for (auto vert : setv)
		{
			vertices.push_back(Vec3f(vert.x, vert.y, vert.z));
		}
	}

	void Model::loadModel(const string modelFilename)
	{
		VLOG(0) << "Start to load model";
		Assimp::Importer importer;

		const aiScene *scene = importer.ReadFile(modelFilename, aiProcessPreset_TargetRealtime_Fast);

		CHECK(scene != nullptr) << "Error::Mesh Model Empty from Path: " << modelFilename;

		aiMesh *mesh = scene->mMeshes[0];

		hasNormals = mesh->HasNormals();

		float inf = numeric_limits<float>::infinity();
		lbn = Vec3f(inf, inf, inf);
		rtf = Vec3f(-inf, -inf, -inf);

		for (int i = 0; i < mesh->mNumFaces; i++)
		{
			aiFace f = mesh->mFaces[i];

			indices.push_back(f.mIndices[0]);
			indices.push_back(f.mIndices[1]);
			indices.push_back(f.mIndices[2]);
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

		cv::Vec3f diagVector(lbn - rtf);
		bbxDiagLen_ = sqrt(diagVector.dot(diagVector));
		VLOG(0) << "bbxDiagLen of model " << bbxDiagLen_;

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

		if (tk::GlobalParam::Instance()->unit_model)
		{
			// the center of the 3d bounding box
			Vec3f bbCenter = (rtf + lbn) / 2;
			// compute a normalization transform that moves the object to the center of its bounding box and scales it according to the prescribed factor
			VLOG(0) << "unit_model";
			T_n = Transformations::scaleMatrix(scaling) * Transformations::translationMatrix(-bbCenter[0], -bbCenter[1], -bbCenter[2]);
		}
		else
		{
			VLOG(0) << "non unit_model";
			T_n = Transformations::scaleMatrix(scaling);
		}

		VLOG(0) << modelFilename + "s";
		if (tk::IsFileExist(modelFilename + 's'))
		{
			loadSimpleModel(modelFilename + 's');
		}
		else
		{
			svertices = vertices;
		}

		if (mesh->mNumVertices < 24 * 3)
		{
			std::vector<aiVector3D> setv;

			for (int i = 0; i < mesh->mNumFaces; i++)
			{
				aiFace f = mesh->mFaces[i];

				aiVector3D dv;

				dv = mesh->mVertices[f.mIndices[0]] - mesh->mVertices[f.mIndices[1]];
				IdenInsert(setv, mesh->mVertices[f.mIndices[1]] + 0.2f * dv);
				IdenInsert(setv, mesh->mVertices[f.mIndices[1]] + 0.4f * dv);
				IdenInsert(setv, mesh->mVertices[f.mIndices[1]] + 0.6f * dv);
				IdenInsert(setv, mesh->mVertices[f.mIndices[1]] + 0.8f * dv);

				dv = mesh->mVertices[f.mIndices[1]] - mesh->mVertices[f.mIndices[2]];
				IdenInsert(setv, mesh->mVertices[f.mIndices[2]] + 0.2f * dv);
				IdenInsert(setv, mesh->mVertices[f.mIndices[2]] + 0.4f * dv);
				IdenInsert(setv, mesh->mVertices[f.mIndices[2]] + 0.6f * dv);
				IdenInsert(setv, mesh->mVertices[f.mIndices[2]] + 0.8f * dv);

				dv = mesh->mVertices[f.mIndices[2]] - mesh->mVertices[f.mIndices[0]];
				IdenInsert(setv, mesh->mVertices[f.mIndices[0]] + 0.2f * dv);
				IdenInsert(setv, mesh->mVertices[f.mIndices[0]] + 0.4f * dv);
				IdenInsert(setv, mesh->mVertices[f.mIndices[0]] + 0.6f * dv);
				IdenInsert(setv, mesh->mVertices[f.mIndices[0]] + 0.8f * dv);
			}

			svertices = vertices;

			for (auto vert : setv)
			{
				svertices.push_back(Vec3f(vert.x, vert.y, vert.z));
			}
		}

		VLOG(0) << "Load model done.";
		return;
	}

	void Model::loadSimpleModel(const string modelFilename)
	{
		Assimp::Importer importer;
		const aiScene *scene = importer.ReadFile(modelFilename, aiProcessPreset_TargetRealtime_Fast);
		aiMesh *mesh = scene->mMeshes[0];

		for (int i = 0; i < mesh->mNumVertices; i++)
		{
			aiVector3D v = mesh->mVertices[i];
			Vec3f p(v.x, v.y, v.z);
			svertices.push_back(p);
		}
	}

} // ns slot