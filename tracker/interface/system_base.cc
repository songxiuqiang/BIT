#include "system_base.h"

#include "glog/logging.h"

using namespace ar3dv;

SystemBase::SystemBase()
{
}

SystemBase::~SystemBase()
{
	if (m_instance)
		delete m_instance;
}

SystemBase *SystemBase::m_instance;
SystemBase *SystemBase::Instance()
{
	if (m_instance == nullptr)
		m_instance = new SystemBase();
	return m_instance;
}

ProcessResult *SystemBase::GetResult()
{
	return &m_processResult;
}

void SystemBase::SetCameras(const CamParams &camParams)
{
	for (auto c : m_camParams)
	{
		if (c.sourceId == camParams.sourceId)
			m_camParams.pop_front();
	}
	m_camParams.push_back(camParams);
}

void SystemBase::SetData(StdData *stdData)
{
	for (auto &d : m_stdDatas)
	{
		if (d->sourceId == stdData->sourceId)
			m_stdDatas.pop_front();
	}
	m_stdDatas.push_back(stdData);
}

void SystemBase::SetModels(const MeshModel &meshModel)
{
	for (auto m : m_models)
	{
		if (m.id == meshModel.id)
			m_models.pop_front();
	}
	m_models.push_back(meshModel);
	return;
}

void SystemBase::SetZnearZfar(const float &zn, const float &zf)
{
	m_zn = zn;
	m_zf = zf;
	return;
}