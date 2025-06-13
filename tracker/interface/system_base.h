#ifndef _SYSTEM_BASE_H_
#define _SYSTEM_BASE_H_

#include "opencv2/opencv.hpp"

#include "definition/global_definition.h"

namespace ar3dv
{

	class SystemBase
	{
	public:
		SystemBase();

		~SystemBase();

		static SystemBase *Instance();

		virtual void SetCameras(const CamParams &camParams);

		virtual void SetData(StdData *stdData);

		virtual void SetModels(const MeshModel &meshModel);

		virtual void SetZnearZfar(const float &zn, const float &zf);

		ProcessResult *GetResult();

		bool m_useTracker{false};
		bool m_useDetector{false};
		bool m_useReconstructor{false};

		std::deque<StdData *> m_stdDatas;
		std::deque<CamParams> m_camParams;
		std::deque<MeshModel> m_models;

		float m_zn{0.5};
		float m_zf{2000};

		ProcessResult m_processResult;

	private:
		static SystemBase *m_instance;

	}; // class

} // ar3dv

#endif // _SYSTEM_BASE_H_
