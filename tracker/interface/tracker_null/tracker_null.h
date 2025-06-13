// Use this tracker when there are no specific trackers in the engine or you don't
// want to use any trackers. In these cases, the NullTracker will do nothong, but
// guarantees the normal operation of the engine..

#pragma once

#include "../tracker_base.h"

#include <glog/logging.h>
#include <iostream>

#include "definition/global_definition.h"

namespace ar3dv
{

	class NullTracker : public TrackerBase
	{

	public:
		NullTracker();

		~NullTracker();

		void Init() override;

		void StartTracking();

		void Estimate() override;

		void SetData();

		void SetModels();

		void SetCameras();
	};

} // namespace ar3dv