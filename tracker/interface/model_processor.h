#pragma once
#ifndef __MODEL_PROCESSOR__
#define __MODEL_PROCESSOR__

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "../ar_utils/data_io/data_loader.h"

namespace ar3dv
{

	struct Vertex
	{
		float x, y, z;
	};

	struct BoundingBox
	{
		Vertex min, max;
		float width, height, depth;
		float diagonal, longestSide;
	};

	std::vector<Vertex> LoadObjModel(const std::string &filename);

	BoundingBox ComputeBoundingBox(const std::vector<Vertex> &vertices);

	void ChangeModelScale(std::string inputFilename, std::string outputFilename, const float &scale);

} // namespace ar3dv

#endif