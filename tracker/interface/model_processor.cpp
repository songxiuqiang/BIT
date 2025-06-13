#include "model_processor.h"

namespace ar3dv
{
	std::vector<Vertex> LoadObjModel(const std::string &filename)
	{
		std::ifstream input(filename);
		std::vector<Vertex> vertices;
		if (input.fail())
		{
			std::cerr << "Failed to open file: " << filename << '\n';
			throw std::runtime_error("Failed to open file");
		}

		std::string line;
		while (std::getline(input, line))
		{
			if (line.substr(0, 2) == "v ")
			{
				Vertex v;
				std::sscanf(line.c_str(), "v %f %f %f", &v.x, &v.y, &v.z);
				vertices.push_back(v);
			}
		}
		input.close();

		return vertices;
	}

	BoundingBox ComputeBoundingBox(const std::vector<Vertex> &vertices)
	{
		BoundingBox bbox;
		bbox.min.x = bbox.max.x = vertices[0].x;
		bbox.min.y = bbox.max.y = vertices[0].y;
		bbox.min.z = bbox.max.z = vertices[0].z;

		for (const auto &vertex : vertices)
		{
			bbox.min.x = std::min(bbox.min.x, vertex.x);
			bbox.min.y = std::min(bbox.min.y, vertex.y);
			bbox.min.z = std::min(bbox.min.z, vertex.z);
			bbox.max.x = std::max(bbox.max.x, vertex.x);
			bbox.max.y = std::max(bbox.max.y, vertex.y);
			bbox.max.z = std::max(bbox.max.z, vertex.z);
		}

		bbox.width = bbox.max.x - bbox.min.x;
		bbox.height = bbox.max.y - bbox.min.y;
		bbox.depth = bbox.max.z - bbox.min.z;

		bbox.diagonal = sqrt(pow(bbox.width, 2) + pow(bbox.height, 2) + pow(bbox.depth, 2));
		bbox.longestSide = std::max(std::max(bbox.width, bbox.height), bbox.depth);
		return bbox;
	}

	void ChangeModelScale(std::string inputFilename, std::string outputFilename, const float &scale)
	{
		static int count = 1;
		std::string outputFilenameWithoutSuffix = ar3dv::DeleteStrParts(outputFilename, ".", BACK_2_FRONT, 1);
		std::string outputFilenameDebug = outputFilenameWithoutSuffix + std::to_string(count) + ".obj";
		std::ifstream ifsDebug(inputFilename);
		std::ofstream ofsDebug(outputFilenameDebug, std::ios::trunc);
		std::vector<std::string> linesDebug;
		if (!ofsDebug.is_open())
			std::cout << "Can not open " << outputFilenameDebug << "\n";
		if (ofsDebug.is_open())
		{
			std::string line;
			while (std::getline(ifsDebug, line))
			{
				if (line.substr(0, 2) == "v ")
				{
					std::string newLine = "v ";
					std::istringstream iss(line);
					std::string token;
					iss >> token;

					float v;
					while (iss >> v)
					{
						v *= scale;
						newLine += (std::to_string(v) + " ");
					}
					newLine.back() = '\n';
					line = newLine;
				}
				else
				{
					line = line + "\n";
				}
				ofsDebug << line;
			}
			ifsDebug.close();
			ofsDebug.close();
		}
		count++;

		std::vector<std::string> lines;
		std::ifstream ifs(inputFilename);
		std::ofstream ofs(outputFilename, std::ios::trunc);
		if (!ofs.is_open())
			std::cout << "Can not open " << outputFilename << "\n";

		if (ifs.is_open())
		{
			std::string line;
			while (std::getline(ifs, line))
			{
				if (line.substr(0, 2) == "v ")
				{
					std::string newLine = "v ";
					std::istringstream iss(line);
					std::string token;
					iss >> token;

					float v;
					while (iss >> v)
					{
						v *= scale;
						newLine += (std::to_string(v) + " ");
					}
					newLine.back() = '\n';
					line = newLine;
				}
				else
				{
					line = line + "\n";
				}
				ofs << line;
			}
			ifs.close();
			ofs.close();
		}
	}

} // namespace ar3dv
