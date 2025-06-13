#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
//#include <windows.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <glog/logging.h>
#include "base/types.hh"

namespace tk {

inline bool IsFileExist (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

inline std::string ToString(int int_val){
	char num[8] = {0};
	//itoa(int_val,num,10);
	snprintf(num, sizeof(num), "%d", int_val);
	std::string text(num);
	return text;   
}

// string utils ///////////////////////////////////////////////

inline std::string trim(const std::string& str) {
	std::string::size_type pos = str.find_first_not_of(' ');
	if (pos == std::string::npos) 
		return str;
	std::string::size_type pos2 = str.find_last_not_of(' ');
	if (pos2 != std::string::npos)
		return str.substr(pos, pos2 - pos + 1);
	return str.substr(pos);
}

inline int split(const std::string& str, std::vector<std::string>& ret_, std::string sep = ",") {
	if (str.empty())
	return 0;

	std::string tmp;
	std::string::size_type pos_begin = str.find_first_not_of(sep);
	std::string::size_type comma_pos = 0;

	while (pos_begin != std::string::npos) {
		comma_pos = str.find(sep, pos_begin);
		if (comma_pos != std::string::npos) {
			tmp = str.substr(pos_begin, comma_pos - pos_begin);
			pos_begin = comma_pos + sep.length();
		}	else {
			tmp = str.substr(pos_begin);
			pos_begin = comma_pos;
		}

		if (!tmp.empty()) {
			ret_.push_back(tmp);
			tmp.clear();
		}
	}
	return 0;
}

inline std::string replace(const std::string& str, const std::string& src, const std::string& dest) {
	std::string ret;

	std::string::size_type pos_begin = 0;
	std::string::size_type pos       = str.find(src);
	while (pos != std::string::npos) {
		std::cout <<"replacexxx:" << pos_begin <<" " << pos <<"\n";
		ret.append(str.data() + pos_begin, pos - pos_begin);
		ret += dest;
		pos_begin = pos + 1;
		pos       = str.find(src, pos_begin);
	}
	if (pos_begin < str.length())
		ret.append(str.begin() + pos_begin, str.end());
	return ret;
}


///////////////////////////////////////////////////

template<typename T>
bool PointInRect(T x, T y, T left, T right, T width, T height) {
	if (x >= left && x < width && y >= right && y < height) return true;
	else return false;
}

#if 0
static void  SetConsoleColor(WORD wAttribute) {
	HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(handle, wAttribute);
}

inline std::ostream&  defcolor(std::ostream& ostr) {
	SetConsoleColor(FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED);
	return ostr;
}

inline std::ostream&  greencolor(std::ostream& ostr) {
	SetConsoleColor(FOREGROUND_GREEN | FOREGROUND_INTENSITY);
	return ostr;
}

inline std::ostream&  bluecolor(std::ostream& ostr) {
	SetConsoleColor(FOREGROUND_BLUE | FOREGROUND_INTENSITY);
	return ostr;
}

inline std::ostream&  redcolor(std::ostream& ostr) {
	SetConsoleColor(FOREGROUND_RED | FOREGROUND_INTENSITY);
	return ostr;
}

inline std::ostream&  lredcolor(std::ostream& ostr) {
	SetConsoleColor(FOREGROUND_RED);
	return ostr;
}

inline std::ostream& yellowcolor(std::ostream& ostr) {
	SetConsoleColor(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
	return ostr;
}

#define GREENTEXT(output) tk::greencolor << output << tk::defcolor
#define REDTEXT(output) tk::redcolor << output << tk::defcolor
#define BLUETEXT(output) tk::bluecolor << output << tk::defcolor 
#define YELLOWTEXT(output) tk::yellowcolor << output << tk::defcolor

#endif

std::vector<std::string> split(const std::string& s, char delim);
std::vector<std::string> split(const std::string& s);
std::pair<std::string, std::string> splitOffDigits(std::string s);
bool endsWith(std::string str, std::string key);
std::string intToString(int number, int minLength = 0);
std::string floatToString(float number);
int clamp(int val, int min_val, int max_val);
std::vector<std::string> getSubPaths(std::string basePath);
std::vector<std::string> getFiles(std::string path, std::string ext, bool silent = false);

inline float GetFPS(int64 t0, int64 t1) {
	return 1.0f / abs(t0 - t1)*cv::getTickFrequency();
}

inline float GetFPS(int64 tin) {
	return cv::getTickFrequency() / abs(tin);
}


inline cv::Matx44f ParseMat(const std::string& pose_str) {
	std::vector<std::string> strs = tk::split(pose_str, ' ');
	float m00 = (float)std::atof(strs[0].c_str());
	float m01 = (float)std::atof(strs[1].c_str());
	float m02 = (float)std::atof(strs[2].c_str());
	float m10 = (float)std::atof(strs[3].c_str());
	float m11 = (float)std::atof(strs[4].c_str());
	float m12 = (float)std::atof(strs[5].c_str());
	float m20 = (float)std::atof(strs[6].c_str());
	float m21 = (float)std::atof(strs[7].c_str());
	float m22 = (float)std::atof(strs[8].c_str());
	float m03 = (float)std::atof(strs[9].c_str());
	float m13 = (float)std::atof(strs[10].c_str());
	float m23 = (float)std::atof(strs[11].c_str());

	cv::Matx44f mat(
		m00, m01, m02, m03,
		m10, m11, m12, m13,
		m20, m21, m22, m23,
		0, 0, 0, 1);

	return mat;
}

} // namespace tk