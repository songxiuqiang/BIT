#include <iterator>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>

#ifdef WIN32
#include "base/dirent_win.h"
#else
#include <dirent.h>
#endif
#include "util.hh"

namespace tk {

std::vector<std::string> split(const std::string& s, char delim) {
	std::vector<std::string> elems;
	std::stringstream ss(s);
	std::string item;

	while (std::getline(ss, item, delim)) elems.push_back(item);

	return elems;
}

std::vector<std::string> split(const std::string& s) {
	std::istringstream iss(s);
	std::vector<std::string> elems;

	std::copy(
		std::istream_iterator<std::string>(iss),
		std::istream_iterator<std::string>(),
		std::back_inserter<std::vector<std::string> >(elems));

	return elems;
}

std::pair<std::string, std::string> splitOffDigits(std::string s) {
	// find first number
	int splitIndex = -1;
	for (int i = 0; i < (int)s.length(); i++) {
		char c = s[i];
		if (('0' <= c && c <= '9') || (c == '.')) {
			splitIndex = i;
			break;
		}
	}

	// split before first number
	if (splitIndex == -1)
		return std::pair<std::string, std::string>(s, "");
	else
		return std::pair<std::string, std::string>(s.substr(0, splitIndex), s.substr(splitIndex, s.length() - splitIndex));
}

bool endsWith(std::string str, std::string key) {
	size_t keylen = key.length();
	size_t strlen = str.length();

	if (keylen <= strlen)
		return str.substr(strlen - keylen, keylen) == key;
	else
		return false;
}

std::string intToString(int number, int minLength) {
	std::stringstream ss; //create a stringstream
	ss << number; //add number to the stream
	std::string out = ss.str();
	while ((int)out.length() < minLength) out = "0" + out;
	return out; //return a string with the contents of the stream
}

std::string floatToString(float number) {
	std::stringstream ss; //create a stringstream
	ss << number; //add number to the stream
	return ss.str(); //return a string with the contents of the stream
}

int clamp(int val, int min_val, int max_val) {
	return std::max(min_val, std::min(max_val, val));
}

std::vector<std::string> getSubPaths(std::string path) {
	std::vector<std::string> subPaths;

	DIR *dir = opendir(path.c_str());
	struct dirent *ent;

	if (dir != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			std::string entry = ent->d_name;
			if (entry.find(".") == std::string::npos)
				subPaths.push_back(path + entry);
		}
		closedir(dir);
	}
	else
		std::cout << "Could not open directory: " << path << std::endl;

	std::sort(subPaths.begin(), subPaths.end());
	return subPaths;
}

std::vector<std::string> getFiles(std::string path, std::string ext, bool silent) {
	std::vector<std::string> files;

	DIR *dir = opendir(path.c_str());
	struct dirent *ent;

	if (dir != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			std::string entry = ent->d_name;
			if (endsWith(entry, ext))
				files.push_back(path + entry);
		}
		closedir(dir);
	}	else if (!silent) 
		std::cout << "Could not open directory: " << path << std::endl;

	std::sort(files.begin(), files.end());
	return files;
}

}