#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <random>
#include "common.h"

template<typename T>
class MnistData
{
private:

	const size_t dataSize = 28 * 28;

	std::vector<std::pair<T*, size_t>> trainData;
	std::vector<std::pair<T*, size_t>> testData;
	std::string folderName;

	void csvLoader(std::string filename, std::vector<std::pair<T*, size_t>>&data);

public:

	MnistData(std::string folderName);
	~MnistData();

	size_t get_trainDataNum();
	size_t get_testDataNum();
	std::vector<std::pair<T*, size_t>> get_MiniBatchData(size_t batchSize);
};

