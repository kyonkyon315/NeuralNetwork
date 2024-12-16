#include "MnistData.h"
template<typename T>
void MnistData<T>::csvLoader(std::string filename, std::vector<std::pair<T*, size_t>>& data) {
	if (std::ifstream ifs{filename}) {
		int dataNum = 0;
		size_t i;
		char chara;
		size_t label;
		int k;
		try {
			for (;;) {
				T* onedata = new T[dataSize];
				ifs >> label;
				std::cout << label << " ";

				for (i = 0; i < dataSize; i++) {
					ifs >> ',' >> k;
					std::cout << " " << k;
					onedata[i] = (T)k;
				}
				data.push_back(std::pair<T*, size_t>(onedata,label));
				ifs >> chara;
				std::cout << "\n";

				if (chara != '\n') {
					break;
				}
			}
		}

		catch (...) {
			for (i = 0; i < data.size(); i++) {
				delete[] data[i].first;
			}
			throw;
		}
	}
}

template<typename T>
MnistData<T>::MnistData(std::string folderName) {
	csvLoader(folderName + "/mnist_train.csv", trainData);
	csvLoader(folderName + "/mnist_test.csv", testData);
}

template<typename T>
MnistData<T>::~MnistData() {
	for (size_t i = 0; i < trainData.size(); i++) {
		delete[] trainData[i].first;
	}
	for (size_t i = 0; i < testData.size(); i++) {
		delete[] testData[i].first;
	}
}

template<typename T>
size_t MnistData<T>::get_trainDataNum() {
	return trainData.size();
}

template<typename T>
size_t MnistData<T>::get_testDataNum() {
	return testData.size();
}
template<typename T>
std::vector<std::pair<T*, size_t>> MnistData<T>::get_MiniBatchData(size_t batchSize) {
	if (batchSize > get_trainDataNum()) {
		std::cout << " Alart :  at std::vector<std::pair<T*, size_t>>"
			" MnistData<T>::get_MiniBatchData(size_t batchSize) {\n"
			" in " __FILE__ "\n"
			"batchSize must not be larger than number of trainData\n"
			"batchSize is recognized as number of trainData.\n";
	}
	std::vector<std::pair<T*, size_t>>retval(batchSize);


	int numNow = 0;
	size_t numOfTrainData=trainData.size();
	for (size_t i = 0; i < numOfTrainData; i++) {
		if ((double)rand()/(double)RAND_MAX < ((double)(batchSize - numNow) / (double)(numOfTrainData - i))) {
			retval[numNow] = trainData[i];
			numNow++;
			if (numNow == batchSize)break;
		}
	}
	return retval;
}


template class MnistData<double>;
template class MnistData<long double>;
template class MnistData<float>;

