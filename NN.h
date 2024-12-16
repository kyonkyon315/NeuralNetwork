#pragma once
#include <vector>
#include "Layer.h"
template<typename T>
class NN
{
private:

	

	bool inputIsSet = false;
	bool outputIsSet = false;

	std::vector<Layer<T>*>layers;
	std::vector<T*>neurons;
	std::vector<T*>d_neurons;
	std::vector<std::string>activationTypes;

	std::vector<int>toporogy;


public:

	~NN();

	void set_input(const int inputSize, T* &d_input,bool allocate=true);
	void addLayer(const int outputSize, std::string activationType);
	void set_output(T* &output,T* &d_output,bool allocate=true);

	void forward(T* input);
	void backward();

	void learn(T rate);

	void saveParam(std::string folderName);
	void loadParam(std::string folderName);
	void random();

};

