#include "NN.h"
#include <filesystem>
#include <direct.h>
#include <stdlib.h>
#include <stdio.h>
template<typename T>
void NN<T>::set_input(int inputSize, T* &d_input, bool allocate) {
	if (inputIsSet) {
		std::cout << "Alart: at NN<T>::set_input() in " __FILE__ "\n"
			"it can't be called after \"addLayer\" or \"set_output\" is called.\n";
		throw std::runtime_error("Problem in NN<T>::set_input()");
	}
	if (inputSize < 1) {
		std::cout << " Alart: at NN<T>::set_input() in " __FILE__ "\n"
			"inputSize<1\n";
		throw std::runtime_error("Problem in NN<T>::addLayer()");
	}
	if (allocate&&(d_input)) {
		std::cout << " Alart: at NN<T>::set_input() in " __FILE__ "\n"
			" d_input must be nullptr if reAllocate==true. \n"
			<< "d_input    :" << (d_input != nullptr ? "!nullptr" : " good") << "\n";
		throw std::runtime_error("Problem in NN<T>::set_input()");
	}
	else if (!allocate && !d_input) {
		std::cout << " Alart: at NN<T>::set_input() in " __FILE__ "\n"
			" d_input must be allocated already if reAllocate==false. \n"
			<< "d_input    :" << (d_input == nullptr ? "nullptr" : " good") << "\n";
		throw std::runtime_error("Problem in NN<T>::set_input()");
	}
	try {
		if (allocate) {
			neurons.push_back(nullptr);
			d_neurons.push_back(new T[inputSize]);
			d_input = d_neurons[0];
		}
		else {
			neurons.push_back(nullptr);
			d_neurons.push_back(d_input);
		}
		toporogy.push_back(inputSize);
	}
	catch (...) {
		this->~NN();
		throw;
	}
	inputIsSet = true;
}

template<typename T>
void NN<T>::addLayer(const int outputSize, std::string activationType) {
	if (outputSize < 1) {
		std::cout << " Alart: at NN<T>::set_input() in " __FILE__ "\n"
			"outputSize<1\n";
		throw std::runtime_error("Problem in NN<T>::addLayer()");
	}
	if (!inputIsSet||outputIsSet) {
		std::cout << "Alart: at NN<T>::addLayer() in " __FILE__ "\n"
			"it can't be called before \"set_input\" is called or after \"set_output\" is called.\n";
		throw std::runtime_error("Problem in NN<T>::addLayer()");
	}
	try {
		neurons.push_back(new T[outputSize]);
		d_neurons.push_back(new T[outputSize]);
		toporogy.push_back(outputSize);
		layers.push_back(
			new Layer<T>(toporogy[toporogy.size() - 2]
				, toporogy[toporogy.size() - 1]
				, neurons[neurons.size() - 2]
				, neurons[neurons.size() - 1]
				, d_neurons[d_neurons.size() - 2]
				, d_neurons[d_neurons.size() - 1]
				, activationType));
		activationTypes.push_back(activationType);
	}
	catch (...) {
		this->~NN();
		throw;
	}
}

template <typename T>
void NN<T>::set_output(T* &output,T* &d_output,bool allocate) {
	if (!inputIsSet ) {
		std::cout << "Alart: at NN<T>::set_output() in " __FILE__ "\n"
			"it can't be called before \"set_input\" is called.\n";
		throw std::runtime_error("Problem in NN<T>::set_output()");
	}
	if (allocate&&(output || d_output)) {
		std::cout << " Alart: at NN<T>::set_output() in " __FILE__ "\n"
			" output and d_output must be nullptr if allocate==true. \n"
			<< "output     :" << (output != nullptr ? "!nullptr" : " good") << "\n"
			<< "d_output   :" << (d_output != nullptr ? "!nullptr" : " good") << "\n";
		throw std::runtime_error("Problem in NN<T>::set_output()");
	}
	else if (!allocate && (!output || !d_output)) {
		std::cout << " Alart: at NN<T>::set_output() in " __FILE__ "\n"
			" output and d_output must be allocated already if allocate==false. \n"
			<< "output     :" << (output == nullptr ? "nullptr" : " good") << "\n"
			<< "d_output   :" << (d_output == nullptr ? "nullptr" : " good") << "\n";
		throw std::runtime_error("Problem in NN<T>::set_output()");
	}
	if (allocate) {
		output = neurons[neurons.size() - 1];
		d_output = d_neurons[d_neurons.size() - 1];
	}
	else {
		delete[] neurons[neurons.size() - 1];
		delete[] d_neurons[d_neurons.size() - 1];
		neurons[neurons.size() - 1]=output;
		d_neurons[d_neurons.size() - 1]=d_output;
	}
	outputIsSet = true;
}

template <typename T>
NN<T>::~NN() {
	std::vector<Layer<T>*>layers;
	std::vector<T*>neurons;
	std::vector<T*>d_neurons;
	std::vector<std::string>activationTypes;
	for (size_t i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
	for (size_t i = 0; i < neurons.size(); i++) {
		delete neurons[i];
	}
	for (size_t i = 0; i < d_neurons.size(); i++) {
		delete d_neurons[i];
	}
}

template<typename T>
void NN<T>::forward(T* input) {
	if (!outputIsSet) {
		std::cout << "Alart: at NN<T>::forward() in " __FILE__ "\n"
			"it can't be called before \"set_output\" is called.\n";
		throw std::runtime_error("Problem in NN<T>::forward()");
	}
	if (layers.size() > 0) {
		layers[0]->inputChanger(input);
		/*
		neurons[0] = input;
		layers[0]->m_input = input;
		layers[0]->affine->m_input = input;
		*/
	}
	for (size_t i = 0; i < layers.size(); i++) {
		layers[i]->forward();
	}
}

template<typename T>
void NN<T>::backward() {
	if (!outputIsSet) {
		std::cout << "Alart: at NN<T>::forward() in " __FILE__ "\n"
			"it can't be called before \" NN<T>::set_output()\" is executed.\n";
		throw std::runtime_error("Problem in NN<T>::backward()");
	}
	for (size_t i = 0; i < layers.size(); i++) {
		layers[layers.size()-i-1]->backward();
	}
}

template<typename T>
void NN<T>::learn(T rate) {
	if (!outputIsSet) {
		std::cout << "Alart: at NN<T>::forward() in " __FILE__ "\n"
			"it can't be called before \"set_output\" is called.\n";
		throw std::runtime_error("Problem in NN<T>::forward()");
	}
	for (size_t i = 0; i < layers.size(); i++) {
		layers[i]->learn(rate);
	}
}

template<typename T>
void NN<T>::random() {
	if (!outputIsSet) {
		std::cout << "Alart: at NN<T>::forward() in " __FILE__ "\n"
			"it can't be called before \"set_output\" is called.\n";
		throw std::runtime_error("Problem in NN<T>::forward()");
	}
	for (size_t i = 0; i < layers.size(); i++) {
		layers[i]->random();
	}
}


static void CreateFolder(std::string FolderName)
{
	const char* foldername = FolderName.c_str();
	_mkdir(foldername);
	/*
	if (_mkdir(foldername) == 0) {
		std::cout << "Directory '" << FolderName << "' was successfully created\n";
	}
	else {
		std::cout << "Problem creating directory '" << FolderName << "' .\n";
	}
	*/
}
template<typename T>
void NN<T>::saveParam(std::string folderName) {
	CreateFolder(folderName);
	for (size_t i = 0; i < layers.size(); i++) {
		layers[i]->saveParam(folderName+"/"+std::to_string(i)+".bin");
	}
}

template<typename T>
void NN<T>::loadParam(std::string folderName) {
	for (size_t i = 0; i < layers.size(); i++) {
		layers[i]->loadParam(folderName + "/" + std::to_string(i) + ".bin");
	}
}



template class NN<double>;
template class NN<long double>;
template class NN<float>;

