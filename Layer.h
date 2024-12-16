#pragma once
#include "Activation.h"
#include "Affine1.h"
template<typename T>
class Layer
{
private:
	int m_inputSize;
	int m_outputSize;

	T* m_input;
	T* m_output;
	T* m_d_input;
	T* m_d_output;

	T* m_mid;
	T* m_d_mid;

	Activation<T> *activation;
	Affine<T> *affine;

public:
	Layer(int inputSize, int outputSize, T* input, T* output,T* d_input,T* d_output,std::string actType);
	~Layer();
	void random();
	void saveParam(std::string filename);
	void loadParam(std::string filename);

	void forward();
	void backward();

	void learn(T rate);

	void inputChanger(T* input_);
};

