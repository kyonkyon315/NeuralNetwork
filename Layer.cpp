#include "Layer.h"

template<typename T>
Layer<T>::Layer(int inputSize, int outputSize, T* input, T* output,T* d_input,T* d_output,std::string actType) 
	:m_inputSize(inputSize)
	,m_outputSize(outputSize)
	,m_input(input)
	,m_output(output)
	,m_d_input(d_input)
	,m_d_output(d_output)

	,m_mid()
	,m_d_mid()
	,activation()
	,affine()
{
	if (inputSize < 1 || outputSize < 1 || /*!input || */ !output || !d_input || !d_output) {
		std::cout << " Alart: at Layer<T>::Layer(..) in " __FILE__ "\n"
			" inputSize=0 or outputSize=0 or (input,output,d_input,d_output)=nullptr \n"
			<< "inputSize  :" << inputSize << "\n"
			<< "outputSize :" << outputSize << "\n"
			<< "input      :" << (input == nullptr ? "nullptr" : " good") << "\n"
			<< "output     :" << (output == nullptr ? "nullptr" : " good") << "\n"
			<< "d_input    :" << (d_input == nullptr ? "nullptr" : " good") << "\n"
			<< "d_output   :" << (d_output == nullptr ? "nullptr" : " good") << "\n";
		throw std::runtime_error("Proble in Layer<T>::Layer(..)");
	}
	
	try {
		m_mid = new T[m_outputSize];
		m_d_mid = new T[m_outputSize];
		affine = new Affine<T>(m_inputSize, m_outputSize, m_input, m_d_input, m_mid, m_d_mid);
		activation = new Activation<T>((unsigned int)m_outputSize,m_mid,m_d_mid,m_output,m_d_output,actType);
	}
	catch (...) {
		this->~Layer();
		throw;
	}
}

template<typename T>
Layer<T>::~Layer() {
	delete[] activation;
	delete[] affine;
	delete[] m_mid;
	delete[] m_d_mid;
}

template<typename T>
void Layer<T>::loadParam(std::string filename) {
	affine->loadData(filename);
}

template<typename T>
void Layer<T>::saveParam(std::string filename) {
	affine->saveData(filename);
}

template<typename T>
void Layer<T>::forward() {
	//std::cout << "Layer::forward()\n";
	affine->forward();
	activation->forward();
}

template<typename T>
void Layer<T>::backward() {
	//std::cout << "Layer::backward()\n";
	activation->backward();
	affine->backward();
}

template<typename T>
void Layer<T>::random() {
	affine->random();
}

template<typename T>
void Layer<T> ::learn(T rate) {
	affine->learn(rate);
}

template<typename T>
void Layer<T>::inputChanger(T* input_) {
	affine->inputChanger(input_);
}


template class Layer<double>;
template class Layer<long double>;
template class Layer<float>;


