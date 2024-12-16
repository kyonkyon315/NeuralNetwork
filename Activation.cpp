#include "Activation.h"

template <typename T>
Activation<T>::Activation(const unsigned int length, T* input, T* d_input, T* output, T* d_output, std::string typeName) 
	:
	m_length(length), m_input(input), m_d_input(d_input), m_output(output), m_d_output(d_output), m_typeName(typeName)
{

}


template<typename T>
void Activation<T>::m_relu_forward() {
	for (unsigned int i = 0; i < m_length; i++) {
		m_output[i] = std::max(static_cast<T>(0.), m_input[i]);
	}
}

template<typename T>
void Activation<T>::m_relu_backward() {
	//std::cout << "Activation::backward()"<<m_length<<"\n";
	for (unsigned int i = 0; i < m_length; i++) {
		m_d_input[i] = (m_input[i] > static_cast<T>(0.) ? m_d_output[i] : static_cast<T>(0.));
	}
}

template<typename T>
void Activation<T>::forward() {
	//std::cout << "Activation::forward()\n";
	if (m_typeName == "relu") {
		m_relu_forward();
	}
	else {
		std::cout << "Alart: at Activation<T>::forward() in " __FILE__ "\n"
			"m_typeName is not valid.\n"
			"m_typeName:" << m_typeName << "\n";
		throw std::runtime_error("Problem in NN<T>::forward()");
	}
}

template<typename T>
void Activation<T>::backward() {
	//std::cout << "Activation::backward()\n";
	if (m_typeName == "relu") {
		m_relu_backward();
	}
	else {
		throw std::runtime_error("");
	}
}

template class Activation<double>;
template class Activation<float>;
template class Activation<long double>;


