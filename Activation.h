#pragma once
#include <string>
#include <stdexcept>
#include <iostream>
template<typename T>
class Activation
{
private:
	void m_relu_forward();
	void m_relu_backward();
	const unsigned int m_length;
	T*  m_input;
	T*  m_d_input;
	T*  m_output;
	T*  m_d_output;
	std::string m_typeName;
public:
	Activation(const unsigned int length,T* input,T* d_input,T* output,T* d_output,std::string typeName);
	Activation(Activation&) = delete;
	Activation(Activation&&) = delete;
	Activation& operator=(const Activation& r) = delete;
	Activation& operator=(Activation&& r) = delete;
	~Activation()=default;

	void forward();
	void backward();
};
