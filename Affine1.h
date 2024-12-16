#pragma once
#include <string>
#include "common.h"
template <typename T>
class Affine
{
private:

	T* m_input;
	T* m_d_input;
	T* m_output;
	T* m_d_output;

	T* m_param;
	T* m_d_param;

	const int m_inputSize;
	const int m_outputSize;

	const int m_length;
	void clearDiff();


public:
	//filename‚ğw’è‚µ‚È‚¯‚ê‚Îrandom‚É‰Šú‰»
	Affine(const int inputSize, const int outputSize, T* input, T* d_input, T* output, T* d_output);
	Affine() = delete;
	Affine(Affine&) = delete;
	Affine(Affine&&) = delete;
	~Affine();

	void forward();
	void backward();


	void learn(T rate);

	//save param data at csvfile

	void loadData(std::string filename);
	void saveData(std::string filename);
	void random();

	void inputChanger(T* input_);

};


