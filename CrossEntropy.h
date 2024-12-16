#pragma once
#include <stdexcept>
#include <iostream>
template <typename T>
class CrossEntropy
{
private:
	size_t m_inputSize;
	T* m_input;
	T* m_d_input;
	T* m_ans;
	T* m_y;
	T m_sum;
	T m_maxV;

	const bool isTeacherOneHot;
	size_t ansId;
public:
	CrossEntropy(size_t inputSize,T* input,T* d_input,const bool isTeacherOneHot_=false);
	~CrossEntropy();

	void forward();
	bool isPredictionCorrect();
	size_t prediction();
	T confidence();
	void backward();

	void set_ans(T* ans);
	void set_ans(size_t id);

	T calcLoss();

};

