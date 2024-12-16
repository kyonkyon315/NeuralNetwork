#include "CrossEntropy.h"
template<typename T>
CrossEntropy<T>::CrossEntropy(size_t inputSize, T* input, T* d_input,const bool isTeacherOneHot_) 
	:isTeacherOneHot(isTeacherOneHot_)
	,m_maxV()
	,m_sum()

{
	if (inputSize == 0) {
		std::cout << "Alart: at CrossEntropy<T>::CrossEntropy() in " __FILE__ "\n"
			"inputSize must be larger than 0.\n";
		throw std::runtime_error("");
	}
	if (!input || !d_input) {
		std::cout << "Alart: at CrossEntropy<T>::CrossEntropy() in " __FILE__ "\n"
			"input or d_input must not be nullptr.\n";
		throw std::runtime_error("");
	}

	m_inputSize = inputSize;
	m_input = input;
	m_d_input = d_input;
	ansId = 0;
	try {
		m_y = new T[inputSize];
		if (isTeacherOneHot) {
			m_ans = new T[inputSize];
			for (int i = 0; i < inputSize; i++) {
				m_ans[i] = static_cast<T>(0.);
			}
		}
		else {
			m_ans = nullptr;
		}
	}
	catch (...) {
		this->~CrossEntropy();
		throw;
	}
}

template <typename T>
CrossEntropy<T>::~CrossEntropy() {
	delete[] m_y;
	if (isTeacherOneHot) {
		delete[] m_ans;
	}
}

template <typename T>
void CrossEntropy<T>::forward() {
	m_maxV = *std::max_element(m_input, m_input + m_inputSize);
	m_sum = static_cast<T>(0.);
	for (int i = 0; i < m_inputSize; i++) {
		m_y[i] = exp(m_input[i] - m_maxV);
		m_sum += m_y[i];
	}
}

template <typename T>
void CrossEntropy<T>::set_ans(T* ans) {
	if (isTeacherOneHot) {
		std::cout << "Alart: at CrossEntropy<T>::set_ans(T* ans) in " __FILE__ "\n"
			"This function can't be called if isTeacherOneHot==true.\n";
		throw std::runtime_error("");
	}
	if (!m_ans) {
		std::cout << "Alart: at CrossEntropy<T>::set_ans(T* ans) in " __FILE__ "\n"
			"ans must not be nullptr.\n";
		throw std::runtime_error("");
	}
	m_ans = ans;
}


template <typename T>
void CrossEntropy<T>::set_ans(size_t id) {
	if (!isTeacherOneHot) {
		std::cout << "Alart: at CrossEntropy<T>::set_ans(size_t id) in " __FILE__ "\n"
			"This function can't be called if isTeacherOneHot==false.\n";
		throw std::runtime_error("");
	}
	if (id>=m_inputSize) {
		std::cout << "Alart: at CrossEntropy<T>::set_ans(size_t id) in " __FILE__ "\n"
			"id must be smaller than m_inputSize.\n";
		throw std::runtime_error("");
	}
	m_ans[ansId] = static_cast<T>(0.);
	m_ans[id] = static_cast<T>(1.);
	ansId = id;
}

template <typename T>
bool CrossEntropy<T>::isPredictionCorrect() {
	if (isTeacherOneHot) {
		return ansId == (size_t)(std::max_element(m_y, m_y + m_inputSize) - m_y);
	}
	else {
		return
			std::max_element(m_ans, m_ans + m_inputSize) - m_ans
			==
			std::max_element(m_y, m_y + m_inputSize) - m_y;
	}
}

template <typename T>
size_t CrossEntropy<T>::prediction() {
	
	return  (size_t)(std::max_element(m_y, m_y + m_inputSize) - m_y);

}

template <typename T>
T CrossEntropy<T>::confidence() {

	return  *std::max_element(m_y, m_y + m_inputSize)/m_sum;

}

template <typename T>
T CrossEntropy<T>::calcLoss() {
	T loss = static_cast<T>(0.);
	for (int i = 0; i < m_inputSize; i++) {
		loss -= m_ans[i]*(m_input[i]-m_maxV);
	}
	loss += log(m_sum);
	return loss;
}

template <typename T>
void CrossEntropy<T>::backward() {
	for (int i = 0; i < m_inputSize; i++) {
		m_d_input[i] = -m_ans[i] + (m_y[i] / m_sum);
	}
}

template class CrossEntropy<double>;
template class CrossEntropy<long double>;
template class CrossEntropy<float>;

