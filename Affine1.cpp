#include "Affine1.h"
#include <random>


template<typename T>
Affine<T>::Affine(const int inputSize, const int outputSize, T* input, T* d_input, T* output, T* d_output)
	:m_inputSize(inputSize)
    ,m_outputSize(outputSize)
    ,m_length((m_inputSize + 1) * m_outputSize)
    ,m_input (input)
    ,m_output (output)
    ,m_d_input (d_input)
    ,m_d_output (d_output)
{
	if (inputSize < 1 || outputSize < 1 || /*!input ||*/ !output || !d_input || !d_output) {
		std::cout << " Alart: at Affine<T>::allocate(...) in " __FILE__ "\n"
			" inputSize=0 or outputSize=0 or (input,output,d_input,d_output)=nullptr \n"
			<< "inputSize  :" << inputSize << "\n"
			<< "outputSize :" << outputSize << "\n"
			<< "input      :" << (input == nullptr ? "nullptr" : " good") << "\n"
			<< "output     :" << (output == nullptr ? "nullptr" : " good") << "\n"
			<< "d_input    :" << (d_input == nullptr ? "nullptr" : " good") << "\n"
			<< "d_output   :" << (d_output == nullptr ? "nullptr" : " good") << "\n";
		throw std::runtime_error("Proble in Affine<T>::allocate(...)");
	}

	m_param = new T[m_length];
	try {
		m_d_param = new T[m_length];
	}
	catch (...) {
		delete[] m_param;
		throw;
	}
	clearDiff();
}

template<typename T>
Affine<T>::~Affine() {
	delete[] m_param;
	delete[] m_d_param;
}

template<typename T>
void Affine<T>::loadData(std::string filename) {
	std::ifstream ifs(filename,std::ios::binary);
	ifs.read(reinterpret_cast<char*>(m_param), sizeof(m_param[0]) * m_length);
	if (!ifs) {
		std::cout << " at void Affine<T>::loadData(std::string filename) in " __FILE__"\n"
			<< " problem loading bin file\n";
		throw std::runtime_error("Probrem loading bin file\n");
	}
}

template <typename T>
void Affine<T>::saveData(std::string filename) {
	std::ofstream ofs(filename,std::ios::binary);
	ofs.write(reinterpret_cast<char*>(m_param), sizeof(m_param[0]) * m_length);
	if (!ofs) {
		std::cout << "Affine<T>::saveData()->  Problem saving params at bin file\n";
		throw std::runtime_error("");
	}
}

template<typename T>
void Affine<T>::random() {
	std::random_device seed_gen;
	std::default_random_engine engine(seed_gen());
	std::normal_distribution<T> dist(static_cast<T>(0.), static_cast<T>(1.) / sqrt((T)m_inputSize));
	for (int i = 0; i < m_length; i++) {
		m_param[i] = dist(engine);
	}
}



template<typename T>
void Affine<T>::forward() {
	//std::cout << "Affine::forward()\n";
	T value;
	int idBase = 0;
	int i, j;
	for (i = 0; i < m_outputSize; i++) {
		value = m_param[idBase + m_inputSize];
		for (j = 0; j < m_inputSize; j++) {

			value += m_param[idBase + j] * m_input[j];
		}
		m_output[i] = value;
		idBase += (m_inputSize + 1);
	}
}


template<typename T>
void Affine<T>::backward() {
	//std::cout << "Affine::backward()"<<m_inputSize<<"\n";
	int i, j, id;
	for (i = 0; i < m_inputSize; i++) {
		m_d_input[i] = static_cast<T>(0.);
		id = i;
		for (j = 0; j < m_outputSize; j++) {
			m_d_input[i] += (m_d_output[j] * m_param[id]);
			m_d_param[id] += (m_input[i] * m_d_output[j]);
			id += (m_inputSize + 1);
		}
	}
	id = m_inputSize;
	for (i = 0; i < m_outputSize; i++) {
		m_d_param[id] += m_d_output[i];
		id += (m_inputSize + 1);
	}
}

template <typename T>
void Affine<T>::clearDiff() {
	for (int i = 0; i < m_length; i++) {
		m_d_param[i] = static_cast<T>(0.);
	}
}

template<typename T>
void Affine<T>::learn(T rate) {
	for (int i = 0; i < m_length; i++) {
		m_param[i] -= (rate * m_d_param[i]);
	}
	clearDiff();
}

template<typename T>
void Affine<T>::inputChanger(T* input_) {
	m_input = input_;
}

template class Affine<float>;
template class Affine<double>;
template class Affine<long double>;


