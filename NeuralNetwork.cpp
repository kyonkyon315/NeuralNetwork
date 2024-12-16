#include <iostream>
#include <random>
#include "Affine1.h"
#include "Timer.h"
#include "NN.h"
#include "CrossEntropy.h"
#include "MnistData.h"

int main()
{
	std::string dataPath = "C:\\Users\\sugin\\source\\repos\\NeuralNetwork\\x64\\Debug\\MNIST_CSV";
	MnistData<double> mnistData(dataPath);
	std::random_device seed_gen;
	std::default_random_engine engine(seed_gen());
	std::normal_distribution<double> dist(static_cast<double>(0.), static_cast<double>(1.) / sqrt((double)100));
	//std::cout<<dist(engine);
	
	const int n = 100000;
	//make traindata
	double** x = new double* [n];
	int* y = new int[n];
	for (int i = 0; i < n; i++) {
		x[i] = new double[2];
	}
	for (int i = 0; i < n; i++) {
		x[i][0] = (double)rand()/(double)RAND_MAX;
		x[i][1] = (double)rand() / (double)RAND_MAX;
		y[i] = (x[i][0] > x[i][1] ? 0 : 1);
		//y[i] = 1 ;
	}

	int inputSize = 2;
	int outputSize = 1;
	double* input=nullptr;
	double* output = nullptr;
	double* d_input = nullptr;
	double* d_output = nullptr;
	
	NN<double> nn;
	nn.set_input(2, d_input);
	nn.addLayer(100, "relu");
	nn.addLayer(100, "relu");
	nn.addLayer(2, "relu");
	nn.set_output(output, d_output);
	CrossEntropy<double> crossEntropy(2, output, d_output, true);

	//nn.random();
	nn.loadParam("./nn");

	
	double test[2] = { 0.88,0.89 };
	nn.forward(test);
	crossEntropy.forward();
	std::cout << "input     : {" << test[0] << ", " << test[1] << "}\n";
	std::cout << "predict   : " << crossEntropy.prediction() << "\n";
	std::cout << "confidence: " << crossEntropy.confidence() << "\n";
	std::cout << "ans       : " << (test[0] > test[1] ? 0:1) <<"\n";
	return 0;
	int epock = 100;
	for (int i = 0; i < epock; i++) {
		double loss = 0.;
		int numOfCorrect = 0;
		for (int j = 0; j < n; j++) {
			nn.forward(x[j]);
			crossEntropy.forward();
			crossEntropy.set_ans((size_t)y[j]);
			loss += crossEntropy.calcLoss();
			crossEntropy.backward();
			nn.backward();
			if (crossEntropy.isPredictionCorrect())numOfCorrect++;
		}
		std::cout << "loss    :" << loss << "\n";
		std::cout << "accuracy:" << (double)numOfCorrect/(double)n << "\n";
		nn.learn(0.1 / (double)n);
	}
	
	nn.saveParam("./nn");

}


