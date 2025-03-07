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
	std::vector<std::pair<double*, size_t>> testData = mnistData.get_testData();
	
	double* input=nullptr;
	double* output = nullptr;
	double* d_input = nullptr;
	double* d_output = nullptr;
	
	NN<double> nn;
	nn.set_input(784, d_input);
	nn.addLayer(1000, "leakyRelu");
	nn.addLayer(800, "leakyRelu");
	nn.addLayer(600, "leakyRelu");
	nn.addLayer(400, "leakyRelu");
	nn.addLayer(200, "leakyRelu");
	nn.addLayer(10, "linear");
	nn.set_output(output, d_output);
	CrossEntropy<double> crossEntropy(10, output, d_output, true);

	//nn.random();
	nn.loadParam("./nn");
	
	/*
	for (size_t j = 0; j < testData.size(); j++) {
		nn.forward(testData[j].first);
		crossEntropy.forward();
		crossEntropy.set_ans(testData[j].second);
		if (!crossEntropy.isPredictionCorrect()) {
			std::cout << j << " ";
		}
	}*/

	
	for (;;) {
		int ID;
		std::cin >> ID;
		if (ID >= testData.size()) {
			continue;
		}
		mnistData.show(ID);
		nn.forward(testData[ID].first);
		crossEntropy.forward();
		crossEntropy.set_ans(testData[ID].second);
		std::cout << "predict   :" << crossEntropy.prediction() << "\n";
		std::cout << "confidence:" << crossEntropy.confidence() << "\n";
		std::cout << "ans       :" << testData[ID].second << "\n";
		std::cout << "\n";
	}
	int batchSize = 1000;
	int epock = 10000;
	for (int i = 0; i < epock; i++) {
		double loss = 0.;
		std::vector<std::pair<double*, size_t>> trainData = mnistData.get_MiniBatchData((size_t)batchSize);
		int numOfCorrect=0;
		for (int j = 0; j < batchSize; j++) {
			nn.forward(trainData[j].first);
			crossEntropy.forward();
			crossEntropy.set_ans(trainData[j].second);
			loss += crossEntropy.calcLoss();
			crossEntropy.backward();
			nn.backward();
			if (crossEntropy.isPredictionCorrect())numOfCorrect++;
		}
		std::cout << "loss    :" << loss << "\n";
		std::cout << "accuracy:" << (double)numOfCorrect/(double)batchSize << "\n";
		nn.learn(0.01 / (double)batchSize);

		if (i % 100 == 99) {
			std::cout << "========================================================\n";
			std::cout << "test\n";
			numOfCorrect = 0;
			for (size_t j = 0; j < testData.size(); j++) {
				nn.forward(testData[j].first);
				crossEntropy.forward();
				crossEntropy.set_ans(testData[j].second);
				if (crossEntropy.isPredictionCorrect())numOfCorrect++;
			}

			std::cout << "loss    :" << loss << "\n";
			std::cout << "accuracy:" << (double)numOfCorrect / (double)testData.size() << "\n";
			std::cout << "========================================================\n";
			nn.saveParam("./nn");
		}
	}
	nn.saveParam("./nn");

}


