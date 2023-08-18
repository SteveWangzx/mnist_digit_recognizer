﻿/// 库
#include <iostream>
#include <string>
#include <windows.h>
#include <vector>
#include <list>
#include <fstream>
#include <sstream>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <random>

/// 头文件
#include "dataCollector.hpp"
#include "fcLayer.hpp"
#include "actvLayer.hpp"

const int SIZE_TRAIN = 33600;
const int SIZE_TEST = 8400;

using std::vector;
using std::list;

int main()
{
	std::cout << "收集训练集" << std::endl;
	dataCollector train("./mnist_digit_train.csv");

	train.collectData();

	vector<fcLayer> graph;
	// 第一层
	graph.push_back(fcLayer(INPUT_SIZE, 30, Actv::RELU));
	actvLayer actvFirst(30, Actv::RELU);
	// 第二层
	graph.push_back(fcLayer(30, 15, Actv::RELU));
	actvLayer actvSecond(15, Actv::RELU);
	// 第三层
	graph.push_back(fcLayer(15, 10, Actv::SIGMOID));
	actvLayer actvOut(10, Actv::SIGMOID);
	const int graphSize = graph.size();

	// training parameters
	const int batch_size = 16;

	// random engine
	std::random_device rd;
	std::default_random_engine eng(rd());
	std::uniform_int_distribution<int> u(0, SIZE_TRAIN - 1);

	// train
	list<float> lossList;
	for (int itr = 0; itr < 10000; ++itr)
	{
		static float ecof = 0.00001f;
		ecof += 0.002;
		ecof = ecof > 1 ? 1 : ecof;
		vector<image> samples;
		for (int idx = 0; idx < batch_size; ++idx)
		{
			int row = u(eng);
			image curr = train.getImage(row);
			vector<float> input;
			for (int i = 0; i < INPUT_SIZE; ++i)
			{
				input.push_back(curr.x[i]);
			}

			// 第一层
			///////////////////////////////////
			// linear compute
			graph[0].SetX(input);
			graph[0].Forward();
			// activation: RELU
			actvFirst.setInput(graph[0].GetY());
			actvFirst.forward_compute();

			// 第二层
			/////////////////////////////////////
			graph[1].SetX(actvFirst.getOutput());
			graph[1].Forward();
			actvSecond.setInput(graph[1].GetY());
			actvSecond.forward_compute();

			// 第三层
			///////////////////////////////////////
			graph[2].SetX(actvFirst.getOutput());
			graph[2].Forward();
			actvOut.setInput(graph[2].GetY());
			actvOut.forward_compute();

			// 输出
			vector<float> output = actvOut.getOutput();
			int label[10] = { 0 };
			label[curr.y] = 1;
			vector<float> loss;

			//// softmax输出
			//// softmax(x) = e^x / e
			//float e_sum = 0.0f;
			//for (int i = 0; i < output.size(); ++i)
			//{
			//	e_sum += expf(output[i]);
			//}
			//for (int i = 0; i < output.size(); ++i)
			//{
			//	output[i] = expf(output[i]) / e_sum;
			//}

			//// softmax loss
			//float lossSum = 0.0f;
			//for (int i = 0; i < output.size(); ++i)
			//{
			//	lossSum += label[i] * log(output[i]);
			//}

			//// loss计算
			//for (int i = 0; i < 10; ++i)
			//{
			//	//loss.push_back(-(label[i] * log(output[i])));
			//	loss.push_back(-lossSum / 10);
			//}

			//float avgLoss = fabs(-lossSum / 10);
			//for (float data : loss)
			//{
			//	avgLoss += fabs(data);
			//}
			//avgLoss /= 10;
			float avgLoss = 0.0f;
			// 差值loss计算
			for (int i = 0; i < output.size(); ++i)
			{
				if (label[i] == 1){
					loss.push_back((label[i] - output[i]));
				}
				else{
					
					loss.push_back((label[i] - output[i]) * ecof);
				}
				
				
				if (label[i] == 1) { avgLoss = abs(label[i] - output[i]); }
			}
			
			/*for (float data : loss)
			{
				avgLoss += fabs(data);
			}
			avgLoss /= 10;*/

			// backward
			actvOut.setDactv(loss);
			actvOut.backward_compute();
			graph[2].SetDY(actvOut.getDactv());
			graph[2].Backward();

			actvSecond.setDactv(graph[2].GetDX());
			actvSecond.backward_compute();
			graph[1].SetDY(actvSecond.getDactv());
			graph[1].Backward();

			actvFirst.setDactv(graph[1].GetDX());
			actvFirst.backward_compute();
			graph[0].SetDY(actvFirst.getDactv());
			graph[0].Backward();

			// display avgrage loss for last 100 loss
			{
				lossList.push_back(avgLoss);
				if (lossList.size() > 100) { lossList.pop_front(); }
				float tloss = 0;
				for (float l : lossList) { tloss += l; }
				tloss /= lossList.size();
				printf("Iterations %d, avgloss:%f\r", itr, tloss);
				::Sleep(1);
			}
		}

		// batch结束更新参数
		for (int i = 0; i < graphSize; ++i)
		{
			graph[i].Update();
		}

		//vector<float> bias = graph[graphSize - 2].GetBias();
		//printf("Bias: ");
		//for (float bia : bias)
		//{
		//	printf("%f ", bia);
		//}
		//printf("\n");

	}
	std::cout << "Trained Complete!" << std::endl;

	// 测试
	////////////////////////////////////////////
	std::cout << "Run test!" << std::endl;
	std::cout << "收集测试集" << std::endl;
	dataCollector test("./mnist_digit_test.csv");
	test.collectData();
	int correct = 0;
	int cnt = 0;

	for (int i = 0; i < SIZE_TEST; ++i)
	{
		image curr = test.getImage(i);

		vector<float> test_input;
		for (int i = 0; i < INPUT_SIZE; ++i)
		{
			test_input.push_back(curr.x[i]);
		}

		graph[0].SetX(test_input);
		graph[0].Forward();
		for (int idx = 1; idx < graphSize; ++idx)
		{
			graph[idx].SetX(graph[idx - 1].GetY());
			graph[idx].Forward();
		}

		vector<float> result = graph[graphSize - 1].GetY();
		int predict = 0;
		float currMax = 0.0f;

		// 得到predict
		for (int k = 0; k < result.size(); ++k)
		{
			if (result[k] > currMax)
			{
				currMax = result[k];
				predict = k;
			}
		}
		if (predict == curr.y)
		{
			++correct;
		}
		++cnt;
		if ((i + 1) % 100 == 0)
		{
			float correctPercentage = (float)correct / (i + 1);
			correctPercentage *= 100.0f;
			printf("	Correctness : %.2f %\r", correctPercentage);
		}
	}
	float correctPercentage = (float)correct / (cnt + 1);
	correctPercentage *= 100.0f;
	printf("	Correctness : %.2f %\r", correctPercentage);
}