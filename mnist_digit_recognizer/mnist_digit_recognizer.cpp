#ifndef _main_cpp__
#define _main_cpp__


/// 库
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
#include <filesystem>

using std::vector;
using std::list;
namespace fs = std::filesystem;

/// 头文件
#include "public/Define.hpp"
#include "dataCollector/dataCollector.hpp"
#include "compute/gemm.hpp"
#include "layers/layer.hpp"
#include "layers/fcLayer.hpp"
#include "layers/actvLayer.hpp"
#include "layers/convLayer.hpp"
#include "layers/softmaxLayer.hpp"
#include "CNNGraph/Graph.hpp"

int main()
{
	// 数据处理
	//////////////////////////////////////
	std::cout << "收集训练集" << std::endl;
	dataCollector train("./mnist_digit_train.csv");

	//train.collectData();

	// 建立神经网络
	// /////////////////////////////////
	// 第一层conv1	----	1 * 28 * 28 -> 8 * 14 * 14
	convLayer conv1_1(28, 28, 1, 3, 3, 2, 1, 8);
	dim conv1_1Dim = conv1_1.getOutputDim();

	actvLayer conv1_1Actv(conv1_1Dim.height * conv1_1Dim.width * conv1_1Dim.channel, Actv::RELU);

	//convLayer conv1_2(conv1_1Dim.width, conv1_1Dim.height, conv1_1Dim.channel, 3, 3, 1, 1, 8);
	//dim conv1_2Dim = conv1_2.getOutputDim();

	//actvLayer conv1_2Actv(conv1_2Dim.height * conv1_2Dim.width * conv1_2Dim.channel, Actv::RELU);

	// 第二层conv2    ----	8 * 14 * 14 -> 16 * 7 * 7
	convLayer conv2_1(conv1_1Dim.width, conv1_1Dim.height, conv1_1Dim.channel, 3, 3, 2, 1, 32);
	dim conv2_1Dim = conv2_1.getOutputDim();

	actvLayer conv2_1Actv(conv2_1Dim.height * conv2_1Dim.width * conv2_1Dim.channel, Actv::RELU);

	//convLayer conv2_2(conv2_1Dim.width, conv2_1Dim.height, conv2_1Dim.channel, 3, 3, 1, 1, 16);
	//dim conv2_2Dim = conv2_2.getOutputDim();

	//actvLayer conv2_2Actv(conv2_2Dim.height * conv2_2Dim.width * conv2_2Dim.channel, Actv::RELU);

	// 第三层fc1    ----	32 * 7 * 7 -> 100    全连接层
	fcLayer fc1(conv2_1Dim.height * conv2_1Dim.width * conv2_1Dim.channel, 100);

	actvLayer fc1Actv(100, Actv::RELU);

	//// 第四层fc2	   ---- 512 -> 64
	//fcLayer fc2(512, 64);

	//actvLayer fc2Actv(64, Actv::RELU);

	// 第五层out层  ---- 100 -> 10	Sigmoid全连接输出层
	fcLayer outLayer(100, 10);

	actvLayer outActv(10, Actv::SIGMOID);
	//softmaxLayer outSoftmax(10);

	vector<layer*> Net = { 
		&conv1_1, &conv1_1Actv, 
		//&conv1_2, &conv1_2Actv, 
		&conv2_1, &conv2_1Actv, 
		//&conv2_2, &conv2_2Actv,
		&fc1, &fc1Actv,
		//&fc2, &fc2Actv,
		&outLayer, &outActv
	};
	Graph model(batch_size, iterations, Net);
	model.setLearningRate(0.001f, 0.2f, 0.8f);
	model.Run();
	model.Validation();
	while (1)
	{
		model.Test();
	}
}


#endif // !
