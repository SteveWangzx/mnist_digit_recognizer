#ifndef _define_hpp__
#define _define_hpp__

/* 公用变量头文件 */

// 数据集size
const int SIZE_TRAIN = 33600;
const int SIZE_TEST = 8400;

// 图像尺寸
const int WIDTH = 28;
const int HEIGHT = 28;
const int INPUT_SIZE = 784;		// width * height

// 训练参数
const int batch_size = 16;
const int iterations = 40000;


enum class Actv
{
	RELU,
	SIGMOID,
	SOFTMAX
};

// 维度结构体
struct dim
{
	int width = 0;
	int height = 0;
	int channel = 1;
};

// MNIST image结构体
struct image
{
	float x[INPUT_SIZE]; // 数据
	int y;				 // 标签
};

#endif // !_define_hpp__
