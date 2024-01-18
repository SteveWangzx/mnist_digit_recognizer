#ifndef _define_hpp__
#define _define_hpp__

/* ���ñ���ͷ�ļ� */

// ���ݼ�size
const int SIZE_TRAIN = 33600;
const int SIZE_TEST = 8400;

// ͼ��ߴ�
const int WIDTH = 28;
const int HEIGHT = 28;
const int INPUT_SIZE = 784;		// width * height

// ѵ������
const int batch_size = 16;
const int iterations = 40000;


enum class Actv
{
	RELU,
	SIGMOID,
	SOFTMAX
};

// ά�Ƚṹ��
struct dim
{
	int width = 0;
	int height = 0;
	int channel = 1;
};

// MNIST image�ṹ��
struct image
{
	float x[INPUT_SIZE]; // ����
	int y;				 // ��ǩ
};

#endif // !_define_hpp__
