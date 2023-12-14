#ifndef _softmax_layer__
#define _softmax_layer__

#include "layer.hpp"

class softmaxLayer: public layer
{
public:
	softmaxLayer() {};
	softmaxLayer(int size)
	{
		this->size = size;

		// 分配内存
		x.resize(size);
		y.resize(size);
		dy.resize(size);
		dx.resize(size);
		loss.resize(size);
	}

	// softmax前向传播篇
	// softmax(x) = e^x / e^sum
	void Forward() override
	{
		float e_sum = 0.0f;

		for (size_t i = 0; i < size; i++)
		{
			e_sum += expf(x[i]);
		}

		for (size_t i = 0; i < size; i++)
		{
			y[i] = expf(x[i]) / e_sum;
		}
	}

	// softmax cross-entropy loss
	// loss = softmax - label
	void lossFunction(int* label)
	{
		avgLoss = 0.0f;
		for (size_t i = 0; i < size; i++)
		{
			if (*label == 1)
			{
				currLabel = i;
				avgLoss = abs(y[i] - *label);
			}
			loss[i] = y[i] - *(label++);
		}
	}

	// softmax反向传播
	// 对softmax求导
	void Backward() override
	{
		for (size_t i = 0; i < size; i++)
		{
			float sum = 0.0f;
			for (size_t j = 0; j < size; j++)
			{
				if (i == j)
				{
					sum += dy[j] * y[i] * (1 - y[i]);
				}
				else
				{
					sum -= dy[j] * y[i] * y[j];
				}
			}
			dx[i] = sum;
		}
	}

	vector<float> getLoss()
	{
		return this->loss;
	}

	// !!!test
	//////////////////////////////////////////

	void printOut()
	{
		std::cout << "softmax output :" << std::endl;
		for (int i = 0; i < size; ++i)
		{
			std::cout << y[i] << " ";
		}
		std::cout << std::endl;
	}

	void printLoss()
	{
		std::cout << "softmax Loss:" << std::endl;
		for (int i = 0; i < size; ++i)
		{
			std::cout << dy[i] << " ";
		}
		std::cout << std::endl;
	}

	//////////////////////////////////////////

private:
	int currLabel;
	vector<float> loss;

	int size;
	float avgLoss = 0.0f;
};

#endif // !_softmax_layer__
