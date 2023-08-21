#ifndef _conv_layer__
#define _conv_layer__

class convLayer
{
public:
	convLayer() {};

private:
	int sizeInput;					// 输入size
	int sizeNeurals;				// 输出size(神经元个数)
	Actv actvType;					// 激活类型

	// Foward Parameters
	vector<float> x;				// 输入vector - size = sizeInput
	vector<float> y;				// 输出vector - size = sizeNeurals
	vector<float> w;				// 权重vector - size = sizeInput * sizeNeurals
	vector<float> b;				// 偏置vector - size = sizeNeurals

	// Backward Parameters ---- 偏导
	vector<float> dx;
	vector<float> dw;
	vector<float> db;
	vector<float> dy;
};

#endif // !_conv_layer__
