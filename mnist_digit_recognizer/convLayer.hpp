#ifndef _conv_layer__
#define _conv_layer__

class convLayer
{
public:
	convLayer() {};

private:
	int sizeInput;					// ����size
	int sizeNeurals;				// ���size(��Ԫ����)
	Actv actvType;					// ��������

	// Foward Parameters
	vector<float> x;				// ����vector - size = sizeInput
	vector<float> y;				// ���vector - size = sizeNeurals
	vector<float> w;				// Ȩ��vector - size = sizeInput * sizeNeurals
	vector<float> b;				// ƫ��vector - size = sizeNeurals

	// Backward Parameters ---- ƫ��
	vector<float> dx;
	vector<float> dw;
	vector<float> db;
	vector<float> dy;
};

#endif // !_conv_layer__
