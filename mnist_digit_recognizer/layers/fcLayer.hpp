#ifndef _full_connected_layer__
#define _full_connected_layer__

class fcLayer: public layer
{
public:
	fcLayer() {};
	fcLayer(int inputSize, int outputSize)
	{
		this->sizeInput = inputSize;
		this->sizeNeurals = outputSize;

		// �����ڴ�
		x.resize(this->sizeInput);
		y.resize(this->sizeNeurals);
		w.resize(this->sizeInput * this->sizeNeurals);
		b.resize(this->sizeNeurals);

		dx.resize(this->sizeInput);
		dy.resize(this->sizeNeurals);
		dw.resize(this->sizeInput * this->sizeNeurals);
		db.resize(this->sizeNeurals);

		// ��ʼ��
		std::random_device rd_neu;
		std::default_random_engine eng_neu(rd_neu());
		std::uniform_real_distribution<float> w_neu(-0.5f, 0.5f);
		std::uniform_real_distribution<float> b_neu(-0.1f, 0.1f);

		for (int n = 0; n < this->sizeNeurals; ++n)
		{
			for (int i = 0; i < this->sizeInput; ++i)
			{
				//Ȩ�س�ʼ��Ϊ�����[-0.5, 0.5]
				w[n * this->sizeInput + i] = w_neu(eng_neu);
				//Ȩ���ݶȳ�ʼ��Ϊ0
				dw[n * this->sizeInput + i] = 0.0f;
			}

			//ƫ�ó�ʼ��Ϊ0
			b[n] = 0.0f;
			//ƫ���ݶȳ�ʼ��Ϊ0
			db[n] = 0.0f;
		}
	}

	vector<float> GetW()
	{
		return w;
	}

	vector<float> GetBias()
	{
		return b;
	}

	/// @brief ��ǰ����
	void Forward() override
	{
		//y = w * x + b
		for (int n = 0; n < sizeNeurals; ++n)
		{
			y[n] = 0.0f;
			for (int i = 0; i < sizeInput; ++i)
			{
				y[n] += x[i] * w[n * sizeInput + i]; //���� * Ȩ��
			}
			y[n] += b[n]; //��ƫ��
		}

		//����
		//switch (actvType)
		//{
		//case Actv::RELU:
		//{
		//	for (int n = 0; n < sizeNeurals; ++n)
		//	{
		//		y[n] = relu(y[n]);
		//	}
		//}break;
		//case Actv::SIGMOID:
		//{
		//	for (int n = 0; n < sizeNeurals; ++n)
		//	{
		//		y[n] = sigmoid(y[n]);
		//	}
		//}break;
		//}
	}

	/// @brief ��󴫲�
	void Backward() override
	{
		//������
		//switch (actvType)
		//{
		//case Actv::RELU:
		//{
		//	for (int n = 0; n < sizeNeurals; ++n)
		//	{
		//		dy[n] = relu_gd(y[n]) * dy[n];
		//	}
		//}break;
		//case Actv::SIGMOID:
		//{
		//	for (int n = 0; n < sizeNeurals; ++n)
		//	{
		//		dy[n] = sigmoid_gd(y[n]) * dy[n];
		//	}
		//}break;
		//}

		//����ƫ���ݶ� (y = w * x + b  b��ƫ��: db = 1.0)
		for (int n = 0; n < sizeNeurals; ++n)
		{
			db[n] += 1.0f * dy[n];
		}

		//����Ȩ���ݶ� (y = w * x + b  w��ƫ��: dw = x)
		for (int n = 0; n < sizeNeurals; ++n)
		{
			for (int i = 0; i < sizeInput; ++i)
			{
				dw[n * sizeInput + i] += x[i] * dy[n];
			}
		}

		//���������ݶ� (y = w * x + b  x��ƫ��: dx = w)  ����ǰһ�㷴�򴫲�
		for (int i = 0; i < sizeInput; ++i)
		{
			dx[i] = 0.0f; //dx�����ۻ���ÿ�μ���ǰ����
			for (int n = 0; n < sizeNeurals; ++n)
			{
				dx[i] += w[n * sizeInput + i] * dy[n];
			}
		}
	}

	/// @brief ���²���
	void Update(float staticLR) override
	{
		//static const float LR = 0.001f; //ѧϰ��
		static const float Momenteum = 0.9f; //����(���Ǳ����)

		//����Ȩ��
		for (int idx = 0; idx < sizeInput * sizeNeurals; ++idx)
		{
			w[idx] += dw[idx] * staticLR;
		}
		//����ƫ��
		for (int idx = 0; idx < sizeNeurals; ++idx)
		{
			b[idx] += db[idx] * staticLR;
		}


		//�������ԣ����δ���ö�����ֱ�����㼴�ɣ�
		for (int idx = 0; idx < sizeInput * sizeNeurals; ++idx)
		{
			dw[idx]  *= Momenteum;
		}
		for (int idx = 0; idx < sizeNeurals; ++idx)
		{
			db[idx]  *= Momenteum;
		}
	}

private:
	float sigmoid(float x)
	{
		return 1.0f / (1.0f + expf(-x));
	}
	float sigmoid_gd(float y)
	{
		return y * (1.0f - y);
	}
	float relu(float x)
	{
		if (x > 0) { return x; }
		else { return 0.0f; }
	}
	float relu_gd(float y /*�ϸ���˵��Ӧ������x*/)
	{
		if (y > 0) { return 1.0f; }
		else { return 0.0f; }
	}

private:
	int sizeInput;					// ����size
	int sizeNeurals;				// ���size(��Ԫ����)
	Actv actvType;					// ��������

	//// Foward Parameters
	//vector<float> x;				// ����vector - size = sizeInput
	//vector<float> y;				// ���vector - size = sizeNeurals
	//vector<float> w;				// Ȩ��vector - size = sizeInput * sizeNeurals
	//vector<float> b;				// ƫ��vector - size = sizeNeurals

	//// Backward Parameters ---- ƫ��
	//vector<float> dx;				
	//vector<float> dw;
	//vector<float> db;
	//vector<float> dy;
};

#endif // !_full_connected_layer__
