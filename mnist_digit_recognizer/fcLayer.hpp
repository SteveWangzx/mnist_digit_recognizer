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

		// 分配内存
		x.resize(this->sizeInput);
		y.resize(this->sizeNeurals);
		w.resize(this->sizeInput * this->sizeNeurals);
		b.resize(this->sizeNeurals);

		dx.resize(this->sizeInput);
		dy.resize(this->sizeNeurals);
		dw.resize(this->sizeInput * this->sizeNeurals);
		db.resize(this->sizeNeurals);

		// 初始化
		std::random_device rd_neu;
		std::default_random_engine eng_neu(rd_neu());
		std::uniform_real_distribution<float> w_neu(-0.5f, 0.5f);
		std::uniform_real_distribution<float> b_neu(-0.1f, 0.1f);

		for (int n = 0; n < this->sizeNeurals; ++n)
		{
			for (int i = 0; i < this->sizeInput; ++i)
			{
				//权重初始化为随机数[-0.5, 0.5]
				w[n * this->sizeInput + i] = w_neu(eng_neu);
				//权重梯度初始化为0
				dw[n * this->sizeInput + i] = 0.0f;
			}

			//偏置初始化为0
			b[n] = 0.0f;
			//偏置梯度初始化为0
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

	/// @brief 向前传播
	void Forward() override
	{
		//y = w * x + b
		for (int n = 0; n < sizeNeurals; ++n)
		{
			y[n] = 0.0f;
			for (int i = 0; i < sizeInput; ++i)
			{
				y[n] += x[i] * w[n * sizeInput + i]; //输入 * 权重
			}
			y[n] += b[n]; //加偏置
		}

		//激活
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

	/// @brief 向后传播
	void Backward() override
	{
		//反激活
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

		//计算偏置梯度 (y = w * x + b  b求偏导: db = 1.0)
		for (int n = 0; n < sizeNeurals; ++n)
		{
			db[n] += 1.0f * dy[n];
		}

		//计算权重梯度 (y = w * x + b  w求偏导: dw = x)
		for (int n = 0; n < sizeNeurals; ++n)
		{
			for (int i = 0; i < sizeInput; ++i)
			{
				dw[n * sizeInput + i] += x[i] * dy[n];
			}
		}

		//计算输入梯度 (y = w * x + b  x求偏导: dx = w)  用于前一层反向传播
		for (int i = 0; i < sizeInput; ++i)
		{
			dx[i] = 0.0f; //dx无需累积，每次计算前清零
			for (int n = 0; n < sizeNeurals; ++n)
			{
				dx[i] += w[n * sizeInput + i] * dy[n];
			}
		}
	}

	/// @brief 更新参数
	void Update(float staticLR) override
	{
		//static const float LR = 0.001f; //学习率
		static const float Momenteum = 0.9f; //动量(不是必须的)

		//更新权重
		for (int idx = 0; idx < sizeInput * sizeNeurals; ++idx)
		{
			w[idx] += dw[idx] * staticLR;
		}
		//更新偏置
		for (int idx = 0; idx < sizeNeurals; ++idx)
		{
			b[idx] += db[idx] * staticLR;
		}


		//动量策略（如果未设置动量，直接清零即可）
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
	float relu_gd(float y /*严格来说，应该输入x*/)
	{
		if (y > 0) { return 1.0f; }
		else { return 0.0f; }
	}

private:
	int sizeInput;					// 输入size
	int sizeNeurals;				// 输出size(神经元个数)
	Actv actvType;					// 激活类型

	//// Foward Parameters
	//vector<float> x;				// 输入vector - size = sizeInput
	//vector<float> y;				// 输出vector - size = sizeNeurals
	//vector<float> w;				// 权重vector - size = sizeInput * sizeNeurals
	//vector<float> b;				// 偏置vector - size = sizeNeurals

	//// Backward Parameters ---- 偏导
	//vector<float> dx;				
	//vector<float> dw;
	//vector<float> db;
	//vector<float> dy;
};

#endif // !_full_connected_layer__
