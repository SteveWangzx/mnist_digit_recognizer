#ifndef _actv_function_hpp__
#define _actv_function_hpp__


/***********************************
* 激活函数层                         *
* 目前包括 ReLU Sigmoid两种激活模式   * 
************************************/

class actvLayer: public layer
{
public:
	actvLayer() {};
	actvLayer(int size, Actv mode)
	{
		this->size = size;
		this->mode = mode;

		x.resize(this->size);
		y.resize(this->size);
		dy.resize(this->size);
		dx.resize(this->size);
	}

	void Forward() override
	{
		switch (mode)
		{
		case Actv::RELU: 
		{
			for (int i = 0; i < size; ++i)
			{
				y[i] = relu(x[i]);
			}
		}
			break;
		case Actv::SIGMOID:
		{
			for (int i = 0; i < size; ++i)
			{
				y[i] = sigmoid(x[i]);
			}
		}
			break;
		default:
			break;
		}
	}

	void Backward() override
	{
		switch (mode)
		{
		case Actv::RELU: 
		{
			for (int i = 0; i < size; ++i)
			{
				dx[i] = relu_gd(y[i]) * dy[i];
			}
		}
			break;
		case Actv::SIGMOID:
		{
			for (int i = 0; i < size; ++i)
			{
				dx[i] = sigmoid_gd(y[i]) * dy[i];
			}
		}
			break;
		default:
			break;
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
	float relu_gd(float y)
	{
		if (y > 0) { return 1.0f; }
		else { return 0.0f; }
	}

private:
	int size;
	Actv mode;
};

#endif // !_actv_function_hpp__
