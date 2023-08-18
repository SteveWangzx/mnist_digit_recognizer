#ifndef _actv_function_hpp__
#define _actv_function_hpp__

class actvLayer
{
public:
	actvLayer() {};
	actvLayer(int size, Actv mode)
	{
		this->size = size;
		this->mode = mode;

		neurals.resize(this->size);
		output.resize(this->size);
		dactv.resize(this->size);
	}

	void setInput(vector<float> input)
	{
		this->neurals = input;
	}

	void setDactv(vector<float> loss)
	{
		this->dactv = loss;
	}

	vector<float> getOutput()
	{
		return output;
	}

	vector<float> getDactv()
	{
		return dactv;
	}

	void forward_compute()
	{
		switch (mode)
		{
		case Actv::RELU: 
		{
			for (int i = 0; i < size; ++i)
			{
				output[i] = relu(neurals[i]);
			}
		}
			break;
		case Actv::SIGMOID:
		{
			for (int i = 0; i < size; ++i)
			{
				output[i] = sigmoid(neurals[i]);
			}
		}
			break;
		case Actv::SOFTMAX:
			break;
		default:
			break;
		}
	}

	void backward_compute()
	{
		switch (mode)
		{
		case Actv::RELU: 
		{
			for (int i = 0; i < size; ++i)
			{
				dactv[i] = relu_gd(output[i]) * dactv[i];
			}
		}
			break;
		case Actv::SIGMOID:
		{
			for (int i = 0; i < size; ++i)
			{
				dactv[i] = sigmoid_gd(output[i]) * dactv[i];
			}
		}
			break;
		case Actv::SOFTMAX:
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
	float relu_gd(float y /*严格来说，应该输入x*/)
	{
		if (y > 0) { return 1.0f; }
		else { return 0.0f; }
	}
private:
	vector<float> neurals;
	vector<float> output;
	vector<float> dactv;
	int size;
	Actv mode;
};

#endif // !_actv_function_hpp__
