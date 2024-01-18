#ifndef _layer_hpp_
#define _layer_hpp_

/* 神经网络层激活层基类 */

class layer
{
public:
	layer() {};

	void setX(vector<float> input)
	{
		this->x = input;
	}

	void setDY(vector<float> loss)
	{
		this->dy = loss;
	}

	vector<float> getY()
	{
		return this->y;
	}

	vector<float> getDX()
	{
		return this->dx;
	}

	// 神经网络前反向传播虚函数，子类具体实现
	virtual void Forward() {};
	virtual void Backward() {};
	virtual void Update(float staticLR) {};

protected:
	/* 前向传播 */
	vector<float> x;	// 输入
	vector<float> y;	// 输出
	vector<float> w;	// 权重 Wieghts
	vector<float> b;	// 偏置 Bias
	
	/* 反向传播偏导 */
	vector<float> dx;	
	vector<float> dy;
	vector<float> dw;
	vector<float> db;
};

#endif // !_layer_hpp_
