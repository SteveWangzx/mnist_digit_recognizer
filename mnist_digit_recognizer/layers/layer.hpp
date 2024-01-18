#ifndef _layer_hpp_
#define _layer_hpp_

/* ������㼤������ */

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

	// ������ǰ���򴫲��麯�����������ʵ��
	virtual void Forward() {};
	virtual void Backward() {};
	virtual void Update(float staticLR) {};

protected:
	/* ǰ�򴫲� */
	vector<float> x;	// ����
	vector<float> y;	// ���
	vector<float> w;	// Ȩ�� Wieghts
	vector<float> b;	// ƫ�� Bias
	
	/* ���򴫲�ƫ�� */
	vector<float> dx;	
	vector<float> dy;
	vector<float> dw;
	vector<float> db;
};

#endif // !_layer_hpp_
