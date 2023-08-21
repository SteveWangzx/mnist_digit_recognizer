#ifndef _conv_layer__
#define _conv_layer__

struct dim
{
	int width = 0;
	int height = 0;
	int tunnel = 0;
};

class convLayer
{
public:
	convLayer() {};
	convLayer(int inputWidth, int inputHeight, int filterWidth, int filterHeight, int stride, int padding)
	{
		this->inputWidth = inputWidth;
		this->inputHeight = inputHeight;
		this->filterHeight = filterHeight;
		this->filterWidth = filterWidth;
		this->stride = stride;
		this->padding = padding;

		// 计算输出矩阵大小
		// padding - 0
		// outHeight = (input_height - filter_height + 2 * padding ) / stride + 1
		// outWidht = (input_width - filter_width + 2 * padding ) / stride + 1
		this->outputWidth = (this->inputWidth - this->filterWidth + 2 * this->padding) / 
			this->stride + 1;
		this->outputHeight = (this->inputHeight - this->filterHeight + 2 * this->padding) / 
			this->stride + 1;

		// 分配内存
		x.resize(this->inputWidth * this->inputHeight);		// 输入矩阵    
		y.resize(this->outputHeight * this->outputWidth);	// 输出矩阵
		w.resize(this->filterWidth * this->filterHeight);	// 权重
		b.resize(this->outputHeight * this->outputWidth);	// 偏置
		
		dx.resize(this->inputWidth * this->inputHeight);
		dw.resize(this->filterWidth * this->filterHeight);
		db.resize(this->outputHeight * this->outputWidth);
		dy.resize(this->outputHeight * this->outputWidth);

		// 初始化
		std::random_device rd_neu;
		std::default_random_engine eng_neu(rd_neu());
		std::uniform_real_distribution<float> w_neu(-0.5f, 0.5f);
		std::uniform_real_distribution<float> b_neu(-0.1f, 0.1f);

		for (int i = 0; i < filterHeight; ++i)
		{
			for (int j = 0; j < filterWidth; ++j)
			{
				w[i * filterHeight + j] = w_neu(eng_neu);
				dw[i * filterHeight + j] = 0.0f;
			}
		}

		for (int i = 0; i < outputHeight; ++i)
		{
			for (int j = 0; j < outputWidth; ++j) 
			{
				b[i * outputHeight + j] = 0.0f;
				db[i * outputHeight + j] = 0.0f;
			}
		}
	}

	dim getOutputDim()
	{
		dim temp;
		temp.width = outputWidth;
		temp.height = outputHeight;
		return temp;
	}

	void setX(vector<float> input)
	{
		this->x = input;
	}

	vector<float> getY()
	{
		return this->y;
	}

	void setDY(vector<float> loss)
	{
		this->dy = loss;
	}

	vector<float> getDX()
	{
		return this->dx;
	}

	void forward()
	{
	}

private:
	int inputWidth;
	int inputHeight;
	int filterWidth;
	int filterHeight;
	int outputWidth;
	int outputHeight;
	int stride;
	int padding;

	// Foward Parameters
	vector<float> x;				// ÊäÈëvector - size = sizeInput
	vector<float> y;				// Êä³övector - size = sizeNeurals
	vector<float> w;				// È¨ÖØvector - size = sizeInput * sizeNeurals
	vector<float> b;				// Æ«ÖÃvector - size = sizeNeurals

	// Backward Parameters ---- Æ«µ¼
	vector<float> dx;
	vector<float> dw;
	vector<float> db;
	vector<float> dy;
};

#endif // !_conv_layer__
