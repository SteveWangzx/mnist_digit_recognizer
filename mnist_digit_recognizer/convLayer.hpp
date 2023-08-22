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
	convLayer()  = default;
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
				w[i * filterWidth + j] = w_neu(eng_neu);
				dw[i * filterWidth + j] = 0.0f;
			}
		}

		for (int i = 0; i < outputHeight; ++i)
		{
			for (int j = 0; j < outputWidth; ++j) 
			{
				b[i * outputWidth + j] = 0.0f;
				db[i * outputWidth + j] = 0.0f;
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

	// !!!only for test
	//////////////////////////////////
	void setW(vector<float> weights)
	{
		this->w = weights;
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
		// 卷积计算 ---- 遍历输出矩阵
		for (int outRow = 0; outRow < outputHeight; ++outRow)
		{
			for (int outCol = 0; outCol < outputWidth; ++outCol)
			{
				// output cell [outIdx, outJdx]
				y[outRow * outputWidth + outCol] = 0;

				// 计算输入矩阵初始index
				int startRow = outRow * stride;
				int startCol = outCol * stride;

				// 遍历filter
				for (int filterRow = 0; filterRow < filterHeight; ++filterRow)
				{
					for (int filterCol = 0; filterCol < filterWidth; filterCol++)
					{
						// 计算当前输入矩阵index
						size_t row = startRow + filterRow;
						size_t col = startCol + filterCol;

						// w * x
						y[outRow * outputWidth + outCol] += w[filterRow * filterWidth + filterCol]
							* x[row * inputWidth + col];
					}
				}

				// y = w * x + b
				y[outRow * outputWidth + outCol] += b[outRow * outputWidth + outCol];
			}
		}
	}

	void backward()
	{
		//计算偏置梯度 (y = w * x + b  b求偏导: db = 1.0)
		// dy 与 db一一对应
		for (size_t i = 0; i < outputHeight; ++i)
		{
			for (size_t j = 0; j < outputWidth; j++)
			{
				db[i * outputWidth + j] += 1.0f * dy[i * outputWidth + j];
			}
		}

		// 计算权重梯度 (y = w * x + b  w求偏导: dw = x)
		// 先遍历dy矩阵
		// dy[i, j]
		// 每个dy会更新所有的dw, 记为一轮更新
		// 每轮更新的一个dw只对应一个x	----  dw += x * dy
		for (size_t i = 0; i < outputHeight; i++)
		{
			for (size_t j = 0; j < outputWidth; j++)
			{
				// dy[i * width + j]
				// 遍历filter矩阵
				for (size_t filterRow = 0; filterRow < filterHeight; filterRow++)
				{
					for (size_t filterCol = 0; filterCol < filterWidth; filterCol++)
					{
						// dw[row * width + col]
						// 计算对应x位置
						int xRow = j * stride + filterRow;
						int xCol = i * stride + filterCol;

						dw[filterRow * filterWidth + filterCol] += 
							x[xRow * inputWidth + xCol] * dy[i * outputWidth + j];
					}
				}
			}
		}

		// 计算输入梯度 (y = w * x + b  x求偏导: dx = w * loss)  用于前一层反向传播
		// dx.size = x.size
		// 遍历x和dx
		// 找到每个x对应的所有w * x + b
		// 遍历dy，y输出矩阵
		// 判断当前y是否由当前x参与计算，是则累计dx，否则跳过
		for (size_t i = 0; i < inputHeight; i++)
		{
			for (size_t j = 0; j < inputWidth; j++)
			{
				dx[i * inputWidth + j] = 0.0f;

				for (size_t yRow = 0; yRow < outputHeight; yRow++)
				{
					for (size_t yCol = 0; yCol < outputWidth; yCol++)
					{
						// 计算当前输出y的卷积核窗口范围
						int startRow = yRow * stride;
						int endRow = startRow + filterHeight;
						int startCol = yCol * stride;
						int endCol = startCol + filterWidth;

						// 判断x是否在窗口内
						if (i >= startRow && i < endRow && j >= startCol && j < endCol)
						{
							// 根据x反推对应w位置
							// xRow = yRow * stride + wRow
							// xCol = xCol * stride + wCol
							int wRow = i - startRow;
							int wCol = j - startCol;

							dx[i * inputWidth + j] +=
								w[wRow * filterWidth + wCol] * dy[yRow * outputWidth + yCol];
						}
						else
							continue;
					}
				}

			}
		}

	}

	void update()
	{
		static const float LR = 0.001f; //学习率
		static const float Momenteum = 0.9f; //动量(不是必须的)
		
		// 更新w
		for (size_t idx = 0; idx < filterHeight * filterWidth; idx++)
		{
			w[idx] += dw[idx] * LR;
		}
		// 更新b
		for (size_t idx = 0; idx < outputHeight * outputWidth; idx++)
		{
			b[idx] += db[idx] * LR;
		}

		// 动量策略
		for (size_t idx = 0; idx < filterHeight * filterWidth; idx++)
		{
			w[idx] *= Momenteum;
		}
		for (size_t idx = 0; idx < outputHeight * outputWidth; idx++)
		{
			b[idx] *= Momenteum;
		}
	}

	/// only for test
	void printY()
	{
		std::cout << "Test ConvLayer:" << std::endl;
		for (size_t i = 0; i < outputHeight; i++)
		{
			for (size_t j = 0; j < outputWidth; j++)
			{
				std::cout << y[i * outputWidth + j] << " ";
			}
			std::cout << std::endl;
		}
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
