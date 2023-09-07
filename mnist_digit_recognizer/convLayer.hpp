#ifndef _conv_layer__
#define _conv_layer__

#include "gemm.hpp"

struct dim
{
	int width = 0;
	int height = 0;
	int channel = 1;
};

class convLayer
{
public:
	convLayer()  = default;
	convLayer(int inputWidth, int inputHeight, int inputChannel,int filterWidth, int filterHeight, int stride, int padding)
	{
		this->inputWidth = inputWidth;
		this->inputHeight = inputHeight;
		this->inputChannel = inputChannel;
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
		//dy.resize(this->outputHeight * this->outputWidth);

		// im2col 内存分配
		//////////////////////////////////////////////////////////////
		this->kernelSize = this->filterHeight * this->filterWidth;
		this->ySize = this->outputHeight * this->outputWidth;
		// Col
		col.resize(kernelSize * ySize * inputChannel);
		dcol.resize(kernelSize * ySize * inputChannel);
		// 权重
		filters.resize(this->filterNum * kernelSize);
		dfilters.resize(this->filterNum * kernelSize);
		// 偏置
		biases.resize(filterNum);
		dbiases.resize(filterNum);
		// 输出矩阵 - 包括多个feature map
		outputMat.resize(filterNum * ySize);
		dy.resize(this->outputHeight * this->outputWidth);
		loss.resize(filterNum * ySize);

		// 初始化
		std::random_device rd_conv;
		std::default_random_engine eng_conv(rd_conv());
		std::uniform_real_distribution<float> w_conv(-0.5f, 0.5f);
		std::uniform_real_distribution<float> b_conv(-0.1f, 0.1f);
		
		// im2col初始化
		for (int i = 0; i < filterNum; ++i)
		{
			for (int j = 0; j < kernelSize; ++j)
			{
				filters[i * kernelSize + j] = w_conv(eng_conv);
			}
		}
		for (int i = 0; i < filterNum; i++)
		{
			 //biases[i] = b_conv(eng_conv);
			biases[i] = 0.0f;
			dbiases[i] = 0.0f;
		}

		////////////////////////////////////////
		for (int i = 0; i < filterHeight; ++i)
		{
			for (int j = 0; j < filterWidth; ++j)
			{
				w[i * filterWidth + j] = w_conv(eng_conv);
				dw[i * filterWidth + j] = 0.0f;
			}
		}

		for (int i = 0; i < outputHeight; ++i)
		{
			for (int j = 0; j < outputWidth; ++j) 
			{
				b[i * outputWidth + j] = b_conv(eng_conv);
				db[i * outputWidth + j] = 0.0f;
			}
		}
	}

	dim getOutputDim()
	{
		dim temp;
		temp.width = outputWidth;
		temp.height = outputHeight;
		temp.channel = filterNum;
		return temp;
	}

	// 将输入x根据filter转化为行列式
	void im2col_noChannel()
	{	
		// 根据stride遍历图像定位滑动窗口
		int column = 0;	// 行列式col index
		for (int outRow = 0; outRow < outputHeight; ++outRow)
		{
			for (int outCol = 0; outCol < outputWidth; ++outCol)
			{
				// 计算输入矩阵初始index
				int startRow = outRow * stride;
				int startCol = outCol * stride;

				// 根据filter遍历滑动窗口
				int row = 0;	// 行列式row index
				for (int filterRow = 0; filterRow < filterHeight; filterRow++)
				{
					for (int filterCol = 0; filterCol < filterWidth; filterCol++)
					{
						// img index
						int inputRow = startRow + filterRow;
						int inputCol = startCol + filterCol;
						
						// 定位当前行列式index
						if (inputCol < 0 || inputCol >= inputWidth || inputRow < 0 || inputRow >= inputHeight)
						{
							continue;
						}else
						{ 
							col[row * ySize + column] = x[inputRow * inputHeight + inputCol];
						}
						++row;
					}
				}

				++column;
			}
		}
	}

	// im2col算法： 加入通道数
	void im2col()
	{
		// 输入:x
		// 输出:col
		//////////////////////////////
		gemm_im2col(x.data(), this->inputChannel, this->inputHeight, this->inputWidth,
			this->filterHeight, this->filterWidth, 0, 0, this->stride, this->stride, 1, 1, this->col.data()
		);
	}

	// 矩阵乘法
	// test for multiplication
	// 卷积核权重filters * 行列式col
	// output size = filtersRow * colWidth
	void mat_mul()
	{
		// 遍历输出矩阵
		for (size_t outRow = 0; outRow < filterNum; outRow++)
		{
			for (size_t outCol = 0; outCol < ySize; outCol++)
			{
				outputMat[outRow * ySize + outCol] = 0.0f;

				for (size_t i = 0; i < kernelSize; i++)
				{
					outputMat[outRow * ySize + outCol] += filters[outRow * kernelSize + i]
						* col[i * ySize + outCol];
				}
			}
		}
	}

	void print_out()
	{
		std::cout << "Test mat mul:" << std::endl;
		for (int row = 0; row < filterNum; ++row)
		{
			for (int col = 0; col < ySize; ++col)
			{
				std::cout << this->outputMat[row * ySize + col] << " ";
			}
			std::cout << std::endl;
		}
	}

	// im2col前向传播
	void im2col_forward()
	{
		// 转化
		im2col();

		// 矩阵相乘 w * x
		// 矩阵A: 卷积核权重 filters - A.shape = M * K
		// 矩阵B: im2col矩阵 col - B.shape = K * N 
		// 矩阵C: 输出矩阵 outputMat - C.shape = M * N
		gemm(0, 0, filterNum, ySize, kernelSize, 1, filters.data(), kernelSize, 
			col.data(), ySize, 0, outputMat.data(), ySize);
		
		// 加入偏置
		for (size_t outRow = 0; outRow < filterNum; outRow++)
		{
			for (size_t outCol = 0; outCol < ySize; outCol++)
			{
				outputMat[outRow * ySize + outCol] += biases[outRow];
			}
		}
	}

	// im2col反向传播
	void im2col_backward()
	{
		// 计算权重梯度 (y = w * x + b  w求偏导: dw = x)
		// dw = dy * col(T)
		// dw = filterNum * kernelSize
		// dy = filterNum * ysize
		// col = kernelSize * ySize	---- col^T = ySize * kernelSize;
		gemm(0, 1, filterNum, kernelSize, ySize, 1, loss.data(), ySize, col.data(), ySize, 0, dfilters.data(), kernelSize);

		// 计算偏置
		// db = dy * [1, 1, 1.....]
		// [1, 1, 1, ......] = filterNum * 1
		// 遍历db
		for (size_t i = 0; i < filterNum; i++)
		{
			// 遍历dy
			for (size_t j = 0; j < ySize; j++)
			{
				dbiases[i] += loss[i * ySize + j] * 1.0f;
			}
		}

		// 计算dx
		// dx = w^T * dy
		// w = filterNum * kernelSize
		// dy = filterNum * ySize
		// dx = kernelSize * ySize
		gemm(1, 0, kernelSize, ySize, filterNum, 1, filters.data(), kernelSize, loss.data(), ySize, 0, dcol.data(), ySize);
	}

	void im2col_update()
	{
		// 更新参数
		static const float LR = 0.001f; //学习率
		static const float Momenteum = 0.9f; //动量(不是必须的)

		// 更新权重
		for (int row = 0; row < filterNum; row++)
		{
			for (int col = 0; col < kernelSize; col++)
			{
				filters[row * kernelSize + col] += dfilters[row * kernelSize + col] * LR;
			}
		}

		// 更新偏置
		for (int i = 0; i < filterNum; ++i)
		{
			biases[i] += dbiases[i] * LR;
		}

		// 动量策略
		/////////////////////////////////////
		for (int row = 0; row < filterNum; row++)
		{
			for (int col = 0; col < kernelSize; col++)
			{
				dfilters[row * kernelSize + col] *= Momenteum;
			}
		}

		for (int i = 0; i < filterNum; ++i)
		{
			dbiases[i] *= Momenteum;
		}
	}

	/// <!!!Test for im2col process>
	// do not use these functions in training process
	//////////////////////////////////////////////////////////
	void setFilters(vector<float> filters)
	{
		this->filters = filters;
	}

	void printDfilters()
	{
		std::cout << "dw :" << std::endl;
		for (int row = 0; row < filterNum; ++row)
		{
			for (int col = 0; col < kernelSize; ++col)
			{
				std::cout << dfilters[row * kernelSize + col] << " ";
			}
			std::cout << std::endl;
		}
	}

	void printDbiases()
	{
		std::cout << "db :" << std::endl;
		for (int i = 0; i < filterNum; ++i)
		{
				std::cout << dbiases[i] << " ";
		}
		std::cout << std::endl;
	}

	void printDcol()
	{
		std::cout << "dx :" << std::endl;
		for (int row = 0; row < kernelSize * inputChannel; ++row)
		{
			for (int col = 0; col < filterNum; ++col)
			{
				std::cout << dcol[row * filterNum + col] << " ";
			}
			std::cout << std::endl;
		}
	}
	//////////////////////////////////////////////////
	/// </!!!Test for im2col process>

	// functions for im2col backward
	/////////////////////////////////////////////////////
	void setLoss(vector<float> loss)
	{
		this->loss = loss;
	}

	vector<float> getDCol()
	{
		return this->dcol;
	}
	/////////////////////////////////////////////////////
	void setX(vector<float> input)
	{
		this->x = input;
	}

	vector<float> getY()
	{
		return this->outputMat;
	}

	void print_Col()
	{
		std::cout << "Test im2col:" << std::endl;
		for (int row = 0; row < kernelSize; ++row)
		{
			for (int col = 0; col < ySize; ++col)
			{
				std::cout << this->col[row * ySize + col] << " ";
			}
			std::cout << std::endl;
		}
	}
	// !!! only for test !!!
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
	int inputChannel;
	int filterWidth;
	int filterHeight;
	int outputWidth;
	int outputHeight;
	int stride;
	int padding;

	int ySize;
	int kernelSize;
	int filterNum = 3;				// 默认卷积核个数

	// Forward Parameters
	vector<float> x;				// 输入vector - size = sizeInput
	vector<float> y;				// 输出vector - size = sizeNeurals
	vector<float> w;				// 权重vector - size = sizeInput * sizeNeurals
	vector<float> b;				// 偏置vector - size = sizeNeurals
	vector<float> filters;			// 权重矩阵 - 多个卷积核 size = kernelNum * kernelSize; 
	vector<float> biases;			// 偏置向量 - size = kernelNum 
	vector<float> col;				// im2col矩阵 - （kernelSize * ySize）
	vector<float> outputMat;		// im2col输出矩阵 - (kernelNum * ySize)

	// Backward Parameters ---- 偏导
	vector<float> dx;
	vector<float> dw;
	vector<float> db;
	vector<float> dy;
	vector<float> dfilters;			// size = filters
	vector<float> dcol;
	vector<float> dbiases;
	vector<float> loss;
};

#endif // !_conv_layer__
