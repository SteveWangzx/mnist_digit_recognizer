#ifndef _conv_layer__
#define _conv_layer__

#include "gemm.hpp"
#include "layer.hpp"

extern float lr;

struct dim
{
	int width = 0;
	int height = 0;
	int channel = 1;
};

class convLayer: public layer
{
public:
	convLayer()  = default;
	convLayer(int inputWidth, int inputHeight, int inputChannel,int filterWidth, int filterHeight, int stride, int padding, int filterNum)
	{
		this->inputWidth = inputWidth;
		this->inputHeight = inputHeight;
		this->inputChannel = inputChannel;
		this->filterHeight = filterHeight;
		this->filterWidth = filterWidth;
		this->stride = stride;
		this->padding = padding;
		this->filterNum = filterNum;

		// 计算输出矩阵大小
		// padding - 0
		// outHeight = (input_height - filter_height + 2 * padding ) / stride + 1
		// outWidht = (input_width - filter_width + 2 * padding ) / stride + 1
		this->outputWidth = (this->inputWidth - this->filterWidth + 2 * this->padding) / 
			this->stride + 1;
		this->outputHeight = (this->inputHeight - this->filterHeight + 2 * this->padding) / 
			this->stride + 1;

		// im2col 内存分配
		//////////////////////////////////////////////////////////////
		// 卷积核通道数与输出通道数一致
		this->kernelSize = this->filterHeight * this->filterWidth * this->inputChannel;
		this->ySize = this->outputHeight * this->outputWidth;
		x.resize(this->inputWidth * this->inputHeight * this->inputChannel);		// 输入矩阵 
		// Col
		col.resize(kernelSize * ySize);												// col
		dcol.resize(kernelSize * ySize);											// dcol
		dx.resize(this->inputWidth * this->inputHeight * this->inputChannel);		// dx
		// 权重
		w.resize(this->filterNum * kernelSize);								// w
		dw.resize(this->filterNum * kernelSize);								// dw
		// 偏置
		b.resize(filterNum);													// b
		db.resize(filterNum);													// db
		// 输出矩阵 - 包括多个feature map
		y.resize(filterNum * ySize);										// y
		dy.resize(filterNum * ySize);										// dy

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
				w[i * kernelSize + j] = w_conv(eng_conv);
			}
		}
		for (int i = 0; i < filterNum; i++)
		{
			b[i] = 0.0f;
		}
	}

	dim getOutputDim()
	{
		dim temp;
		temp.width = outputWidth;
		temp.height = outputHeight;
		temp.channel = filterNum;	// outChannel = 卷积核数量
		return temp;
	}

	// im2col算法： 加入通道数
	void im2col()
	{
		// 输入:x
		// 输出:col
		//////////////////////////////
		gemm_im2col(x.data(), this->inputChannel, this->inputHeight, this->inputWidth,
			this->filterHeight, this->filterWidth, this->padding, this->padding, this->stride, this->stride, 1, 1, this->col.data()
		);
	}

	void print_out()
	{
		std::cout << "Test mat mul:" << std::endl;
		for (int row = 0; row < filterNum; ++row)
		{
			for (int col = 0; col < ySize; ++col)
			{
				std::cout << this->y[row * ySize + col] << " ";
			}
			std::cout << std::endl;
		}
	}

	// im2col前向传播
	void Forward() override
	{
		// 转化
		im2col();

		// 矩阵相乘 w * x
		// 矩阵A: 卷积核权重 filters - A.shape = M * K
		// 矩阵B: im2col矩阵 col - B.shape = K * N 
		// 矩阵C: 输出矩阵 outputMat - C.shape = M * N
		gemm(0, 0, filterNum, ySize, kernelSize, 1, w.data(), kernelSize, 
			col.data(), ySize, 0, y.data(), ySize);
		
		// 加入偏置
		for (size_t outRow = 0; outRow < filterNum; outRow++)
		{
			for (size_t outCol = 0; outCol < ySize; outCol++)
			{
				y[outRow * ySize + outCol] += b[outRow];
			}
		}
	}

	// im2col反向传播
	void Backward() override
	{
		// loss 转col
		

		// 计算权重梯度 (y = w * x + b  w求偏导: dw = x)
		// dw = dy * col(T)
		// dw = filterNum * kernelSize
		// dy = filterNum * ysize
		// col = kernelSize * ySize	---- col^T = ySize * kernelSize;
		gemm(0, 1, filterNum, kernelSize, ySize, 1, dy.data(), ySize, col.data(), ySize, 0, dw.data(), kernelSize);

		// 计算偏置
		// db = dy * [1, 1, 1.....]
		// [1, 1, 1, ......] = filterNum * 1
		// 遍历db
		for (size_t i = 0; i < filterNum; i++)
		{
			// 遍历dy
			for (size_t j = 0; j < ySize; j++)
			{
				db[i] += dy[i * ySize + j] * 1.0f;
			}
		}
		check_number(db.data(), db.size());
		// 计算dx
		// dx = w^T * dy
		// w = filterNum * kernelSize
		// dy = filterNum * ySize
		// dx = kernelSize * ySize
		gemm(1, 0, kernelSize, ySize, filterNum, 1, w.data(), kernelSize, dy.data(), ySize, 0, dcol.data(), ySize);
		check_number(dcol.data(), dcol.size());
		// col2im
		// dcol转为im形式
		// dcol = kernelSize * ySize
		// im = oH * ow * channel
		for (float& f : dx)
		{
			f = 0;
		}
		gemm_col2im(dcol.data(), this->inputChannel, this->inputHeight, this->inputWidth, filterHeight, filterWidth,
			this->padding, this->padding, this->stride, this->stride, 1, 1, dx.data()
		);
		check_number(dx.data(), dx.size());
	}

	void check_number(float* ptr, size_t length)
	{
		for (size_t i = 0; i < length; i++)
		{
			if (isnan(ptr[i]) || isinf(ptr[i]))
			{
				printf("error");
			}
		}
	}

	void Update(float staticLR) override
	{
		// 更新参数
		//static const float LR = 0.001f; //学习率
		static const float Momenteum = 0.9f; //动量(不是必须的)

		// 更新权重
		for (int row = 0; row < filterNum; row++)
		{
			for (int col = 0; col < kernelSize; col++)
			{
				w[row * kernelSize + col] += dw[row * kernelSize + col] * staticLR;
			}
		}

		// 更新偏置
		for (int i = 0; i < filterNum; ++i)
		{
			b[i] += db[i] * staticLR;
		}

		// 动量策略
		/////////////////////////////////////
		for (int row = 0; row < filterNum; row++)
		{
			for (int col = 0; col < kernelSize; col++)
			{
				dw[row * kernelSize + col] *= Momenteum;
			}
		}

		for (int i = 0; i < filterNum; ++i)
		{
			db[i] *= Momenteum;
		}
	}

	/// <!!!Test for im2col process>
	// do not use these functions in training process
	//////////////////////////////////////////////////////////
	void setFilters(vector<float> filters)
	{
		this->w = filters;
	}

	void printDfilters()
	{
		std::cout << "dw :" << std::endl;
		for (int row = 0; row < filterNum; ++row)
		{
			for (int col = 0; col < kernelSize; ++col)
			{
				std::cout << dw[row * kernelSize + col] << " ";
			}
			std::cout << std::endl;
		}
	}

	void printDbiases()
	{
		std::cout << "db :" << std::endl;
		for (int i = 0; i < filterNum; ++i)
		{
				std::cout << db[i] << " ";
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

	void printDimg()
	{
		int imgSize = inputHeight * inputHeight;
		std::cout << "dx after col2im :" << std::endl;
		for (size_t i = 0; i < inputChannel; i++)
		{
			for (int row = 0; row < inputHeight; ++row)
			{
				for (int col = 0; col < inputWidth; ++col)
				{
					std::cout << dx[i * imgSize + row * inputHeight  + col] << " ";
				}
				std::cout << std::endl;
			}
		}
	}
	//////////////////////////////////////////////////
	/// </!!!Test for im2col process>

	// functions for im2col backward
	/////////////////////////////////////////////////////
	vector<float> getDCol()
	{
		return this->dcol;
	}
	/////////////////////////////////////////////////////

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
	int filterNum;				// 默认卷积核个数

	// Forward Parameters
	vector<float> col;				// im2col矩阵 - （kernelSize * ySize）

	// Backward Parameters ---- 偏导
	vector<float> dcol;
};

#endif // !_conv_layer__
