#ifndef _gemm_hpp_
#define _gemm_hpp_


void gemm_im2col(
	const float* x,	//输入数据的指针
	const int c, const int h, const int w, //输入数据的通道/高/宽
	const int kh, const int kw, //卷积核尺寸
	const int ph, const int pw, //补边大小，无补边传参数0
	const int sh, const int sw, //步长大小
	const int dh, const int dw, //膨胀大小，无膨胀传参数1
	float* y	// 输出行列式
)
{
	const int oh = (h + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;		// 计算输出大小
	const int ow = (w + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;
	const int icsize = h * w;

	for (int _c = c; _c--; x += icsize)
	{
		for (int _kh = 0; _kh < kh; _kh++)
		{
			for (int _kw = 0; _kw < kw; _kw++)
			{
				int _ih = -ph + _kh * dh;
				for (int _oh = oh; _oh; _oh--)
				{
					if (((unsigned)_ih) < ((unsigned)h))
					{
						int _iw = -pw + _kw * dw;
						for (int _ow = ow; _ow; _ow--)
						{
							if (((unsigned)_iw) < ((unsigned)w)) {
								*(y++) = x[_ih * w + _iw];
							}
							else {
								*(y++) = 0;
							}
							_iw += sw;
						}
					}
					else
					{
						for (int _ow = ow; _ow; _ow--) {
							*(y++) = 0;
						}
					}
					_ih += sh;
				}
			}
		}
	}
}

void gemm_col2im(
	const float* x,		// 输入: col = kernelSize * ySize * channel
	const int c, const int h, const int w, // 输出数据的通道/高/宽
	const int kh, const int kw, //卷积核尺寸
	const int ph, const int pw, //补边大小，无补边传参数0
	const int sh, const int sw, //步长大小
	const int dh, const int dw, //膨胀大小，无膨胀传参数1
	float* y	// 输出: image
)
{
	const int oh = (h + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;		// 计算输出大小
	const int ow = (w + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;
	const int icsize = h * w;
	const int colSize = kh * kw * oh * ow;	// 一个channel的col大小： kernelSize * ySize

	// image = c * h * w
	int col_index = 0;
	for (size_t channel = 0; channel < c; channel++)
	{
		for (size_t row = 0; row < oh; row++)
		{
			for (size_t col = 0; col < ow; col++)
			{
				for (size_t i = 0; i < kh; i++)
				{
					for (size_t j = 0; j < kw; j++)
					{
						int im_row = row * sh + i - ph;
						int im_col = col * sw + j - pw;
						if (im_row >= 0 && im_row < h && im_col >= 0 && im_col < w) {
							y[channel * icsize + im_row * w + im_col] += x[col_index];
						}
						col_index++;
					}
				}
			}
		}
	}
}

void gemm_nn(int M, int N, int K, float ALPHA,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc)
{
	//#pragma omp parallel for
	for (int i = 0; i < M; ++i)
	{
		for (int k = 0; k < K; ++k)
		{
			float A_PART = ALPHA * A[i * lda + k];
			for (int j = 0; j < N; ++j)
			{
				C[i * ldc + j] += A_PART * B[k * ldb + j];
			}
		}
	}
}
void gemm_tn(int M, int N, int K, float ALPHA,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			float A_PART = ALPHA * A[k * lda + i];
			for (j = 0; j < N; ++j) {
				C[i * ldc + j] += A_PART * B[k * ldb + j];
			}
		}
	}
}
void gemm_nt(int M, int N, int K, float ALPHA,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc)
{
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			float sum = 0;
			for (int k = 0; k < K; ++k) {
				sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
			}
			C[i * ldc + j] += sum;
		}
	}
}
void gemm_tt(int M, int N, int K, float ALPHA,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			float sum = 0;
			for (k = 0; k < K; ++k) {
				sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
			}
			C[i * ldc + j] += sum;
		}
	}
}





//C = (A * B) * alpha + beta * C
void gemm(
	int TA, //是否转置A矩阵
	int TB, //是否转置B矩阵
	int M, int N, int K, // A.shape = M * K    B.shape = K * N     
	float ALPHA, // 缩放系数
	float* A, int lda, //A矩阵指针和A的步长，（步长一般情况下等与矩阵的宽）
	float* B, int ldb,  //B矩阵指针和B的步长
	float BETA, //缩放系数
	float* C, int ldc //C矩阵指针和B的步长
)
{
	if (BETA == 0.0f)
	{
		for (int i = 0; i < M; ++i)
		{
			for (int j = 0; j < N; ++j) {
				C[i * ldc + j] = 0.0f;
			}
		}
	}
	else if (BETA != 1.0f)
	{
		for (int i = 0; i < M; ++i)
		{
			for (int j = 0; j < N; ++j) {
				C[i * ldc + j] *= BETA;
			}
		}
	}

	if (!TA && !TB) gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else if (TA && !TB) gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else if (!TA && TB) gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}


#endif // !_gemm_hpp_
