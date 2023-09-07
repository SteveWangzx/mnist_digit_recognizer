#ifndef _gemm_hpp_
#define _gemm_hpp_


void gemm_im2col(
	const float* x,	//�������ݵ�ָ��
	const int c, const int h, const int w, //�������ݵ�ͨ��/��/��
	const int kh, const int kw, //����˳ߴ�
	const int ph, const int pw, //���ߴ�С���޲��ߴ�����0
	const int sh, const int sw, //������С
	const int dh, const int dw, //���ʹ�С�������ʹ�����1
	float* y	// �������ʽ
)
{
	const int oh = (h + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;		// ���������С
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
	int TA, //�Ƿ�ת��A����
	int TB, //�Ƿ�ת��B����
	int M, int N, int K, // A.shape = M * K    B.shape = K * N     
	float ALPHA, // ����ϵ��
	float* A, int lda, //A����ָ���A�Ĳ�����������һ������µ������Ŀ�
	float* B, int ldb,  //B����ָ���B�Ĳ���
	float BETA, //����ϵ��
	float* C, int ldc //C����ָ���B�Ĳ���
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
