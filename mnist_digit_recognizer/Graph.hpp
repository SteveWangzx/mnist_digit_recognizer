#ifndef _GRAPH_hpp_
#define _GRAPH_hpp_

#include "layer.hpp"

// random engine
std::random_device graphRandomDevice;
std::default_random_engine engine(graphRandomDevice());

class Graph
{
public:
	Graph() = default;
	Graph(int bSize, int itr): BATCH_SIZE(bSize), ITERATIONS(itr)
	{
		// code for building dataset
		samples.addPath("./mnist_digit_train.csv");
		samples.collectData();
		samples.split_train_test(0.2);
		input.resize(INPUT_SIZE);
		output.resize(10);
		loss.resize(10);
	};
	Graph(int bSize, int itr, vector<layer*> layers) : BATCH_SIZE(bSize), ITERATIONS(itr), graph(layers) 
	{
		// code for building dataset
		samples.addPath("./mnist_digit_train.csv");
		samples.collectData();
		samples.split_train_test(0.2);
		input.resize(INPUT_SIZE);
		output.resize(10);
		loss.resize(10);
		graphSize = graph.size();
	}
	
	void insertLayer(vector<layer*> layers)
	{
		graph = layers;
		graphSize = graph.size();
	}

	void setLearningRate(float lr, float increasePoint, float decreasePoint)
	{
		maxLR = lr;
		increaseLR = increasePoint;
		decreaseLR = decreasePoint;
		increasePart = (increasePoint == 0.0f) ? 0.0f : maxLR / (ITERATIONS * increasePoint);
		decreasePart = (decreasePoint == 1.0f) ? 0.0f : maxLR / (ITERATIONS * (1 - decreasePoint));
	}

	void onSetInput()
	{
		// 随机抽取训练样本
		std::uniform_int_distribution<int> uni(0, samples.trainSize - 1);
		int sampleID = uni(engine);
		currSample = samples.getImage(sampleID);			
		for (int i = 0; i < INPUT_SIZE; ++i)
		{
			input[i] = currSample.x[i];
		}
	}

	void onComputeLoss()
	{
		float currMax = 0.0f;
		for (int i = 0; i < 10; ++i)
		{
			// onehot
			if (i == currSample.y)
			{
				loss[i] = 1.0f - output[i];
				losslist.push_back(fabs(loss[i]));
				if (losslist.size() > 1000) { losslist.pop_front(); }
			}
			else
			{
				loss[i] = (0.0f - output[i]) * ecof;
			}

			if (output[i] > currMax)
			{
				predict = i;
				currMax = output[i];
			}
		}

		if (predict == currSample.y)
		{
			correctList.push_back(1);
		}
		else
		{
			correctList.push_back(0);
		}

		if (correctList.size() > 1000) { correctList.pop_front(); }
	}

	/* 训练过程 */
	void Run()
	{
		for (int itr = 0; itr < ITERATIONS; ++itr)
		{
			currItr = itr + 1;
			updateLR();				// 更新学习率
			ecof += 0.0002f;
			if (ecof >= 1.0f) { ecof = 1.0f; }
			
			for (int _b = 0; _b < BATCH_SIZE; ++_b)
			{
				onSetInput();			// 输入
				Forward();				// 前向传播
				onComputeLoss();		// 计算Loss
				Backward();				// 反向传播
			}
			Update();				// 每个batch之后更新参数
			// 打印训练信息
			{
				float tloss = 0;
				for (float l : losslist) { tloss += l; }
				int hit = 0;
				for (int corr : correctList)
				{
					if (corr == 1)
					{
						++hit;
					}
				}
				tloss /= losslist.size();
				float curr_correct = (float)hit / correctList.size();
				printf("Training...<%d|%d> LR: %f ecof: %f Loss: %f Correctness: %f \r", 
					currItr, ITERATIONS, LearningRate, ecof, tloss, curr_correct);
			}
		}
	}

	void Validation()
	{
		// 遍历测试集
		// 训练集：0 - (trainSize-1)     测试集：trainSize - sampleSize
		printf("\n");
		int total = 0;
		int correct = 0;
		for (int i = samples.trainSize; i < samples.sampleSize; i++)
		{
			++total;
			image currVal = samples.getImage(i);
			for (int j = 0; j < INPUT_SIZE; ++j)
			{
				input[j] = currVal.x[j];
			}
			this->Forward();
			float currMax = 0.0f;
			for (int j = 0; j < 10; ++j)
			{
				if (output[j] > currMax)
				{
					predict = j;
					currMax = output[j];
				}
			}

			if(predict == currVal.y)
			{
				++correct;
			}
			{
				float accuracy = (float)correct / total;
				printf("Run Validation Set: <%d|%d>  Accuracy: %f \r", total, samples.testSize, accuracy);
			}
		}
	}

	void Test()
	{

	}


private:
	void Forward()
	{
		// 第一层传入input
		graph[0]->setX(input);
		graph[0]->Forward();

		// 后续层前向传播
		for (int i = 1; i < graphSize; ++i)
		{
			graph[i]->setX(graph[i - 1]->getY());
			graph[i]->Forward();
		}

		// 得到输出
		output = graph[graphSize - 1]->getY();
	}

	void Backward()
	{
		graph[graphSize - 1]->setDY(loss);
		graph[graphSize - 1]->Backward();
		for (int i = graphSize - 2; i >= 0; --i)
		{
			graph[i]->setDY(graph[i + 1]->getDX());
			graph[i]->Backward();
		}
	}

	void Update()
	{
		for (layer* node : graph)
		{
			node->Update(LearningRate);
		}
	}

	void avoidNaN(vector<float> &data)
	{
		for (int i = 0; i < data.size(); ++i)
		{
			data[i] = std::isnan(data[i]) ? 0.0f : data[i];
			data[i] = std::isinf(data[i]) ? 0.0f : data[i];
		}
	}

	/* 动态学习率 */
	void updateLR()
	{
		float currPoint = currItr / (float)ITERATIONS;

		if (currPoint < increaseLR)
		{
			LearningRate += increasePart;
		}
		else if (currPoint >= decreaseLR)
		{
			LearningRate -= decreasePart;
		}
		else
		{
			LearningRate = maxLR;
		}
	}

private:
	int BATCH_SIZE = 0;
	int ITERATIONS = 0;
	int currItr = 0;

	/* 学习率 */
	float LearningRate = 0.0f;
	float maxLR = 1.0f;
	/* 动态学习率上升结束节点，下降开始节点（迭代次数百分比） */
	float increaseLR = 0.0f;	
	float decreaseLR = 0.0f;
	// 每次迭代之后学习率变化量
	float increasePart = 0.0f;
	float decreasePart = 0.0f;

	/* 衰减变量 */
	float ecof = 0.0;

	/* 训练传递变量 */
	image currSample;
	vector<float> input;
	vector<float> output;
	vector<float> loss;

	/* 打印信息 */
	int predict;
	list<float> losslist;
	list<float> correctList;

private:
	vector<layer*> graph;
	int graphSize = 0;
	dataCollector samples;

};

#endif // !_GRAPH_hpp_
