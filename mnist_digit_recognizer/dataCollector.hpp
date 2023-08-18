#ifndef _data_collector_hpp__
#define _data_coolector_hpp__

const int WIDTH = 28;
const int HEIGHT = 28;
const int INPUT_SIZE = 784;		// width * height

struct image
{
	float x[INPUT_SIZE];
	int y;
};

class dataCollector
{
public:
	dataCollector() {};
	dataCollector(const char* path)
	{
		this->path = path;
	}
	void collectData();
	void peak(int row);
	image getImage(int row);

private:
	void dataHandler(std::vector<std::string> &words);

private:
	std::string path;
	std::vector<image> samples;
};

// void dataCollector::collectData()
// @brief: 读取csv文件
// @param: 
void dataCollector::collectData()
{
	std::ifstream csvFile(path, std::ios::in);
	std::string line, word;
	std::istringstream sin;
	std::vector<std::string> words;

	if (!csvFile.is_open())
	{
		std::cout << "File Not Found!" << std::endl;
		exit(1);
	}
	std::cout << "File Open Successfully!" << std::endl;

	// 跳过标题
	std::getline(csvFile, line);
	// 读取数据
	int idx = 1;
	while (std::getline(csvFile, line))
	{
		if (line.length() == 0)
		{
			continue;
		}

		sin.clear();
		sin.str(line);
		words.clear();

		while (std::getline(sin, word, ','))
		{
			words.push_back(word);
		}
		// 处理数据
		dataHandler(words);
		printf("Collected %d rows of data\r", idx++);
	}
	printf("\n");
	std::cout << "Collect Data Successfully!" << std::endl;
	csvFile.close();
}

void dataCollector::dataHandler(std::vector<std::string>& words)
{
	image temp;
	int idx = 0;
	
	// 输入y
	temp.y = std::stoi(words[idx]);
	++idx;

	// 输入x
	for (int i = 0; i < INPUT_SIZE; ++i)
	{
		/// 归一化处理 [0, 1]
		float norm_x = std::stoi(words[idx]) / 255.0f;
		temp.x[i] = norm_x;
		++idx;
	}
	samples.push_back(temp);
}

// Select certain row of data to display 
void dataCollector::peak(int row)
{
	std::cout << "第" << row << "项" << "数据展示: " << std::endl;
	std::cout << "label: " << samples[row].y << std::endl;
	for (int i = 0; i < HEIGHT; ++i)
	{
		for (int j = 0; j < WIDTH; ++j)
		{
			std::cout << (samples[row].x[i * HEIGHT + j] > 0) ? 1 : 0;
		}
		std::cout << std::endl;
	}
}


// return certain image
image dataCollector::getImage(int row)
{
	return samples[row];
}

#endif // !_data_collector_hpp__
