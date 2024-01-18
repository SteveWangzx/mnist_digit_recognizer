#ifndef _data_collector_hpp__
#define _data_coolector_hpp__

/*  
* �����ռ���
* ��ȡMNIST csv�ļ�
* ����image����
*/

class dataCollector
{
public:
	dataCollector() {};
	dataCollector(const char* path)
	{
		this->path = path;
	}
	void addPath(const char* path)
	{
		this->path = path;
	}
	void collectData();
	void peak(int row);
	void split_train_test(float ration);
	image getImage(int row);

public:
	int sampleSize = 0;
	int trainSize = 0;
	int testSize = 0;

private:
	void dataHandler(std::vector<std::string> &words);

private:
	std::string path;
	std::vector<image> samples;
};

// @brief: �ռ����ݶ�ȡcsv�ļ�
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

	// ��������
	std::getline(csvFile, line);
	// ��ȡ����
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
		// ��������
		dataHandler(words);
		printf("Collected %d rows of data\r", idx++);
	}
	printf("\n");
	sampleSize = samples.size();
	std::cout << "Collect Data Successfully!" << std::endl;
	csvFile.close();
}

/* ������������ */
void dataCollector::dataHandler(std::vector<std::string>& words)
{
	image temp;
	int idx = 0;
	
	// �����ǩy
	temp.y = std::stoi(words[idx]);
	++idx;

	// ����x
	for (int i = 0; i < INPUT_SIZE; ++i)
	{
		// ��һ������ [0, 1]
		float norm_x = std::stoi(words[idx]) / 255.0f;
		temp.x[i] = norm_x;
		++idx;
	}
	samples.push_back(temp);
}

// Select certain row of data to display 
/* ��ӡָ��ͼƬ */
void dataCollector::peak(int row)
{
	std::cout << "��" << row << "��" << "ͼƬչʾ: " << std::endl;
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

/* �ָ����ݼ� ratio: ���Լ�ռ�� */
void dataCollector::split_train_test(float ratio)
{
	testSize = sampleSize * ratio;
	trainSize = sampleSize - testSize;
}


/* ����ָ��ͼƬ */
image dataCollector::getImage(int row)
{
	return samples[row];
}

#endif // !_data_collector_hpp__