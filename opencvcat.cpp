// opencvcat.cpp : 定义控制台应用程序的入口点。
//


#include <thread>

#include "stdafx.h"
#include "opencv2/opencv.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <time.h>
#include<windows.h>

#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
using namespace cv;
using namespace std;
using namespace dnn;

int catfound = 0;
int ssd();
int test();
int voice1();
/* Find best class for the blob (i. e. class with maximal probability) */
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
	Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
	Point classNumber;
	minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}
static std::vector<String> readClassNames(const char *filename)
{
	std::vector<String> classNames;
	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name.substr(name.find(' ') + 1));
	}
	fp.close();
	return classNames;
}


int test()
{
	VideoCapture capture("http://admin:admin@192.168.1.100:8081");//第2个USB摄像头
	Mat frame_orig;//采集的原始图像
	Mat frame;//处理过的用于测距的图像
	int my_col = 640;//缩减后的图像宽度 单位;像素
	int my_row = 480;//缩减后的图像高度 单位;像素
	int W, H;//图像宽度和高度 单位:像素
	H = my_row;
	W = my_col;

	while (1)
	{
		capture >> frame_orig;//采集
		//resize(frame_orig, frame, Size(my_col, my_row));//缩减像素
		imshow("MyTest", frame_orig);
		//Sleep(10);//为了稳定做的延时
		waitKey(1);
	}
    return 0;
}

int ssd()
{
	String prototxt = "MobileNetSSD_deploy.prototxt";
	String caffemodel = "MobileNetSSD_deploy.caffemodel";
	//String prototxt = "deploy.prototxt";
	//String caffemodel = "mobilenetssd.caffemodel";
	Net net = readNetFromCaffe(prototxt, caffemodel);

	const char* classNames[] = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
		"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

	float detect_thresh = 0.10;
	if (true)
	{
		net.setPreferableTarget(0);
	}
	//VideoCapture capture("http://admin:admin@192.168.1.100:8081");
	VideoCapture capture(0);

	Mat frame;
	while (true)
	{
		capture >> frame;//采集
		clock_t start_t = clock();
		net.setInput(blobFromImage(frame, 1.0 / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false));
		Mat cvOut = net.forward();
		cout << "Cost time: " << clock() - start_t << endl;

		Mat detectionMat(cvOut.size[2], cvOut.size[3], CV_32F, cvOut.ptr<float>());
		for (int i = 0; i < detectionMat.rows; i++)
		{
			int obj_class = detectionMat.at<float>(i, 1);
			float confidence = detectionMat.at<float>(i, 2);

			if (confidence > detect_thresh)
			{
				size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

				int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

				ostringstream ss;
				int tmpI = 100 * confidence;
				ss << tmpI;
				//ss << confidence;
				String conf(ss.str());

				Rect object((int)xLeftBottom, (int)yLeftBottom,
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));
				if (classNames[objectClass] == "cat"|| classNames[objectClass] == "dog")
				{
					rectangle(frame, object, Scalar(0, 0, 255), 2);
					//String label = String(classNames[objectClass]) + ": " + conf;
					//String label = String(classNames[objectClass]);
					String label = String(classNames[objectClass]) + ": " + conf + "%";
					//putText(image, label, Point(xLeftBottom, yLeftBottom - 10), 3, 1.0, Scalar(0, 0, 255), 2);
					putText(frame, label, Point(xLeftBottom, yLeftBottom + 30), 3, 1.0, Scalar(0, 0, 255), 2);
					catfound=catfound+2;
				}

			}
		}
		if (catfound >=1)
			catfound = catfound - 1;
		cout << "cat found " << catfound << endl;
		Mat frame2;
		resize(frame, frame2, Size(960, 540));
		imshow("test", frame2);
		if (cv::waitKey(500) > 1) break;
	}

	return 0;
}

int voice1()
{

	while (1)
	{
		if (catfound>=3)
		{
			
			catfound = 0;
			PlaySound(TEXT("1.wav"), NULL, SND_FILENAME | SND_ASYNC);
		}
		waitKey(5000);
	}
	return 0;
}

void thread1()
{
	ssd();
	//test();
}
void thread2()
{
	voice1();
}

int main()
{
	thread th1(thread1);
	thread th2(thread2);
	th1.join();
	th2.join();
	return 0;
}