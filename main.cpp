/*
* 文件名称：main.cpp
* 依赖库：opencv3.1.0或其他版本
* 测试环境：VS2015 x64
* 摘 要：	calcOpticalFlowFeaturesPyrLKSAD函数的测试函数，测试数据集为ETH数据集，总共包含107帧图像
* 包含文件：calcOpticalFlowFeaturesPyrLKSAD.cpp/.h，主要测试文件；ReadImages.cpp/.h读入照片类
* 部分测试结果：pyrLKSize这个参数比较影响用时：取pyrLKSize=5，107帧用时约3.4秒；取15用时约6.5秒、
*				计算光流数目numOF也比较影响时间，一般取numOF=18或者稍多一些（试过取15，耗时少了近0.4秒）
* 使用实例：见示例程序main.cpp
* 2016.05.24
*/

#pragma once
#include "calcOpticalFlowFeaturesPyrLKSAD.h"
#include "ReadImages.h"
#include <time.h>

int main()
{
	const int numOF = 18;				//计算光流数目
	const int pyrLKSize = 5;			//金字塔LK搜索区域
	const int sadBlockSize = 4;			//SAD匹配块大小（真实大小为2*sadBlockSize+1）
	const int blockSize = 5;			//非极大值抑制窗口真实大小
	const int maxPixShift = 30;			//设定的最大像素位移（相邻帧间）

	std::vector<cv::Point2f> prevXY;	//存储计算光流点在上一帧中的坐标
	std::vector<cv::Point2f> currXY;	//存储计算光流点在下一帧中的坐标

	float ofXY[numOF][2];				//存储计算出来的光流

	//图片路径：ETH数据集图片，总共107帧
	ReadImages imageReader("G:\\ETH ASL\\ETH-IMAGE-MODIFIED\\Img1", "", ".png");

	cv::Mat prev;
	cv::Mat curr;

	//先计算第一二帧
	prev = imageReader.loadImage(1);
	curr = imageReader.loadImage(2);

	calcOpticalFlowFeaturesPyrLKSAD(
		prev,
		curr,
		prevXY,
		currXY,
		ofXY,
		1,
		numOF,
		pyrLKSize,
		sadBlockSize,
		blockSize,
		maxPixShift);

	//为不污染原图片，复制原图片，在复制图片上绘点，显示结果
	cv::Mat prev_copy, curr_copy;
	prev.copyTo(prev_copy);
	curr.copyTo(curr_copy);

	//显示第一二帧的结果
	for (int iter2 = 0;iter2 < prevXY.size();iter2++)
	{
		std::cout << prevXY[iter2].x << ", " << prevXY[iter2].y << std::endl;
		std::cout << currXY[iter2].x << ", " << currXY[iter2].y << std::endl;
		std::cout << ofXY[iter2][0] << ", " << ofXY[iter2][1] << std::endl << std::endl;
		cv::circle(prev_copy, prevXY[iter2], 3, cv::Scalar(0, 0, 255), -1, 8);
		cv::circle(curr_copy, currXY[iter2], 3, cv::Scalar(0, 0, 255), -1, 8);
	}
	cv::imshow("prevImg", prev_copy);
	cv::imshow("currImg", curr_copy);
	cv::waitKey(1000);

	curr.copyTo(prev);
	int imgID = 3;

	//计时
	double timeConsume;
	clock_t start, finish;
	start = clock();

	//进入循环测试
	while (1)
	{
		curr = imageReader.loadImage(imgID);	//读入当前帧

		if (curr.empty())	break;				//循环终止条件

		prev.copyTo(prev_copy);
		curr.copyTo(curr_copy);
		prevXY.swap(currXY);					//交换：当前帧变为上一帧的坐标
		currXY.clear();							//*****记得清零*****

		calcOpticalFlowFeaturesPyrLKSAD(
			prev,
			curr,
			prevXY,
			currXY,
			ofXY,
			0,
			numOF,
			pyrLKSize,
			sadBlockSize,
			blockSize,
			maxPixShift);

		//显示结果
		for (int iter2 = 0;iter2 < prevXY.size();iter2++)
		{
			//std::cout << prevXY[iter2].x << ", " << prevXY[iter2].y << std::endl;
			//std::cout << currXY[iter2].x << ", " << currXY[iter2].y << std::endl;
			std::cout << ofXY[iter2][0] << ", " << ofXY[iter2][1] << std::endl << std::endl;
			cv::circle(prev_copy, prevXY[iter2], 3, cv::Scalar(0, 0, 255), -1, 8);
			cv::circle(curr_copy, currXY[iter2], 3, cv::Scalar(0, 0, 255), -1, 8);
		}
		cv::imshow("prevImg", prev_copy);
		cv::imshow("currImg", curr_copy);
		cv::waitKey(500);

		curr.copyTo(prev);						//当前帧变为下一帧
		std::cout << std::endl << "****************************已完成第" << imgID << "帧****************************" << std::endl << std::endl;
		imgID++;
	}

	finish = clock();

	//计算并显示用时
	timeConsume = (double)(finish - start) / CLOCKS_PER_SEC;
	std::cout << std::endl << "timeConsume= " << timeConsume << std::endl << std::endl;

	system("pause");
}