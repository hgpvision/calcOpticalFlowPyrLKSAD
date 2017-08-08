#include "ReadImages.h"
#pragma once

ReadImages::ReadImages(std::string basepath, const std::string imagename, const std::string suffix)
{
	//将路径中的文件夹分隔符统一为'/'（可以不需要这个for循环，会自动统一，该语句使用的命令参考C++ Primer）
	for (auto &c : basepath)
	{
		if (c == '\\')
		{
			c = '/';
		}
	}

	_imgSource._basepath = basepath + "/";	//这里采用'/'，而不是'\\'
	_imgSource._imagename = imagename;		//图片名（不含编号）
	_imgSource._suffix = suffix;				//图像扩展名

}

//读入单张图片
//输入：imgId，一般要读入序列图片，图片都有个编号
cv::Mat ReadImages::loadImage(int imgId)
{	
	//将图片编号转好字符串
	std::stringstream ss;
	std::string imgNum;
	ss << imgId;
	ss >> imgNum;

	//得到图片的完整绝对路径
	std::string path = _imgSource._basepath + _imgSource._imagename + imgNum + _imgSource._suffix;
	
	cv::Mat img = cv::imread(path,0);

	return img;
}