#pragma once
#include <opencv2/opencv.hpp>
//#include <sstream>

typedef struct ImgSource
{
	std::string _basepath = "";		//图片基本路径
	std::string _imagename = "";	//图片名（不含编号）
	std::string _suffix = "";		//图片后缀
}ImageSource;

class ReadImages
{
public:
	ReadImages() {}
	ReadImages(const std::string basepath, const std::string imagename, const std::string suffix);

	cv::Mat loadImage(int imgId);	//读入图片
private:
	ImageSource _imgSource;			//图片路径信息
	cv::Rect _roi;
};
