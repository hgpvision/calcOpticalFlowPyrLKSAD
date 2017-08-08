/*
* 文件名称：Matcher.cpp
* 依赖库：opencv3.1.0或其他版本
* 测试环境：VS2015 x64
* 摘 要：	利用限制性特征点检测+金字塔LK精化+最多7次SAD匹配计算光流（SAD：计算块内所有像素对应灰度值差的绝对值的和）
*			限制性：指的是，真正用于光流计算的特征点，是从提取出的众多点中，选择出满足一定的约束条件的点，点与点之间满足一定的距离：maxPixShift+blockSize
*			金字塔LK：指的是，特征点检测完之后，使用金字塔LK算法校正特征点提取的值并精确到亚像素
*			最多7次SAD匹配：每一次在下一帧对应区域提取出至多7个点，然后再7个点中进行SAD匹配，选出最小SAD值作为匹配点
*关键点：限制距离，7次匹配，重新初始化（重新初始化是非常重要的，当点跟丢时，就需要重新初始化，否则将继续跟踪上一帧中的点）
* 使用实例：见示例程序main.cpp
* 注意：	1）本程序计算出来的光流个数是固定的，即ofXY的维数是不变的；本程序是单纯的光流计算函数，不涉及角速率补偿以及光流排序及选择等操作
*			2）本程序对应的选择光流程序应为：在XY两个方向不加绝对值排序，在XY方向上各舍去2个最大和2个最小光流值，不能低于2，因为程序中已设定
*			达到3个点计算出来的光流超过设定的最大像素位移时，重新初始化，这样，将最多舍去8个点（可能有重叠）；当然，如果计算
*			光流数较多的话，可以考虑舍去更多（因为本程序没有存储每个点的SAD值，所以不支持SAD排序选择光流）
*			3）本程序计算光流数，一般为numOF=18左右，如果取18，则舍去8个点后，还有10个有用点用来计算所需的速度等信息，当然，如果不需要这么多有用点，
*			可以少些；当如果图片质量比较好，纹理多但不杂乱，图片尺寸适当大，可以取多点，但耗时也会稍多些。当图片质量不好时，numOF就得少些了，因为
*			可能取不到这么多的点。
* 2016.05.24
*/
#pragma once
#include<opencv2/opencv.hpp>

//@Brief：计算光流，使用特征点检测+金字塔LK算法+SAD匹配计算光流
//@Input：	prev：上一帧图像,CV_8U1
//			curr：当前帧图像,CV_8U1
//			prevXY：Point2f的vector容器，函数外声明，为计算光流点在上一帧图像中的x,y坐标，含有点的个数为numOF
//			currXY：Point2f的vector容器，函数外声明，为计算光流点在当前帧图像中的x,y坐标，含有点的个数为numOF
//			ofXY[][2]：二维静态数组，函数外声明，计算出来的光流值（含x,y）（相对位移），行数为numOF
//			needInit：是否需要初始化，在跟踪过程中，之前搜索出来的点可能跟丢，那就需要在初始化重新找出适合跟踪的最优角点（当然第一次初始化是必须的）
//			numOF：需要计算的光流矢量数
//			pyrLKSize：金字塔LK计算区，实际大小为2*pyrLKSize+1
//			sadBlockSize：SAD匹配中SAD块的大小，实际大小为2*sadBlockSize+1
//			blockSize：该参数被goodFeaturesToTrack()函数使用，这个参数应该就是进行非最大值抑制的窗口（真实大小）（类似于Harris中的非最大值抑制窗口），越大，提取到的角点越少（该参数很大程度上影响提取到的角点数）
//			maxPixShift：设定目标在相邻帧之间的最大像素位移
//@Output：	无直接输出，直接改变prevXY，currXY，ofXY,以及needInit
//@Innotation：	注意，本程序中所有调用goodFeaturesToTrack()的地方，除了maxCorner这个参数可以不同之外，其他参数必须要保持相同
void calcOpticalFlowFeaturesPyrLKSAD(
	cv::Mat prev,
	cv::Mat curr,
	std::vector<cv::Point2f>& prevXY,
	std::vector<cv::Point2f>& currXY,
	float ofXY[][2],
	bool needInit,
	int numOF = 9, 
	int pyrLKSize=5,
	int sadBlockSize = 4,
	int blockSize = 5,
	int maxPixShift=20);


//@Brief：初始化最优角点，通过对整个图像执行goodFeaturesToTrack()函数提取出所需计算光流的点的坐标
//@Input：	prev：上一帧图像，CV_8U1,每次初始化（或者重新初始化），都是在上一帧图像中重新找最优角点
//			prevXY：最终确定计算光流点的坐标（实际是输出）
//			numOF：需要计算的光流矢量数
//			numPoints：检测点的个数，然后将从这些点中，跳出numOF个，计算光流并跟踪
//			maxPixShift：假定目标在相邻帧之间的最大像素位移，以此确定计算光流点之间的像素距离
//			blockSize：该参数被goodFeaturesToTrack()函数使用
//@Output：无直接输出，直接改变prevXY[][2]的值
//@Innotation：	1）之所以不直接取numPoints=numOF，是因为本算法计算光流点之间满足一定的距离约束，需要从图片中提取出numPonints个点，然后按照响应值
//				从大到小排序，取满足距离约束的前numOF个点作为计算光流点
void initGoodFeatures(
	cv::Mat prev,
	std::vector<cv::Point2f>& prevXY,
	int numOF,
	int numPoints,
	int maxPixShift,
	int blockSize);


//@Brief：初始化之后，调用该函数计算光流
//@Input：	curr：当前帧,CV_8U1
//			prevXY：点在上一帧图像中的坐标
//			currXY：点在当前帧图像中的坐标
//			ofXY：计算出的光流
//			numOF：需要计算的光流数
//			pyrLKSize：金字塔LK计算区，实际大小为2*pyrLKSize+1
//			sadBlockSize：SAD匹配中SAD块的大小，实际大小为2*sadBlockSize+1
//			blockSize：该参数被goodFeaturesToTrack()函数使用，这个参数应该就是进行非最大值抑制的窗口（真实大小）（类似于Harris中的非最大值抑制窗口），越大，提取到的角点越少（该参数很大程度上影响提取到的角点数）
//			maxPixShift：设定目标在相邻帧之间的最大像素位移
//@Output：无直接输出，直接改变currXY[][2]和ofXY[][2]，另外有可能改变prevXY[][2]（如果重新初始化的话）
//@Innotation：	1）该函数，首先在上一帧中点所在周围的正方形区域，利用goodFeaturesToTrack()函数（使用相同参数，除searcSize不再加2）查找出7个点
//				然后使用金字塔LK函数精确到亚像素对goodFeaturesToTrack()函数提取出的角点进行校正，在计算每个点所在块的SAD值，选取出最小的SAD
//				值作为匹配点
//				2）本函数一般只用到curr当前帧，因为已经利用了prevXY，不需要每次都在图片中重新查找好的角点，只有在跟丢的时候，才会调用初始化函数，重新
//				查找角点，这样可以提高速度，输入的perv就是跟丢的时候，重新初始化用的
//				3）之所以是7个点，是因为经过合理的分析，根据本算法，一般最多就会出现7个干扰点
void calcOF(
	cv::Mat prev,
	cv::Mat curr,
	std::vector<cv::Point2f>& prevXY,
	std::vector<cv::Point2f>& currXY,
	float ofXY[][2],
	int numOF,
	int pyrLKSize,
	int sadBlockSize,
	int blockSize,
	int maxPixShift);

