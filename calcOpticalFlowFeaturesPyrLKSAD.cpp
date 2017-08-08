/*
* 文件名称：Matcher.h
* 依赖库：opencv3.1.0或其他版本
* 测试环境：VS2015 x64
* 摘 要：	利用限制性特征点检测+金字塔LK精化+最多7次SAD匹配计算光流（SAD：计算块内所有像素对应灰度值差的绝对值的和）
*			限制性：指的是，真正用于光流计算的特征点，是从提取出的众多点中，选择出满足一定的约束条件的点，点与点之间满足一定的距离：maxPixShift+blockSize
*			金字塔LK：指的是，特征点检测完之后，使用金字塔LK算法校正特征点提取的值并精确到亚像素
*			最多7次SAD匹配：每一次在下一帧对应区域提取出至多7个点，然后再7个点中进行SAD匹配，选出最小SAD值作为匹配点
* 关键点：限制距离，7次匹配，重新初始化（重新初始化是非常重要的，当点跟丢时，就需要重新初始化，否则将继续跟踪上一帧中的点）
* 注意：本程序计算出来的光流个数是固定的，即ofXY的维数是不变的；本程序是单纯的光流计算函数，不涉及角速率补偿以及光流排序及选择等操作
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

#include "calcOpticalFlowFeaturesPyrLKSAD.h"

void calcOpticalFlowFeaturesPyrLKSAD(
	cv::Mat prev,
	cv::Mat curr,
	std::vector<cv::Point2f>& prevXY,
	std::vector<cv::Point2f>& currXY,
	float ofXY[][2],
	bool needInit,
	int numOF,
	int pyrLKSize,
	int sadBlockSize,
	int blockSize,
	int maxPixShift)
{
	int numPoints = 30 * numOF;
	if (needInit)
	{
		initGoodFeatures(
			prev,
			prevXY,
			numOF,
			numPoints,
			maxPixShift,
			blockSize);
	}

	calcOF(
		prev,
		curr,
		prevXY,
		currXY,
		ofXY,
		numOF,
		pyrLKSize,
		sadBlockSize,
		blockSize,
		maxPixShift);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/*----------------------------------------------------函数分割线----------------------------------------------------*/
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

void initGoodFeatures(
	cv::Mat prev,
	std::vector<cv::Point2f>& prevXY,
	int numOF,
	int numPoints,
	int maxPixShift,
	int blockSize)
{
	int minDis = maxPixShift + blockSize;
	cv::Mat prev_copy;
	prev.copyTo(prev_copy);

	std::vector<cv::Point2f> goodCorners;
	cv::goodFeaturesToTrack(
		prev,						//要提取goodFeatures的图像区域
		goodCorners,				//用来存储提取到的好角点矢量容器
		numPoints,					//提取出并存入到goodCorners的最大角点数，这里等于numOF，即为所需计算光流的个数
		0.001,						//质量等级，越大，提取到的角点质量越好，提取到角点数越少
		0,							//提取到角点之间的最小距离，设为0，使得角点之间不受距离影响
		cv::Mat(),					//定义需要找角点的区域（mask）
		blockSize,					//blockSize，这个参数应该就是进行非最大值抑制的窗口（类似于Harris中的非最大值抑制窗口），越大，提取到的角点越少（该参数很大程度上影响提取到的角点数）
		true,						//是否使用Harris角点检测
		0.01);						//k参数：Harris角点检测中的参数，详见opencv手册或者《图像局部不变性特征与描述》一书50页，k就是alpha

	int detected = goodCorners.size();	//提取到的点的个数不一定有numPoints
	int rows = prev.rows;
	int cols = prev.cols;
	int counter = 0;

	for (int i = 0;i < detected;i++)
	{
		int xx = goodCorners[i].x;
		int yy = goodCorners[i].y;

		bool qualified = false;
		/******这个if语句中的判别函数非常重要，判断是否接近边缘，如果接近边缘，则不是很好的跟踪点，可以更改其范围，使得找出的角点长时间适合跟踪，以减少较为耗时的初始化过程******/
		/*if (xx > (2 * minDis + 1) && xx < (cols - 2 * minDis + 1) && yy >(2 * minDis + 1) && yy < (rows - 2 * minDis + 1))*/
		if (xx >(2 * minDis + 1) && xx < (cols - 2 * minDis + 1) && yy >(2 * minDis + 1) && yy < (rows - 2 * minDis + 1))
		{
			qualified = true;
			for (int j = 0;j < i;j++)
			{
				//这个if语句是判断该角点与比其角点响应值更大的所有角点是否存在一定的距离，如果小于给定的搜索距离，则舍弃，这样可有效避免这些响应值更大的点的干扰，加1只是为了确保每必要的可能影响
				if (abs(xx - goodCorners[j].x)<= minDis +1 && abs(yy - goodCorners[j].y) <= minDis +1)
				{
					qualified = false;
					break;
				}
			}
			if (qualified)
			{
				prevXY.push_back(goodCorners[i]);
				counter++;
				if (counter >= numOF)
				{
					break;					//找满numOF个适合跟踪的好角点，跳出循环
				}
			}
		}
	}

	//如果在numPoints中没有找到合适的跟踪点，则递归调用本函数，numPoints翻倍，找出更多的点再选择
	if (prevXY.size() >= numOF)
	{
		//递归函数必要要有终止环，所以这里必须把cornerSubPix()函数放入到else中，不能直接放到外面
		//精确到亚像素
		cv::cornerSubPix(prev, prevXY, cv::Size(blockSize / 2, blockSize / 2), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 0.01, 30));

		std::vector<cv::Point2f>().swap(goodCorners);

		return;
	}
	else
	{
		prevXY.clear();					//清空prevXY
		numPoints *= 2;
		
		//如果numPoints>3000还没找到角点，那就找不到了
		if (numPoints > 3000)
		{
			std::cout << "检测不到适合计算光流并进行跟踪的角点！" << std::endl << std::endl;
			assert(0);
			return;
		}
		initGoodFeatures(
			prev,
			prevXY,
			numOF,
			numPoints,
			maxPixShift,
			blockSize);
	}
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/*----------------------------------------------------函数分割线----------------------------------------------------*/
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

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
	int maxPixShift)
{
	//前面在提取点时候，限制的距离是maxPixShift+blockSize，因为接下来还有一个金字塔LK过程，因此，要加上pyrLKSize，防止超出图像边缘
	int minDis = maxPixShift + pyrLKSize + blockSize;		//用于边界检查，防止超出图像边缘
	int rows = curr.rows;
	int cols = curr.cols;
	int counterFailure = 0;						//计算出光流异常的点
	bool reInit = false;						//是否需要重新初始化标志位

	//金字塔光流精化计算函数用到的参数
	std::vector<uchar> status;					//must be uchar, can't be char!!!!!
	std::vector<float> err;
	cv::Size winSize(7, 7);
	int maxLevel = 3;							//金字塔层数
	cv::TermCriteria termCrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	int flag = 0;
	double minEigThreshold = 1e-3;				//金字塔LK迭代终止条件2

	for (int i = 0;i < numOF;i++)
	{
		int xx = prevXY[i].x;
		int yy = prevXY[i].y;

		//在这检测点是否可能跟丢：如果搜索窗口在图像区域之外，则重新初始化
		if ((xx - minDis) < 0 || (xx + minDis) >= cols || (yy - minDis) < 0 || (yy + minDis) >= rows)
		{
			//重新初始化之后，将有一组新的prevXY
			reInit = true;
			break;
		}

		//根据最大像素偏移，确定搜索区域（+1是因为Range()右不包含）
		cv::Mat searchRegion = curr(cv::Range(prevXY[i].y - maxPixShift - blockSize, prevXY[i].y + maxPixShift + blockSize + 1),
			cv::Range(prevXY[i].x - maxPixShift - blockSize, prevXY[i].x + maxPixShift + blockSize + 1));

		std::vector<cv::Point2f> corner;
		cv::goodFeaturesToTrack(
			searchRegion,				//搜索区域
			corner,						//存储点的矢量容器
			7,							//找7个点，一般最多出现7个干扰点，如果图片质量实在变化的剧烈，至多算是一个跟丢的点，不用这个数据或者积累到3个重新初始化就可以了
			0.001,						//质量等级，越大，提取到的角点质量越好，提取到角点数越少
			0,							//提取到角点之间的最小欧式像素距离，越大，提取到的角点越少
			cv::Mat(),					//定义需要找角点的区域（mask）
			blockSize,					//blockSize，这个参数应该就是进行非最大值抑制的窗口，真实尺寸
			true,						//是否使用Harris角点检测
			0.01);						//k参数：Harris角点检测中的参数，详见opencv手册或者《图像局部不变性特征与描述》一书50页，k就是alpha

		//上一帧中金字塔LK的搜索区域
		cv::Mat prev_patch = prev(cv::Range(prevXY[i].y - pyrLKSize, prevXY[i].y + pyrLKSize + 1),
			cv::Range(prevXY[i].x - pyrLKSize, prevXY[i].x + pyrLKSize + 1));

		//上一帧金字塔LK搜索区域中点的坐标（恒以点为中心，金字塔搜索区域就是以该点为中心的一个区域，因此上一帧中点的坐标恒为搜索区域的中心点的相对坐标）
		std::vector<cv::Point2f> preCorner;
		preCorner.push_back(cv::Point2f(pyrLKSize, pyrLKSize));

		int sad[7];							//7个点的SAD值

		int detectedNum = corner.size();	//虽说指定提取7个点，但有可能出现提取不到7个点的情况
		cv::Point2f matchedCorner;			//7个点中最终匹配成功的点
		int minSAD = 10000;					//初始最小SAD值（稍微取大一点）
		int counterOverflow = 0;
		for (int j = 0;j < detectedNum;j++)
		{
			//这个curr_patch很关键，因为LK不适用于大运动（虽然使用了金字塔），因此前面先使用goodFeaturesToTrack()函数检测点的位置，
			//再使用LK精化，这个时候，curr_patch就是以goodFeaturesToTrack()检测出的点为中心的周围的一个区域，这样，上一帧和下一帧在开始时刻，
			//点的位置都是在搜索区域中心点，LK精化后的坐标减去中心点的坐标就是LK精化过程变化的坐标，加上goodFeaturesToTrack()检测出的坐标，
			//就是最终得到的亚像素的点的坐标
			corner[j].x += prevXY[i].x - maxPixShift - blockSize;	//根据goodFeaturesToTrack()检测结果，定位到当前帧整幅图片的相对坐标
			corner[j].y += prevXY[i].y - maxPixShift - blockSize;

			//提取LK搜索区域
			cv::Mat curr_patch = curr(cv::Range(corner[j].y - pyrLKSize, corner[j].y + pyrLKSize+1),
				cv::Range(corner[j].x - pyrLKSize, corner[j].x + pyrLKSize+1));

			//点在下一帧金字塔LK搜索区域中的坐标
			std::vector<cv::Point2f>nextCorner;	//pyrLK精化后的坐标

			cv::calcOpticalFlowPyrLK(
				prev_patch,
				curr_patch,
				preCorner,
				nextCorner,
				status,
				err,
				winSize,
				maxLevel,
				termCrit,
				flag,
				minEigThreshold);

			//不知道什么原因，金字塔LK会算出一些奇怪的值：超出图片范围的坐标，这里设定一个机制，滤除这些点
			//如果算出奇怪的值，那肯定这个点有问题，要么图像质量变弱，要么，前面特征点提取不精确，以至于prev_patch和curr_patch风马牛不相及，
			//最后一种情况，就是LK的不稳定了，总之，需要过滤这些点
			if (nextCorner[0].x<0 || nextCorner[0].y<0 || nextCorner[0].x > 2*pyrLKSize || nextCorner[0].y > 2*pyrLKSize) 
			{ 
				counterOverflow++;
				//如果LK结果全部算出奇异值（超出搜索区域），则直接取为响应值最大的点位匹配点，可能有偏差，但不至于过大（如果没有这个if，
				//那么这个点在下帧的坐标就没有赋值了（从输出好像可以看出默认是（0,0）），这样就不好了）
				if (counterOverflow == 7)
				{
					matchedCorner = corner[0];
				}
				continue; 
			}
			
			//金字塔LK精化后变化的坐标+精化前点在当前帧整幅图片的坐标，就是最终的点在当前帧中的新坐标（亚像素精度）
			corner[j].x += nextCorner[0].x - preCorner[0].x;
			corner[j].y += nextCorner[0].y - preCorner[0].y;

			//先分别提取SAD计算块，然后计算SAD值
			cv::Mat prevPatch = prev(cv::Range(prevXY[i].y - sadBlockSize, prevXY[i].y + sadBlockSize+1),
				cv::Range(prevXY[i].x - sadBlockSize, prevXY[i].x + sadBlockSize+1));
			cv::Mat currPatch = curr(cv::Range(corner[j].y - sadBlockSize, corner[j].y + sadBlockSize+1),
				cv::Range(corner[j].x - sadBlockSize, corner[j].x + sadBlockSize+1));

			sad[j] = cv::norm(prevPatch, currPatch, cv::NORM_L1);

			if (sad[j] < minSAD)
			{
				minSAD = sad[j];
				matchedCorner = corner[j];
			}
		}

		//std::cout << counterOverflow <<", "<< detectedNum<< std::endl;

		currXY.push_back(matchedCorner);

		//上下帧坐标差值即为光流（当前帧-上一帧）
		ofXY[i][0] = currXY[i].x - prevXY[i].x;
		ofXY[i][1] = currXY[i].y - prevXY[i].y;
		
		//std::cout << ofXY[i][0] << ", " << ofXY[i][1] << std::endl;

		//当计算出的光流大于指定最大像素位移时，说明这个点跟丢（比如这个点被遮挡），当跟丢点数达到3时，那么重新初始化，重新计算光流
		//之所以是3，是根据配套的舍弃策略来定的，最终将在XY方向各舍去4个点（两个最大值，两个最小值），最多舍去8个点，这样，达到3的时候，就有可能
		//舍去不掉了（超过最大像素位移的点，肯定是计算错误的点）
		if (abs(ofXY[i][0]) >= maxPixShift || abs(ofXY[i][1]) >= maxPixShift)
		{
			counterFailure++;
			if (counterFailure == 4)
			{
				reInit = true;
				break;
			}
		}
	}

	if (reInit)
	{
		//重新初始化，这两个一定要清零
		prevXY.clear();
		currXY.clear();
		initGoodFeatures(
			prev,
			prevXY,
			numOF,
			40*numOF,
			maxPixShift,
			blockSize);

		calcOF(
			prev,
			curr,
			prevXY,
			currXY,
			ofXY,
			numOF,
			pyrLKSize,
			sadBlockSize,
			blockSize,
			maxPixShift);
	}
}