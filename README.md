# calcOpticalFlowPyrLKSAD
Calculate the sparse optical flow using pyramid Lucas-Kanade and sum of differences (SAD) matching.

Three steps to calculate sparse optical flow (for more information, please refer to code comments (in Chinese)):

1) Extract keypoint using cv::goodFeaturesToTrack(), and find out sevral points for optical flow calculation. The points used are selected by a "strange" strategy. That is for any two points, the Euclidean distance is less than the assumed max pixel shift between consecutive frames.

2) Use cv::calcOpticalFlowPyrLK() to track and refine the pixel coordinates of the points.

3) Use SAD matcher to find out correspondences and calculate the optical flow (thanks to the aforementioned "strange" strategy, we don't need to do much work on SAD matching).
