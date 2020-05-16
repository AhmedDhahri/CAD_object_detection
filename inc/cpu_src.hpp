#ifndef CPU_SRC_HPP
	#define CPU_SRC_HPP
	#include <opencv2/core/core_c.h>
	#include <opencv2/opencv.hpp>
	#include <fstream> 

using namespace std;
using namespace cv;

#define SAMPLING 15

typedef struct{
	Point p;
	double c;
}result;

cv::VideoCapture init(int dev);
Mat get_edge(Mat src);
Mat get_edge_distance(Mat src);
Mat show_text(Mat src, String text);
void locate(Mat src, Mat src1, Mat res);
result get_center(Mat src);
#endif
