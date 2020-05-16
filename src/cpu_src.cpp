#include "cpu_src.hpp"

extern ofstream file;

cv::VideoCapture init(int dev){
    cv::VideoCapture cap;
	while(!cap.open(dev));
	return cap;
}

Mat get_edge(Mat res){
	Mat src;
	GaussianBlur(res,src,Size( 5, 5 ),0,0);
	cvtColor(src,src,COLOR_BGR2Luv);
	src.convertTo(src, CV_32FC3);
	subtract(src, Scalar(126,133,124), src);
	pow(src, 2.0, src);
	Mat cls[3];split(src, cls);
	src = cls[0] + cls[1] + cls[2];
	sqrt(src,src);
	src.convertTo(src,CV_8U);
	Canny(src, src, 35, 70);
	return src;
}


Mat get_edge_distance(Mat src){
	register float d1 = 1.5;
	register float d2 = 2.0;
	register float dist0,dist1,dist2,dist3;
	src = get_edge(src);
	bitwise_not(src, src);
	src.convertTo(src, CV_32F);
	
	for(int i=src.rows-2;i>0;i--){
		for(int j=src.cols-2;j>0;j--){
			dist0 = d1 + src.at<float>(i,j+1);
			dist1 = d2 + src.at<float>(i+1,j+1);
			dist2 = d1 + src.at<float>(i+1,j);
			dist3 = d2 + src.at<float>(i-1,j+1);
			src.at<float>(i,j) = min(min(src.at<float>(i,j), min(dist0,dist1)), min(dist2,dist3));
			
		}
	}
	for(int i=1;i<src.rows-1;i++){
		for(int j=1;j<src.cols-1;j++){
			dist0 = d1 + src.at<float>(i,j-1);
			dist1 = d2 + src.at<float>(i-1,j-1);
			dist2 = d1 + src.at<float>(i-1,j);
			dist3 = d2 + src.at<float>(i+1,j-1);
			src.at<float>(i,j) = min(min(src.at<float>(i,j), min(dist0,dist1)), min(dist2,dist3));
		}
	}
	normalize(src,src,0,255,NORM_MINMAX, CV_8U);
	return src;
}

Mat show_text(Mat src, String text){
	Mat padded;
    int m = src.rows + 30;
    copyMakeBorder(src, src, 0, m - src.rows, 0, 0, BORDER_CONSTANT, Scalar::all(0));
	putText(src, text, Point(5,m-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1, 12, false );
	return src;
}

result get_center(Mat src){
	double x=0, y=0, w=0, wt = 0, c;
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			w = 0.1 - src.at<float>(i,j);
			x += j*w;
			y += i*w;
			wt += w;
		}
	}
	c = 10000*wt/(src.rows*src.cols);
	if((c>20)||(c<0))c=0;
	return {Point((int)(x/wt),(int)(y/wt)),c};
}

void locate(Mat src, Mat src1, Mat res){
	matchTemplate(src, src1, res, TM_CCORR_NORMED);	
}
