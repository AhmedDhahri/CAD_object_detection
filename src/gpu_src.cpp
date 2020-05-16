#include "cpu_src.hpp"
#include "gpu_src.hpp"

extern int sel;
extern ofstream file;

cuda::GpuMat gpu_src;
cuda::Stream stream[SAMPLING];
cuda::GpuMat gpu_cand[SAMPLING];
cuda::GpuMat sel_min_gpu;
Mat sel_min_cpu[SAMPLING];
double sel_min[SAMPLING];

Ptr<cuda::Filter> filter_gauss1  = cuda::createGaussianFilter(5 ,5 , Size(7,7),7,7,BORDER_DEFAULT,-1);	
Ptr<cuda::TemplateMatching> tmpl_match = cuda::createTemplateMatching(CV_8U, TM_CCORR_NORMED);
Ptr<cuda::CannyEdgeDetector> edg_detect = cuda::createCannyEdgeDetector(35, 70, 3, false);   
Ptr<cuda::Filter> filter_gauss  = cuda::createGaussianFilter(CV_8UC3 ,CV_32FC3 , Size(5,5),5,5,BORDER_DEFAULT,-1);
double minVal,maxVal;Point minLoc,maxLoc;
	
cuda::GpuMat get_edge_cuda(cuda::GpuMat src, cuda::Stream s){

	src.convertTo(src,CV_8UC3,s);
	filter_gauss->apply(src,src,s);
	cvtColor(src,src, COLOR_BGR2Luv,0,s);
	cuda::subtract(src,Scalar(126,133,124),src,noArray(),-1,s);
	cuda::pow(src,2,src,s);;
	cuda::GpuMat tmp[3];
	cuda::split(src,tmp,s);
	cuda::add (tmp[0],tmp[2],src,noArray(),-1,s);
	cuda::add (tmp[2],src,src,noArray(),-1,s);
	cuda::sqrt(src,src,s);
	cuda::normalize(src,src,0,255,NORM_MINMAX,CV_8U,noArray(),s);
	edg_detect->detect(src,src,s);
	return src;
}


Mat locate_cuda(Mat src, cuda::GpuMat d_templ){
	Mat res;
	gpu_src.upload(src);
	for(int i=0;i<SAMPLING;i++){
		cuda::resize(d_templ,gpu_cand[i],Size(0,0),1+((double)i/SAMPLING),1+((double)i/SAMPLING),INTER_LINEAR,stream[i]);
		gpu_cand[i] = get_edge_cuda(gpu_cand[i],stream[i]);
		tmpl_match->match(gpu_src, gpu_cand[i], gpu_cand[i],stream[i]);
		filter_gauss1->apply(gpu_cand[i],gpu_cand[i],stream[i]);
		cv::cuda::threshold(gpu_cand[i],gpu_cand[i],0.1,1.0,THRESH_TRUNC,stream[i]);
	}
	
	for(int i=0;i<SAMPLING;i++){
		stream[i].waitForCompletion();
		cuda::minMaxLoc(gpu_cand[i],&minVal,&maxVal,&minLoc,&maxLoc,noArray());
		sel_min[i] = minVal;
	}
	sel = (min_element(sel_min,sel_min+SAMPLING)-sel_min);
	gpu_cand[sel].download(res);
	return res;
}

