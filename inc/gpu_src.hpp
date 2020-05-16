#ifndef GPU_SRC_HPP
	#define GPU_SRC_HPP
	#include <opencv2/core/cuda.hpp>
	#include <opencv2/cudaimgproc.hpp>
	#include <opencv2/cudaarithm.hpp>
	#include <opencv2/cudawarping.hpp>
	#include <opencv2/cudafilters.hpp>
	
	using namespace std;
	using namespace cv;


	//Thershhold
	//sampling
	//srg color
	//canny thersh
	cuda::GpuMat  get_edge_cuda(cuda::GpuMat src, cuda::Stream s);
	Mat locate_cuda(Mat src, cuda::GpuMat d_templ);
#endif
