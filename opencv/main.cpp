#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
using namespace dnn;

class pphuman_seg
{
public:
	pphuman_seg();
	Mat inference(Mat cv_image);
private:
	const int inpWidth = 192;
	const int inpHeight = 192;
	const float conf_threshold = 0.5;
	Net net;
	Mat preprocess(Mat srcimg);
};

pphuman_seg::pphuman_seg()
{
	string model_path = "model_float32.onnx";
	this->net = readNet(model_path);    ////opencv4.5.1读取出错，换成opencv4.5.5读取就成功了
}

Mat pphuman_seg::preprocess(Mat srcimg)
{
	Mat dstimg;
	resize(srcimg, dstimg, Size(this->inpWidth, this->inpHeight), INTER_LINEAR);
	dstimg.convertTo(dstimg, CV_32FC3, 1 / (255.0*0.5), -1.0);

	/*int i = 0, j = 0;
	for (i = 0; i < dstimg.rows; i++)   ///用convertTo函数是更高明的方法
	{
		float* pdata = (float*)(dstimg.data + i * dstimg.step);
		for (j = 0; j < dstimg.cols; j++)
		{
			pdata[0] = (pdata[0] / 255.0 - 0.5) / 0.5;
			pdata[1] = (pdata[1] / 255.0 - 0.5) / 0.5;
			pdata[2] = (pdata[2] / 255.0 - 0.5) / 0.5;
			pdata += 3;
		}
	}*/
	return dstimg;
}

Mat pphuman_seg::inference(Mat srcimg)
{
	Mat img = this->preprocess(srcimg);
	Mat blob = blobFromImage(img);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	/// post process.																																					
	float *mask_ptr = (float*)outs[0].data;
	const int out_h = outs[0].size[1];
	const int out_w = outs[0].size[2];
	Mat mask_out(out_h, out_w, CV_32FC2, mask_ptr);
	Mat segmentation_map;
	resize(mask_out, segmentation_map, Size(srcimg.cols, srcimg.rows));
	
	Mat dstimg = srcimg.clone();
	for (int h = 0; h < srcimg.rows; h++)
	{
		for (int w = 0; w < srcimg.cols; w++)
		{
			float pix = segmentation_map.ptr<float>(h)[w * 2];
			if (pix > this->conf_threshold)
			{
				float b = (float)srcimg.at<Vec3b>(h, w)[0];
				dstimg.at<Vec3b>(h, w)[0] = uchar(b * 0.5 + 1);
				float g = (float)srcimg.at<Vec3b>(h, w)[1];
				dstimg.at<Vec3b>(h, w)[1] = uchar(g * 0.5 + 1);
				float r = (float)srcimg.at<Vec3b>(h, w)[2];
				dstimg.at<Vec3b>(h, w)[2] = uchar(r * 0.5 + 1);
			}
		}
	}

	for (int h = 0; h < srcimg.rows; h++)
	{
		for (int w = 0; w < srcimg.cols; w++)
		{
			float pix = segmentation_map.ptr<float>(h)[w * 2 + 1];
			if (pix > this->conf_threshold)
			{
				float b = (float)dstimg.at<Vec3b>(h, w)[0];
				dstimg.at<Vec3b>(h, w)[0] = uchar(b * 0.5 + 1);
				float g = (float)dstimg.at<Vec3b>(h, w)[1] + 255.0;
				dstimg.at<Vec3b>(h, w)[1] = uchar(g * 0.5 + 1);
				float r = (float)dstimg.at<Vec3b>(h, w)[2];
				dstimg.at<Vec3b>(h, w)[2] = uchar(r * 0.5 + 1);
			}
		}
	}
	return dstimg;
}

int main()
{
	const int use_video = 0;
	pphuman_seg mynet;
	if (use_video)
	{
		cv::VideoCapture video_capture(0);  ///也可以是视频文件
		if (!video_capture.isOpened())
		{
			std::cout << "Can not open video " << endl;
			return -1;
		}

		cv::Mat frame;
		while (video_capture.read(frame))
		{
			Mat dstimg = mynet.inference(frame);
			string kWinName = "Deep learning OpenCV with pphuman seg";
			namedWindow(kWinName, WINDOW_NORMAL);
			imshow(kWinName, dstimg);
			waitKey(1);
		}
		destroyAllWindows();
	}
	else
	{
		string imgpath = "testimgs/1.jpg";
		Mat srcimg = imread(imgpath);
		Mat dstimg = mynet.inference(srcimg);

		namedWindow("srcimg", WINDOW_NORMAL);
		imshow("srcimg", srcimg);
		static const string kWinName = "Deep learning OpenCV with pphuman seg";
		namedWindow(kWinName, WINDOW_NORMAL);
		imshow(kWinName, dstimg);
		waitKey(0);
		destroyAllWindows();
	}
}