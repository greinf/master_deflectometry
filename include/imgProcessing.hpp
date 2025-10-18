#ifndef IMGPROCESSING_H
#define IMGPROCESSING_H

#include "flagHandler.hpp"
#include "imageHandler.hpp"
#include <opencv2/opencv.hpp>


class ImageProcessing {
public:
	ImageProcessing() = default;


private:
	std::vector<cv::Mat> img_phaseshift{};


};





#endif