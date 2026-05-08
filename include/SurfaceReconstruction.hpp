#ifndef SURFACE_RECONSTRUCTION_HPP	
#define SURFACE_RECONSTRUCTION_HPP
#include <opencv2/opencv.hpp>
#include <vector>

class SurfaceReconstruction_config {
	cv::Size sz{};

};

class SurfaceReconstruction_data {
	std::vector<cv::Mat>* unwrap_prim{};
	std::vector<cv::Mat>* unwrap_secon{};
	std::vector<cv::Mat>* contrast_prim{};
	std::vector<cv::Mat>* contrast_secon{};

	cv::Mat* mask_prim{};
	cv::Mat* mask_second{};
	cv::Point2d PixelCoordiante{};
};

class SurfaceReconstruction {
	SurfaceReconstruction(const SurfaceReconstruction_config& config)
};

#endif