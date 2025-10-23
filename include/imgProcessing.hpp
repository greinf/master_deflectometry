#ifndef IMGPROCESSING_H
#define IMGPROCESSING_H

#include "flagHandler.hpp"
#include "imageHandler.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class ImageProcessing {
public:
	ImageProcessing();

	void wrapped_phase();

	void bayerToGray();

	void unwrapped_phase();
	void goldsteinUnwrap();

	void saveImages(std::string& path);

	std::array<cv::Mat, (std::size_t) 2> m_baseIntensity;
	std::array<cv::Mat, (std::size_t) 2> m_contrast;
	std::array<cv::Mat, (std::size_t) 2> m_phase;

	//void convertToFloat(std::vector);
	~ImageProcessing();
	ImageProcessing(const ImageProcessing&) = delete;
	ImageProcessing& operator=(const ImageProcessing&) = delete;
	ImageProcessing(ImageProcessing&&) = delete;
	ImageProcessing& operator=(ImageProcessing&&) = delete;

	void assginFrames(std::vector<cv::Mat>&& mat) {
		m_frames = mat;
	}


private:	
	// If number_of_frames_per_pattern > 0. Here mean values get stored. 
	// Starting point for Image Processing steps. 
	static inline int instance_counter{ 0 };
	std::vector<cv::Mat> m_frames{};
	std::vector<cv::Mat> m_raw_phase{};
	
	std::array<cv::Mat, (std::size_t)2> m_unwrapped_phase{};
	std::array<cv::Mat, (std::size_t)2> m_wrapped_phase{};
	cv::Mat mean(std::vector<cv::Mat>);

	std::array<cv::Mat, (std::size_t)2> m_s1{};
	std::array<cv::Mat, (std::size_t)2> m_s2{};
	std::array<cv::Mat, (std::size_t)2> m_s3{};
	
	std::string path{ "C:\\Users\\grein\\Desktop\\Master\\Project\\deflectometrie\\out" };
	void create(std::vector<cv::Mat>& vec);
	cv::Mat load_images(std::string path_to_image);

};





#endif