#ifndef GRAYCODEDECODER_HPP
#define GRAYCODEDECODER_HPP	
 
#include <vector>
#include <opencv2/opencv.hpp>
#include <stddef.h>
#include <boost/dynamic_bitset.hpp>
#include "GrayCodeConfig.hpp"


class ImageProcessing;
struct GrayCodeConfig;

class GrayCodeDecoder {
private:
	ImageProcessing& m_img_processing;

	struct Samples {
		Samples() = default;
		
		enum orientation {
			x_dir,
			y_dir
		} orientation{};

		boost::dynamic_bitset<> m_data{};

		int m_result{ -1 };

		int pixel_x{-1};
		int pixel_y{-1};
	};
	
	// Creates Binary Images from GrayValue pics
	// The last Picutre is a Mask to get valid pixels
	// all pictures 
	std::vector<cv::Mat> binaries(
		const std::vector<cv::Mat>& img,
		GrayCodeConfig& config
	);

	// Input[0]: const std::vector<cv::Mat>&. Vector of all images for one direction,
	// Input[1]: GrayCodeConfig&: Holds Information about the used GrayCalib
	// -> Hold the Information about Bitdepth for x and y 
	// Input[2]: Out Variable. Sample struct that holds GrayCodeinformation for one Pixel
	// -> Holds Inforamtion about pixel Position. GrayCode Value
	void extractSample(
		const std::vector<cv::Mat>& img,
		GrayCodeConfig& config,
		Samples& sam
	);

	// Input[0]: GrayCodeConfig-> holds Information about the GrayCode 
	// Input[1]: Stuct to hold the GrayCodedata. 
	// -> Does hold a boost::dynamic_bitset where the bits are set according to the images
	// -> Function does the transfer from GrayCode to Bin 
	// -> Value is saved to int .result
	void gray2dec(
		GrayCodeConfig& config,
		Samples& sam
	);

public:
	GrayCodeDecoder(ImageProcessing& imgProcess)
		:m_img_processing{imgProcess}
	{ }

	void decoding(GrayCodeConfig&);


};

#endif