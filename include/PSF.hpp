#ifndef PSF_HPP
#define	PSF_HPP	

#include <opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>

struct PSF_Config {
	PSF_Config(const double wavelength, const int bins, const cv::Size& sz);
	
	double m_wavelength{0.0};
	int m_bins{ 0 };
	cv::Size m_sz;

	bool validData() const;
};

struct PSF_Data {
	std::vector<cv::Mat>* unwrap = nullptr;
	cv::Mat* mask = nullptr;

	bool validData() const;
};

struct PSF_Result {
	cv::Mat histo;
	int min_x{}, max_x{}, min_y{}, max_y{};
};


class PSF {
public:
	explicit PSF(const PSF_Config& config)
		:m_config{ config }
	{
		if (!m_config.validData())
			throw std::invalid_argument("PSF_Data is invalid \n");
	}


	// 
	PSF_Result computePSF(const PSF_Data& data);

private:
	const PSF_Config& m_config;
};



#endif