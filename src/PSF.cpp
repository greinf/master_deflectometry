#include "PSF.hpp"
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>


bool PSF_Data::validData() const
{
	// Mask check
	if (mask == nullptr) return false;
	if (mask->type() != CV_8U) return false;
	cv::Size working_sz = mask->size();
	if (unwrap == nullptr) return false;
	if (!std::all_of(unwrap->begin(), unwrap->end(),
		[&](const cv::Mat& img) -> bool
		{
			return (img.size() == working_sz) &&
				(img.type() == CV_64F);
		})
		) return false;

	return true;
}

PSF_Config::PSF_Config(const double wavelength, const int bins, const cv::Size& sz)
	:m_wavelength{wavelength}
	,m_bins{bins}
	,m_sz{ sz } {}

bool PSF_Config::validData() const
{
	if (m_wavelength < 1) return false;
	if (m_bins < 1) return false;
	if (m_sz.empty()) return false;
	if (m_sz.area() <= 0) return false;

	return true;
}


PSF_Result PSF::computePSF(const PSF_Data& data)
{
	if (data.validData() == false) 
		throw std::invalid_argument("Data invalid for PSF-Computation");

	PSF_Result result{};
	// Working size. sz for all the camera images. 
	const cv::Size sz = data.mask->size();
	// The histogramm size is with respect to the dispaly coordiantes -> m_config.size
	result.histo = cv::Mat(m_config.m_sz, CV_16U, cv::Scalar(0));
	cv::Mat& histo = result.histo;

	int& min_x{ result.min_x }, max_x{result.max_x},
		min_y{result.min_y}, max_y{result.max_y};

	cv::Mat 
		unwrapx = (*data.unwrap)[0].clone(), 
		unwrapy = (*data.unwrap)[1].clone();
		
	//cv::Mat nanMask = cv::Mat(unwrapx == unwrapx) & (unwrapy == unwrapy);
	unwrapx = cv::max(unwrapx, 0.0);
	unwrapy = cv::max(unwrapy, 0.0);
	
	unwrapx *= (m_config.m_wavelength / CV_2PI); 
	unwrapy *= (m_config.m_wavelength / CV_2PI);

	// Binning 
	unwrapx /= m_config.m_bins;
	unwrapy /= m_config.m_bins;
	unwrapx.forEach<double>([](double& val, const int* pos) -> void {
		val = std::floor(val);
		});
	unwrapy.forEach<double>([](double& val, const int* pos) -> void {
		val = std::floor(val);
		});

	unwrapx *= m_config.m_bins;
	unwrapy *= m_config.m_bins;

	/*cv::parallel_for_(cv::Range(0, sz.height),
		[&](const cv::Range& range) {*/

	for (int row = 0; row < sz.height; ++row) {
		const double* x_ptr = unwrapx.ptr<double>(row);
		const double* y_ptr = unwrapy.ptr<double>(row);
		const uchar* mask_ptr = data.mask->ptr<uchar>(row);
		for (int col = 0; col < sz.width; ++col) {
			if (mask_ptr[col] == 0 ||
				std::isnan(x_ptr[col]) ||
				std::isnan(y_ptr[col])) continue;
			cv::Point2i coordinates(
				static_cast<int>(x_ptr[col]),
				static_cast<int>(y_ptr[col]));

			if (coordinates.x < 0 ||
				(coordinates.x + m_config.m_bins) > m_config.m_sz.width)
			{
				std::cout << "X out of Range: \nx:" << coordinates.x << "\n" <<
					"y:" << coordinates.y << std::endl;
				continue;
			}
			if (coordinates.y < 0 ||
				(coordinates.y + m_config.m_bins) > m_config.m_sz.height)
			{
				std::cout << "Y out of Range: \nx:" << coordinates.x << "\n" <<
					"y:" << coordinates.y << std::endl;
				continue;
			}
			max_x = std::max(static_cast<int>(coordinates.x), max_x);
			min_x = std::min(static_cast<int>(coordinates.x), min_x);

			max_y = std::max(static_cast<int>(coordinates.y), max_y);
			min_y = std::min(static_cast<int>(coordinates.y), min_y);

			if (m_config.m_bins == 1) {
				histo.ptr<std::uint16_t>(coordinates.y)[coordinates.x] += 1;
			}
			else {
				cv::Mat roi =
					histo(cv::Rect(
						coordinates.x,
						coordinates.y,
						m_config.m_bins,
						m_config.m_bins));
				roi += 1;
			}
		}
	}
		
	return result;
}
