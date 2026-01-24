#ifndef GRAYCALIBRATION_HPP
#define GRAYCALIBRATION_HPP

#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <memory>
#include "RowPolicy.hpp"
#include <opencv2/core.hpp>


// Tag Dispatching for the GrayCalibration Class
namespace gr_calib {
	struct NoCalibration {};
	struct ActiveCalibration {};
	struct PassiveCalibration {};
	struct LUTCalibration {};
}
class ImageStore;



class GrayCalibration {
private:
	//void prepareLUT();
	
	ImageStore& m_img_store;

	struct Impl;        

	std::unique_ptr<Impl> m_impl = nullptr;  
	
	//void prepareLUT();

public:
	explicit GrayCalibration(ImageStore& image_store);
	~GrayCalibration();

	bool setupCalibrationMethod(gr_calib::NoCalibration) 
	{ return true; }

	bool setupCalibrationMethod(
		const gr_calib::ActiveCalibration,
		const std::string& path);

	bool setupCalibrationMethod(
		const gr_calib::PassiveCalibration,
		const std::string& path);

	bool setupCalibrationMethod(
		const gr_calib::LUTCalibration,
		std::vector<std::pair<double, double>> grayLut);
		
	/*std::vector<cv::Mat> run_gray_calib(
		const std::vector<cv::Mat>&);*/

};

/*
double linear_gray(double);


//Look Up Table
std::optional<std::vector<std::pair<double, double>>> m_LUT{};
std::vector<std::pair<double, double>> m_sortedLUT{};
bool m_lut_ready{ false };
double m_lut_scale_factor{};
double m_lut_offset{};



void load_gray_calib_data(std::vector<std::pair<double, double>>&& lut) {
	if (m_LUT.has_value()) {
		std::cout << "WARNING: Old LUT gets overridden \n";
	}
	m_LUT.emplace(std::move(lut));
	prepareLUT();
}


// Even more expensive computation
// Same reasoning as above.
double mean = mean_value / amplitude;
pattern = mean * (1.0 + cosine);
pattern *= amplitude;
if (m_LUT.has_value()) {
	cv::parallel_for_(cv::Range(0, pattern.rows),
		[&](const cv::Range& r) {
			for (int y = r.start; y < r.end; ++y) {
				double* ptr = pattern.ptr<double>(y);
				for (int x = 0; x < pattern.cols; ++x) {
					ptr[x] = linear_gray(ptr[x]);
				}
			}
		});
}


double Pattern::linear_gray(double s)
{
	if (!m_lut_ready || m_sortedLUT.empty()) {
		std::cout << "WARNING: LUT not found. Create IMG without calib \n";
		return s; // fallback: no LUT active
	}

	// transform 0..255 range into LUT-range
	double target = s * m_lut_scale_factor + m_lut_offset;

	// binary search on "second" values
	auto it = std::lower_bound(
		m_sortedLUT.begin(),
		m_sortedLUT.end(),
		target,
		[](const auto& a, double val) {
			return a.second < val;
		}
	);

	if (it == m_sortedLUT.begin())
		return it->first;

	if (it == m_sortedLUT.end())
		return std::prev(it)->first;

	// choose closer of the two neighbors
	double hi_dist = std::abs(it->second - target);
	double lo_dist = std::abs(std::prev(it)->second - target);

	if (lo_dist < hi_dist)
		return std::prev(it)->first;
	else
		return it->first;
}

*/

#endif // !GRAYCALIB_HPP
