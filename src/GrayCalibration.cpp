#include "GrayCalibration.hpp"
#include "cassert"
#include <memory>
#include "imageStore.hpp"
#include "RowPolicy.hpp"
#include <opencv2/core.hpp>


struct GrayCalibration::Impl {
	std::vector<cv::Mat> active_gray{};
	std::vector<cv::Mat> passive_gray{};
	std::vector<std::pair<double, double>> gray_Lut{};
};


GrayCalibration::GrayCalibration(ImageStore& imgStore)
	:m_img_store{ imgStore }
	,m_impl{ std::make_unique<Impl>() }
	{}

GrayCalibration::~GrayCalibration() = default;


bool GrayCalibration::setupCalibrationMethod(
	const gr_calib::ActiveCalibration,
	const std::string& pat)
{
	
	

	return false;
}

bool GrayCalibration::setupCalibrationMethod(
	const gr_calib::PassiveCalibration,
	const std::string& path)
{
	assert(!path.empty() && "Path to active Calibration Frames is needed \n");
	return false;
}

bool GrayCalibration::setupCalibrationMethod(
	const gr_calib::LUTCalibration,
	std::vector<std::pair<double,double>> grayLut)
{
	assert(!grayLut.empty());
	assert(grayLut.size() > 255);
	m_impl->gray_Lut = grayLut;


	return true;
}


//void GrayCalibration::prepareLUT()
//{
//	if (!m_LUT.has_value()) {
//		m_lut_ready = false;
//		return;
//	}
//
//	m_sortedLUT = m_LUT.value();
//	std::sort(m_sortedLUT.begin(), m_sortedLUT.end(),
//		[](auto& a, auto& b) { return a.second < b.second; });
//
//	double minv = m_sortedLUT.front().second;
//	double maxv = m_sortedLUT.back().second;
//
//	double range = maxv - minv;
//	double range_safety = range * 0.9;      // keep 10% margin
//
//	m_lut_scale_factor = range_safety / 255.0;
//	m_lut_offset = minv + range * 0.05;
//
//	m_lut_ready = true;
//}
