#ifndef GRAYCALIBRATION_HPP
#define GRAYCALIBRATION_HPP

#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <memory>
#include "RowPolicy.hpp"
#include <opencv2/core.hpp>
#include "utils.hpp"

// Tag Dispatching for the GrayCalibration Class
namespace gr_calib {
	struct NoCalibration {};
	struct ActiveCalibration {};
	struct PassiveCalibration {};
	struct LUTCalibration {};
}
class ImageStore;

struct Gray_Calib_Result;

class GrayCalibration {
private:
	//void prepareLUT();
	
	ImageStore& m_img_store;

	struct Impl;        

	std::unique_ptr<Impl> m_impl = nullptr;  
	
	//void prepareLUT();

	Gray_Calib_Result fitGamma_LM_andBuildLUT(
		const std::array<double, 256>& measured,
		double sat_cut_rel = 1,    // Sättigung raus
		double eps = 1e-12,
		int max_iters = 100,
		double tol_rel_sse = 1e-10,
		double tol_step = 1e-10,
		double damping = 5.0e-3);


	// Fita function of type I_mess = I_max * (x/255)^lambda 
	// for  the parameters lambda and I_max. This only works if there is no bias 
	// Parameter eps is used for cutting values nere zero -> val < eps is discarded
	// paramter sat_cut cuts values that are near the 
	Gray_Calib_Result fitGamma(
		const std::array<double, 256>& meassured,
		const double eps = 1e-10,
		const double sat_cut = 1
	);

	std::array<std::pair<double, double>, 256> createLut(
		const std::vector<cv::Mat>& images,
		const cv::Mat& mask
	);

	// In place sorting of the according array. 
	bool prepareLUT(
		std::array<std::pair<double, double>, 256>&
	);

	cv::Mat applyLut(
		const cv::Mat&
	);

	double getLutVal(
		const double val
	);

	cv::Mat applyPassive(
		const cv::Mat& img
	);

	Gray_Calib_Result fitGammaBias_LM(
		const std::array<double, 256>& meassured,
		const double eps = 1e-10,
		const double sat_cut = 1
	);

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
		const std::string& path);
		
	/*std::vector<cv::Mat> run_gray_calib(
		const std::vector<cv::Mat>&);*/

	bool doCalibration(
		const CalibrationMethod method,
		const std::vector<cv::Mat>& images,
		const cv::Mat& mask
	);

	cv::Mat applyCalibration(
		const CalibrationMethod method,
		const cv::Mat& image
	);

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

*/

#endif // !GRAYCALIB_HPP
