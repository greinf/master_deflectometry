#ifndef GEOMETRICCALIBRATION_HPP
#define GEOMETRICCALIBRATION_HPP
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>


struct GeometricCalibrationConfig {
	cv::Size2i pattern_size{};
	double point_dist{};
	double wavelength_phase{};
	double displayPixelPitch{};

	bool validData() const { // easy scaleable
		bool ok = true;
		if (!(pattern_size.area() > 0)) return false;
		if (!(point_dist > 0)) return false;
		if (!(wavelength_phase > 0)) return false;
		if (!(displayPixelPitch > 0)) return false;
		return ok;
	}
};

struct GeometricCalibrationData {
	const cv::Mat& camMat;
	const cv::Mat& distCoeffs;

	std::vector<cv::Mat>* unwrap = nullptr;
	std::vector<cv::Mat>* contrast = nullptr;
	std::vector<cv::Mat>* biasIntensity = nullptr;
	
	bool validData() const  {
		bool ok = true;
		if (camMat.empty()) return false;
		if (!(camMat.size() == cv::Size(3, 3))) return false;
		if (camMat.type() != CV_64FC1) return false;
		if (distCoeffs.empty()) return false;
		if (distCoeffs.type() != CV_64FC1) return false;
		if (unwrap == nullptr) return false;
		if (unwrap->size() != 2) return false;
		if (contrast == nullptr) return false;
		if (contrast->size() != 2) return false;
		if (biasIntensity == nullptr) return false;
		if (biasIntensity->size() != 2) return false;

		const cv::Size sz = (*unwrap)[0].size();
		for (const auto& img : *unwrap) {
			if (img.size() != sz) return false;
		}
		for (const auto& img : *contrast) {
			if (img.size() != sz) return false;
		}
		for (const auto& img : *biasIntensity) {
			if (img.size() != sz) return false;
		}
		return ok;
	}
};


struct GeometricCalibrationTestData : GeometricCalibrationData
{
	cv::Mat disp2cam_rvec{};
	cv::Mat disp2cam_tvec{};

	cv::Mat cam2mir_rvec{};
	cv::Mat cam2mir_tvec{};

	cv::Vec3d surfacePoint{};

	bool validData() const {
		if (GeometricCalibrationData::validData() == false) return false;
		if (disp2cam_rvec.empty()) return false;
		if (disp2cam_rvec.type() != CV_64F) return false;
		if (disp2cam_tvec.empty()) return false;
		if (disp2cam_tvec.type() != CV_64F) return false;
		return true;
	}
};

struct ResultNormal_Vectors {
	double n_vectors{};

	std::vector<cv::Vec3d> calc_vectors{};
	cv::Vec3d surface_normal{};

	cv::Mat surface_normal_images{};
	cv::Mat reprojection_error_image{};
};


struct GeometricCalibrationResult {
	cv::Mat disp2cam_rvec{};
	cv::Mat disp2cam_tvec{};

	cv::Mat cam2mir_rvec{};
	cv::Mat cam2mir_tvec{};
};

class ImageProcessing;

class GeometricCalibration {
public:
	explicit GeometricCalibration(const GeometricCalibrationConfig& config, ImageProcessing& process);
	
	GeometricCalibrationResult calibrate(const GeometricCalibrationData&);

	cv::Mat test_calibration(const GeometricCalibrationTestData&);

	~GeometricCalibration();

	GeometricCalibration(const GeometricCalibration&) = delete;
	GeometricCalibration(GeometricCalibration&&) = delete;
	GeometricCalibration& operator=(GeometricCalibration&) = delete;
	GeometricCalibration& operator=(GeometricCalibration&&) = delete;

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl = nullptr;

	ImageProcessing& m_img_processing;

	cv::Mat calculateSurfaceNormals(
		const cv::Mat& rays,
		const cv::Mat& rays_refelcted,
		const cv::Mat& mask
	);

	cv::Mat generateCoordinateImage(
		const cv::Size& sz,
		const int dimension = 2
	);

	
};




#endif