#ifndef GEOMETRICCALIBRATION_HPP
#define GEOMETRICCALIBRATION_HPP
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <Eigen/dense>
#include <fstream>



struct GeometricCalibrationConfig {
public:
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
	cv::Mat camMat;
	cv::Mat distCoeffs;

	cv::Mat mask_ROI;

	std::string path{};

	std::vector<cv::Mat>* unwrap = nullptr;
	std::vector<cv::Mat>* contrast = nullptr;
	std::vector<cv::Mat>* biasIntensity = nullptr;
	
	bool validData() const  {
		bool ok = true;
		if (camMat.empty() == true) return false;
		if (camMat.size() != cv::Size(3, 3)) return false;
		if (camMat.type() != CV_64F) return false;
		if (distCoeffs.empty() == true) return false;
		if (distCoeffs.type() != CV_64F) return false;
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

struct GeometricCalibrationData_Stereo : GeometricCalibrationData {
	 cv::Mat camMat_secundaryCam;
	 cv::Mat distCoeffs_secondaryCam;

	 cv::Mat mask_ROI_secon;

	 cv::Mat cam2cam_tvec;
	 cv::Mat cam2cam_rotMat;

	 std::vector<cv::Mat>* biasIntensity_sec_cam = nullptr;
	 std::vector<cv::Mat>* contrast_sec_cam = nullptr;
	 std::vector<cv::Mat>* unwrap_sec_cam = nullptr;


	 bool validData() const {
		 if (GeometricCalibrationData::validData() == false) return false;
		 if (camMat_secundaryCam.empty() == true) return false;
		 if (camMat_secundaryCam.size() != cv::Size(3, 3)) return false;
		 if (camMat_secundaryCam.type() != CV_64FC1) return false;
		 if (distCoeffs_secondaryCam.empty() == true) return false;
		 if (distCoeffs_secondaryCam.type() != CV_64FC1) return false;
		 if (cam2cam_tvec.empty() == true) return false;
		 if (cam2cam_tvec.type() != CV_64FC1) return false;
		 if (cam2cam_rotMat.empty() == true) return false;
		 if (cam2cam_rotMat.type() != CV_64FC1) return false;
		 if (biasIntensity_sec_cam == nullptr) return false;
		 if (biasIntensity->size() != 2) return false;
		 if (contrast_sec_cam->empty() == true) return false;
		 if (contrast_sec_cam->size() != 2) return false;
		 /*if (unwrap_sec_cam->empty() == true) return false;
		 if (unwrap_sec_cam->size() != 2) return false;*/

		 return true;
	 }
};

struct GeometricCalibrationTestData : GeometricCalibrationData
{
	cv::Mat disp2cam_rvec{};
	cv::Mat disp2cam_tvec{};

	cv::Mat mir2cam_rvec{};
	cv::Mat mir2cam_tvec{};

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
namespace open3d {
	namespace geometry {
		class PointCloud;
	}
}

struct StereoCalibrationPoints {
	StereoCalibrationPoints() = delete;

	std::vector<cv::Point2f> imagePoints_prim{};
	std::vector<cv::Point2f> imagePoints_secon{};
};

class GeometricCalibration {
public:
	explicit GeometricCalibration(const GeometricCalibrationConfig& config, ImageProcessing& process);
	
	GeometricCalibrationResult calibrateMono(const GeometricCalibrationData&);

	GeometricCalibrationResult calibrateStereo(GeometricCalibrationData_Stereo&);

	cv::Mat test_calibration(const GeometricCalibrationTestData&);

	~GeometricCalibration();

	GeometricCalibration(const GeometricCalibration&) = delete;
	GeometricCalibration(GeometricCalibration&&) = delete;
	GeometricCalibration& operator=(GeometricCalibration&) = delete;
	GeometricCalibration& operator=(GeometricCalibration&&) = delete;

	std::vector<std::size_t> calculateDistanceToPlane(
		const open3d::geometry::PointCloud& points,
		const double inliers_distance,
		bool visualize = true
	);


	// Gives back for both the Primary Camera and Secondary Camera 
	// Dipslay Points (in dispaly Coordinates) that were found in both images
	StereoCalibrationPoints getCommonImagePointsfromStereoUnwrap(
		const std::vector<cv::Mat>& unwrap_prim,
		const std::vector<cv::Mat>& unwrap_secon,
		const cv::Mat& contrastMask_prim,
		const cv::Mat& contrastMask_secon,
		const double& wavelength
	);
	
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

	cv::Vec2d normalizePoints(
		const cv::Vec2d& p,
		const cv::Mat& K);

	void backToWorld(
		cv::Mat& rvec_disp2cam,
		cv::Mat& tvec_disp2cam,
		const cv::Mat& rvec_virtdisp2cam,
		const cv::Mat& tvec_virt2disp2cam,
		const cv::Mat& cam_mirror_rvec,
		const cv::Mat& cam_mirror_trans,
		const cv::Mat& mir2cam_rvec,
		const cv::Mat& mir2cam_tvec);

	Eigen::Vector4d fitPlane(
		const open3d::geometry::PointCloud&,
		double distance_threshold
	);

	void savePointsToCSV(const std::string& filename,
		const std::vector<cv::Point3f>& pts)
	{
		std::ofstream file(filename);
		if (!file.is_open()) {
			throw std::runtime_error("Could not open file");
		}

		file << "x,y,z\n";

		for (const auto& p : pts) {
			file << p.x << "," << p.y << "," << p.z << "\n";
		}
	}
	
};




#endif