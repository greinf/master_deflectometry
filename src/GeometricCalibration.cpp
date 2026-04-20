#include "GeometricCalibration.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "imgProcessing.hpp"


struct GeometricCalibration::Impl {
private:
	struct CamToMirror {
		cv::Mat rvec{};
		cv::Mat tvec{};
	};
	struct CamToVirtDisp {
		cv::Mat rvec{};
		cv::Mat tvec{};
	};

public:
	explicit Impl(const GeometricCalibrationConfig& config) 
		:m_config{config}{ }
	
	const GeometricCalibrationConfig& m_config;

	cv::Mat working_img;
	cv::Mat mask;
	cv::Size working_size;
	CamToMirror cam2mir{};
	CamToVirtDisp cam2virDisp{};
};


GeometricCalibrationResult GeometricCalibration::calibrate(
	const GeometricCalibrationData& data)
{
	if (!data.validData()) throw std::invalid_argument("Dataholder is not valid for Calibration");

	//m_impl->mask = m_img_processing.createMask(*data.contrast, 0.3, false);

	cv::Mat worker = (*data.contrast)[0];

	cv::Mat nanMask = (worker == worker);

	m_impl->mask = cv::Mat::ones(worker.size(), CV_8U);

	m_impl->mask.setTo(0, ~nanMask);

	m_impl->working_size = m_impl->mask.size();

	// Virtual Display Pose Estimation 
	// Through Unwrap Pictures. 
	std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>> calibPoints =
		m_img_processing.do_calibration_Points(
			*data.unwrap,
			m_impl->mask,
			m_impl->m_config.wavelength_phase,
			1,
			1,
			m_impl->m_config.displayPixelPitch);

	// Debugging 
	//
	//cv::Mat mean = m_img_processing.mean(*data.biasIntensity);

	//// Mirror Pose estimation 
	//// imagePoints - Circles in the images position
	//std::vector<cv::Vec2d> imagePoints =
	//	m_img_processing.getCircleCoordinates(
	//		mean,
	//		m_impl->mask,
	//		m_impl->m_config.pattern_size);

	//// Object Points - Created in z = 0
	//std::vector<cv::Vec3d> patternObjectPoints =
	//	m_img_processing.createCalibPatternObjectPoints(
	//		m_impl->m_config.pattern_size,
	//		m_impl->m_config.point_dist
	//	);
	
	// Mirror Points 
	/*std::vector<cv::Point3d> object;
	std::vector<cv::Point2d> image;*/

	// Virtual Display Points 
	std::vector<cv::Point3d> objectPointsDisp;
	std::vector<cv::Point2d> imagePointsDisp;

	/*CV_Assert(patternObjectPoints.size() == imagePoints.size());

	for (std::size_t i = 0; i < patternObjectPoints.size(); ++i) {
		object.push_back(cv::Point3d(patternObjectPoints[i]));
		image.push_back(cv::Point2d(imagePoints[i]));
	}*/

	for (std::size_t i = 0; i < calibPoints.first.size(); ++i) {
		objectPointsDisp.push_back(cv::Point3d(calibPoints.second[i]));
		imagePointsDisp.push_back(cv::Point2d(calibPoints.first[i]));
	}

	// Calculates effectively Mirror to Camera, since the world coordinate system
	// is in the mirror.  
	/*bool solvePnP = cv::solvePnP(
		object,
		image,
		data.camMat,
		data.distCoeffs,
		m_impl->cam2mir.rvec,
		m_impl->cam2mir.tvec,
		false,
		cv::SOLVEPNP_ITERATIVE);*/

	m_impl->cam2mir.rvec = cv::Mat(3, 1, CV_64F, cv::Scalar(0));
	m_impl->cam2mir.tvec = cv::Mat(3, 1, CV_64F);

	m_impl->cam2mir.tvec.at<double>(0, 0) = 100.0;
	m_impl->cam2mir.tvec.at<double>(1, 0) = 100.0;
	m_impl->cam2mir.tvec.at<double>(2, 0) = -4000.0;


	/*if (!solvePnP) {
		std::cout << "SolvePnP failed for calculating the mirror pose \n";
		return{};
	}*/

	// Calculates effectively Virtual Dispaly to Camera,
	// since the world coordinate system
	// is in the mirror. 
	bool solvePnP = cv::solvePnP(
		objectPointsDisp,
		imagePointsDisp,
		data.camMat,
		data.distCoeffs,
		m_impl->cam2virDisp.rvec,
		m_impl->cam2virDisp.tvec,
		false,
		cv::SOLVEPNP_ITERATIVE);

	if (!solvePnP) {
		std::cout << "SolvePnP failed for calculating the virtual dispaly pose \n";
		return{};
	}

	cv::Mat H;

	GeometricCalibrationResult result{};

	m_img_processing.backToWorld(
		result.Rvec,
		result.tvec,
		m_impl->cam2virDisp.rvec,
		m_impl->cam2virDisp.tvec,
		m_impl->cam2mir.rvec,
		m_impl->cam2mir.tvec);


	return result;
}

GeometricCalibration::GeometricCalibration(const GeometricCalibrationConfig& config, ImageProcessing& process)
	:m_img_processing{ process }
{
	if (!config.validData()) throw std::runtime_error("Config Data is not valid");
	m_impl = std::make_unique<Impl>(config);
}

GeometricCalibration::~GeometricCalibration() = default;