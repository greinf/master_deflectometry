#include "GeometricCalibration.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "imgProcessing.hpp"
#include "cmath"


struct GeometricCalibration::Impl {
private:
	struct Cam2Mirror {
		cv::Mat rvec{};
		cv::Mat tvec{};
	};
	struct VirtDisp2Cam {
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
	Cam2Mirror cam2mirror{};
	VirtDisp2Cam virDisp2cam{};
};




GeometricCalibrationResult GeometricCalibration::calibrate(
	const GeometricCalibrationData& data)
{
	if (!data.validData()) throw std::invalid_argument("Dataholder is not valid for Calibration");

	m_impl->mask = m_img_processing.createMask(*data.contrast, 0.3, false);

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

	m_impl->cam2mirror.rvec = cv::Mat(3, 1, CV_64F, cv::Scalar(0));
	m_impl->cam2mirror.tvec = cv::Mat(3, 1, CV_64F);

	m_impl->cam2mirror.tvec.at<double>(0, 0) = -1000.0;
	m_impl->cam2mirror.tvec.at<double>(1, 0) = -1000.0;
	m_impl->cam2mirror.tvec.at<double>(2, 0) = 4000.0;

	/*if (!solvePnP) {
		std::cout << "SolvePnP failed for calculating the mirror pose \n";
		return{};
	}*/

	// Calculates effectively Virtual Dispaly to Camera,
	// since the world coordinate system
	// is in the mirror. 
	cv::Mat virt2cam_rvec;
	cv::Mat virt2cam_tvec;
	bool solvePnP = cv::solvePnP(
		objectPointsDisp,
		imagePointsDisp,
		data.camMat,
		data.distCoeffs,
		virt2cam_rvec,
		virt2cam_tvec,
		false,
		cv::SOLVEPNP_ITERATIVE);

	if (!solvePnP) {
		std::cout << "SolvePnP failed for calculating the virtual dispaly pose \n";
		return{};
	}

	cv::Mat H;

	GeometricCalibrationResult result{};

	result.cam2mir_rvec = m_impl->cam2mirror.rvec;
	result.cam2mir_tvec = m_impl->cam2mirror.tvec;

	m_img_processing.backToWorld(
		result.disp2cam_rvec,
		result.disp2cam_tvec,
		virt2cam_rvec,
		virt2cam_tvec,
		m_impl->cam2mirror.rvec,
		m_impl->cam2mirror.tvec);

	return result;
}

GeometricCalibration::GeometricCalibration(const GeometricCalibrationConfig& config, ImageProcessing& process)
	:m_img_processing{ process }
{
	if (!config.validData()) throw std::invalid_argument("Config Data is not valid");
	m_impl = std::make_unique<Impl>(config);
}

cv::Mat GeometricCalibration::calculateSurfaceNormals(
	const cv::Mat& rays,
	const cv::Mat& rays_refelcted,
	const cv::Mat& mask)
{
	CV_Assert(rays.size() == rays_refelcted.size());
	CV_Assert(mask.size() == rays_refelcted.size());
	CV_Assert(rays.type() == CV_64FC3);
	CV_Assert(rays_refelcted.type() == CV_64FC3);
	CV_Assert(mask.type() == CV_8U);

	cv::Mat surface_normals(mask.size(), CV_64FC3, cv::Vec3d(0.0, 0.0, 0.0));

	cv::parallel_for_(cv::Range(0, mask.rows),
		[&](const cv::Range& range) {
			for (int row = range.start; row < range.end; ++row) {
				const cv::Vec3d* rays_ptr = rays.ptr<cv::Vec3d>(row);
				const cv::Vec3d* rays_reflected_ptr = rays_refelcted.ptr<cv::Vec3d>(row);
				const uchar* mask_ptr = mask.ptr<uchar>(row);
				cv::Vec3d* surface_normals_ptr = surface_normals.ptr<cv::Vec3d>(row);
				for (int col = 0; col < mask.cols; ++col) {
					if (mask_ptr[col] == 0) continue;
					if (std::isnan(cv::norm(rays_reflected_ptr[col]))) continue;
					if (std::isnan(cv::norm(rays_ptr[col]))) continue;
					if (rays_reflected_ptr[col][0] == -1.0 &&
						rays_reflected_ptr[col][1] == -1.0 &&
						rays_reflected_ptr[col][2] == -1.0) continue;
					if (rays_ptr[col][0] == -1.0 &&
						rays_ptr[col][1] == -1.0 &&
						rays_ptr[col][2] == -1.0) continue;

					cv::Vec3d ray = rays_ptr[col] / cv::norm(rays_ptr[col]);
					cv::Vec3d rays_reflected = rays_reflected_ptr[col] / cv::norm(rays_reflected_ptr[col]);

					cv::Vec3d surface_n = rays_reflected - ray;
					surface_n /= cv::norm(surface_n);
					surface_normals_ptr[col] = surface_n;
				}
			}
		});
	
	return surface_normals;
}

cv::Mat GeometricCalibration::generateCoordinateImage(
	const cv::Size& sz,
	const int dimension)
{
	CV_Assert(sz.area() > 0);
	CV_Assert(dimension == 2 || dimension == 3);

	cv::Mat coordinate_img(sz, CV_64FC2);

	cv::parallel_for_(cv::Range(0, sz.height),
		[&](const cv::Range& range) {
			for (int row = range.start; row < range.end; ++row) {
				cv::Vec2d* row_ptr = coordinate_img.ptr<cv::Vec2d>(row);
				for (int cols = 0; cols < sz.width; ++cols) {
					row_ptr[cols] = cv::Vec2d(cols, row);
				}

			}
		});

	if (dimension == 3) {
		cv::Mat dimension3 = cv::Mat::zeros(sz, CV_64F);
		cv::Mat in[2] = { coordinate_img, dimension3 };
		int fromTo[] = { 0,0 , 1,1 , 2,2 };
		cv::Mat out(sz, CV_64FC3);

		cv::mixChannels(in, 2, &out, 1, fromTo, 3);
		return out;
	}
	return coordinate_img;
}


cv::Mat GeometricCalibration::test_calibration(
	const GeometricCalibrationTestData& data)
{
	if (!data.validData()) throw std::invalid_argument("Invalid Data in GemoetricCalibrationTestData");
	
	m_impl->mask = m_img_processing.createMask(
		*data.contrast,
		0.3);

	cv::Size sz = m_impl->mask.size();

	cv::Mat coordiante_sensor =
		generateCoordinateImage(sz);

	cv::Mat rays = 
		m_img_processing.calulateRays(
			data.camMat, 
			data.distCoeffs, 
			coordiante_sensor);

	cv::Mat coordiante_mirror =
		generateCoordinateImage(
			m_impl->m_config.pattern_size,
			3);
		
	coordiante_mirror *= m_impl->m_config.point_dist;

	cv::Mat coordiante_mirror_rot =
		m_img_processing.rotateCoordinatedGrid(
			coordiante_mirror,
			data.cam2mir_rvec,
			Rotation::rodrigeuz);

	cv::Mat coordiante_mirror_shift =
		m_img_processing.shiftCoordinateGrid(
			coordiante_mirror_rot,
			data.cam2mir_tvec
		);

	std::vector<cv::Mat> hitpints_mirror = m_img_processing.calculateHitPoints(
		rays,
		coordiante_mirror_shift);
	
	cv::Mat unwrap_world =
		m_img_processing.convertUnwrapToWorldCoord(
			*(data.unwrap),
			m_impl->m_config.wavelength_phase,
			m_impl->m_config.displayPixelPitch);

	cv::Mat cam2disp_rvec = data.disp2cam_rvec * -1.0;

	cv::Mat R;

	cv::Rodrigues(cam2disp_rvec, R);

	cv::Mat cam2disp_tvec = -R * data.disp2cam_tvec;

	cv::Mat unwrap_world_rot =
		m_img_processing.rotateCoordinatedGrid(
			unwrap_world,
			cv::Vec3d(cam2disp_rvec),
			Rotation::rodrigeuz
		);

	cv::Mat unwrap_world_rot_shift =
		m_img_processing.shiftCoordinateGrid(
			unwrap_world_rot,
			data.disp2cam_tvec
		);

	cv::Mat reflected_rays = unwrap_world_rot_shift - hitpints_mirror[0];

	cv::Mat surface_normals = calculateSurfaceNormals(
		rays,
		reflected_rays,
		m_impl->mask
	);

	return surface_normals;
}


GeometricCalibration::~GeometricCalibration() = default;