#include "GeometricCalibration.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "imgProcessing.hpp"
#include <open3d/pipelines/registration/TransformationEstimation.h>
#include <open3d/geometry/PointCloud.h>
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


auto createaffine = [](const cv::Mat& r_vec,
	const cv::Mat& tvec,
	cv::Matx44d& out)
	{
		cv::Mat R;
		cv::Rodrigues(r_vec, R);

		// 2. Die 4x4 Matrix initialisieren
		out = cv::Matx44d::eye(); // Erzeugt Einheitsmatrix (unten steht schon 0,0,0,1)

		cv::Mat R_d, t_d;
		R.convertTo(R_d, CV_64F);
		tvec.convertTo(t_d, CV_64F);

		// 3. R und tvec in T kopieren
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				out(i, j) = R.at<double>(i, j);
			}
			out(i, 3) = tvec.at<double>(i, 0);
		}
	};


GeometricCalibrationResult GeometricCalibration::calibrateStereo(
	const GeometricCalibrationData_Stereo& data)
{
	if (!data.validData()) throw std::invalid_argument("Dataholder is not valid for geometric Stereo Calibration");
	m_impl->mask = data.mask;
	m_impl->working_size = m_impl->mask.size();

	cv::Mat biasIntensityPrimary, biasIntensitySecondary,
		biasIntensityPrimary8U, biasIntensitySecondary8U;

	biasIntensityPrimary = m_img_processing.mean(*data.biasIntensity);
	biasIntensitySecondary = m_img_processing.mean(*data.biasIntensity_sec_cam);
	cv::normalize(biasIntensityPrimary, biasIntensityPrimary8U, 0, 255.0, cv::NORM_MINMAX, CV_8U);
	cv::normalize(biasIntensitySecondary, biasIntensitySecondary8U, 0, 255.0, cv::NORM_MINMAX, CV_8U);

	std::vector<cv::Vec2d> circleImgCoordPrimary, circleImgCoordSecondary,
		circleImgCoordPrimary_undist, circleImgCoordSecondary_undist;
	std::vector<cv::Point2f> circleImgCoordPrimary_p, circleImgCoordSecondary_p;
	circleImgCoordPrimary = m_img_processing.getCircleCoordinates(
		biasIntensityPrimary8U,
		m_impl->mask,
		m_impl->working_size);
	circleImgCoordSecondary = m_img_processing.getCircleCoordinates(
		biasIntensitySecondary8U,
		m_impl->mask,
		m_impl->working_size);

	for (const auto& imgCoordinate : circleImgCoordPrimary) {
		circleImgCoordPrimary_undist.push_back(m_img_processing.undistortImagePts(
			imgCoordinate, data.camMat, data.distCoeffs
		));
	}

	for (const auto& imgCoordiante : circleImgCoordSecondary) {
		circleImgCoordSecondary_undist.push_back(m_img_processing.undistortImagePts(
			imgCoordiante, data.camMat_secundaryCam, data.distCoeffs_secondaryCam
		));
	}

	CV_Assert(circleImgCoordPrimary_undist.size() == circleImgCoordSecondary_undist.size());

	cv::Mat projectionMatrixPrim, projectionMatrixSeco;

	for (std::size_t i = 0; i < circleImgCoordPrimary.size(); ++i) {
		circleImgCoordPrimary_p.push_back(cv::Point2f(circleImgCoordPrimary_undist[i]));
		circleImgCoordSecondary_p.push_back(cv::Point2f(circleImgCoordSecondary_undist[i]));
	}

	cv::stereoRectify(
		data.camMat,
		data.distCoeffs,
		data.camMat_secundaryCam,
		data.distCoeffs_secondaryCam,
		m_impl->working_size,
		data.cam2cam_rotMat,
		data.cam2cam_tvec,
		cv::Mat(),
		cv::Mat(),
		projectionMatrixPrim,
		projectionMatrixSeco,
		cv::Mat()
	);

	cv::Mat homogeneousPoints;

	cv::triangulatePoints(
		projectionMatrixPrim,
		projectionMatrixSeco,
		circleImgCoordPrimary_p,
		circleImgCoordSecondary_p,
		homogeneousPoints
	);

	std::vector<cv::Point3d> pts3D;

	// Back to 3d
	for (int i = 0; i < homogeneousPoints.cols; ++i) {
		double x = homogeneousPoints.at<double>(0, i);
		double y = homogeneousPoints.at<double>(1, i);
		double z = homogeneousPoints.at<double>(2, i);
		double w = homogeneousPoints.at<double>(3, i);

		pts3D.emplace_back(x / w, y / w, z / w);
	}

	// Object Points - Created in z = 0
	std::vector<cv::Vec3d> patternObjectPoints =
		m_img_processing.createCalibPatternObjectPoints(
			m_impl->m_config.pattern_size,
			m_impl->m_config.point_dist
		);

	CV_Assert(patternObjectPoints.size() == pts3D.size());

	std::vector<Eigen::Vector3d> triangulated_pts, pattern_pts;
	std::vector<Eigen::Vector2i> correspondence;

	for (std::size_t i = 0; i < patternObjectPoints.size(); ++i) {
		triangulated_pts.push_back(Eigen::Vector3d(pts3D[i].x, pts3D[i].y, pts3D[i].z));
		pattern_pts.push_back(Eigen::Vector3d(
			patternObjectPoints[i].val[0],
			patternObjectPoints[i].val[1],
			patternObjectPoints[i].val[2]));
		correspondence.push_back(Eigen::Vector2i(i, i));
	}

	open3d::geometry::PointCloud triangulated(triangulated_pts);
	open3d::geometry::PointCloud pattern(pattern_pts);

	open3d::pipelines::registration::TransformationEstimationPointToPoint poseEstimation(false);

	Eigen::Matrix4d transform = poseEstimation.ComputeTransformation(
		pattern,
		triangulated,
		correspondence);

	cv::Mat cam2mir_Rot(3, 3, CV_64F), mir2cam_Rot(3, 3, CV_64F), mir2cam_rvec(3, 3, CV_64F);
	cv::Mat cam2mir_trans(3, 1, CV_64F), mir2cam_tvec(3, 1, CV_64F);

	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			cam2mir_Rot.at<double>(i, j) = transform(i, j);

	cam2mir_trans.at<double>(0, 0) = transform(0, 3);
	cam2mir_trans.at<double>(1, 0) = transform(1, 3);
	cam2mir_trans.at<double>(2, 0) = transform(2, 3);

	mir2cam_Rot = cam2mir_Rot.t();
	mir2cam_tvec = -cam2mir_Rot * cam2mir_trans;
	cv::Rodrigues(mir2cam_Rot, mir2cam_rvec);
	
	std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>> calibPoints =
		m_img_processing.do_calibration_Points(
			*data.unwrap,
			m_impl->mask,
			m_impl->m_config.wavelength_phase,
			1,
			1,
			m_impl->m_config.displayPixelPitch);

	// Virtual Display Points 
	std::vector<cv::Point3d> objectPointsDisp;
	std::vector<cv::Point2d> imagePointsDisp;

	for (std::size_t i = 0; i < calibPoints.first.size(); ++i) {
		objectPointsDisp.push_back(cv::Point3d(calibPoints.second[i]));
		imagePointsDisp.push_back(cv::Point2d(calibPoints.first[i]));
	}

	cv::Mat virt2cam_rvec, virt2cam_tvec;

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
	GeometricCalibrationResult result{};
	cv::Rodrigues(cam2mir_Rot, result.cam2mir_rvec);
	result.cam2mir_tvec = cam2mir_trans;

	backToWorld(
		result.disp2cam_rvec,
		result.disp2cam_tvec,
		virt2cam_rvec,
		virt2cam_tvec,
		result.cam2mir_rvec,
		result.cam2mir_tvec,
		mir2cam_rvec,
		mir2cam_tvec);

	return result;
}

GeometricCalibrationResult GeometricCalibration::calibrateMono(
	const GeometricCalibrationData& data)
{
	if (!data.validData()) throw std::invalid_argument("Dataholder is not valid for Calibration");
	m_impl->mask = data.mask;
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

	cv::Mat mean = m_img_processing.mean(*data.biasIntensity);

	cv::Mat biasIntensity;

	cv::normalize(mean, biasIntensity, 0, 255, cv::NORM_MINMAX, CV_8U);

	// Mirror Pose estimation 
	// imagePoints - Circles in the images position
	std::vector<cv::Vec2d> imagePoints =
		m_img_processing.getCircleCoordinates(
			biasIntensity,
			m_impl->mask,
			m_impl->m_config.pattern_size);

	// Object Points - Created in z = 0
	std::vector<cv::Vec3d> patternObjectPoints =
		m_img_processing.createCalibPatternObjectPoints(
			m_impl->m_config.pattern_size,
			m_impl->m_config.point_dist
		);
	
	// Mirror Points 
	std::vector<cv::Point3d> object;
	std::vector<cv::Point2d> image;

	// Virtual Display Points 
	std::vector<cv::Point3d> objectPointsDisp;
	std::vector<cv::Point2d> imagePointsDisp;

	CV_Assert(patternObjectPoints.size() == imagePoints.size());

	for (std::size_t i = 0; i < patternObjectPoints.size(); ++i) {
		object.push_back(cv::Point3d(patternObjectPoints[i]));
		image.push_back(cv::Point2d(imagePoints[i]));
	}

	for (std::size_t i = 0; i < calibPoints.first.size(); ++i) {
		objectPointsDisp.push_back(cv::Point3d(calibPoints.second[i]));
		imagePointsDisp.push_back(cv::Point2d(calibPoints.first[i]));
	}
	 
	std::vector<cv::Mat> mir2cam_allSolutions_rvec, mir2cam_allSolutions_tvec;
	bool solvePnP = cv::solvePnPGeneric(
		object,
		image,
		data.camMat,
		data.distCoeffs,
		mir2cam_allSolutions_rvec,
		mir2cam_allSolutions_tvec,
		false,
		cv::SOLVEPNP_IPPE);

	if (!solvePnP || mir2cam_allSolutions_rvec.empty()) {
		std::cout << "SolvePnP failed for calculating the mirror pose\n";
		return {};
	}

	cv::Mat mir2cam_rvec, mir2cam_tvec;

	for (std::size_t i = 0; i < mir2cam_allSolutions_rvec.size(); ++i) {
		// It is assumed that there is a Mirror Coordiante System with nearly no Rotation
		// with respect to Cammera coordinate System
		if (cv::norm(mir2cam_allSolutions_rvec[i]) < CV_PI) {
			mir2cam_rvec = mir2cam_allSolutions_rvec[i].clone();
			mir2cam_tvec = mir2cam_allSolutions_tvec[i].clone();
			break;
		}
	}
	
	cv::solvePnPRefineLM(
		object,
		image,
		data.camMat,
		data.distCoeffs,
		mir2cam_rvec,
		mir2cam_tvec
	);

	cv::Mat mir2cam_R;
	cv::Rodrigues(mir2cam_rvec, mir2cam_R);

	cv::Mat cam2mir_R = mir2cam_R.t();
	cv::Mat cam2mir_t = -cam2mir_R * mir2cam_tvec;

	m_impl->cam2mirror.tvec = cam2mir_t;
	cv::Rodrigues(cam2mir_R, m_impl->cam2mirror.rvec);

	cv::Mat virt2cam_rvec = (cv::Mat_<double>(3, 1) << 0.0, -CV_PI, 0.0);
	cv::Mat virt2cam_tvec = mir2cam_tvec * 2.0;

	/*m_impl->cam2mirror.rvec = cv::Mat(3, 1, CV_64F, cv::Scalar(0));
	m_impl->cam2mirror.tvec = cv::Mat(3, 1, CV_64F);

	m_impl->cam2mirror.tvec.at<double>(0, 0) = -1000.0;
	m_impl->cam2mirror.tvec.at<double>(1, 0) = -1000.0;
	m_impl->cam2mirror.tvec.at<double>(2, 0) = 4000.0;*/

	// Calculates effectively Virtual Dispaly to Camera,
	// since the world coordinate system
	// is in the mirror. 
	solvePnP = cv::solvePnP(
		objectPointsDisp,
		imagePointsDisp,
		data.camMat,
		data.distCoeffs,
		virt2cam_rvec,
		virt2cam_tvec,
		true,
		cv::SOLVEPNP_ITERATIVE);

	if (!solvePnP) {
		std::cout << "SolvePnP failed for calculating the virtual dispaly pose \n";
		return{};
	}

	GeometricCalibrationResult result{};

	result.cam2mir_rvec = m_impl->cam2mirror.rvec;
	result.cam2mir_tvec = m_impl->cam2mirror.tvec;

	backToWorld(
		result.disp2cam_rvec,
		result.disp2cam_tvec,
		virt2cam_rvec,
		virt2cam_tvec,
		m_impl->cam2mirror.rvec,
		m_impl->cam2mirror.tvec,
		mir2cam_rvec,
		mir2cam_tvec);

	return result;
}

void GeometricCalibration::backToWorld(
		cv::Mat& rvec_w,
		cv::Mat& tvec_w,
		const cv::Mat& vdisp2cam_rvec,  // Vitual dispaly -> cam
		const cv::Mat& vdisp2cam_tvec,  // vitual dipslay -> cam 
		const cv::Mat& cam2mirror_rvec, // cam -> mirr
		const cv::Mat& cam2mirror_tvec,
		const cv::Mat& mir2cam_rvec,
		const cv::Mat& mir2cam_tvec) // Mirror -> cam
{
	cv::Matx44d virtualdisp2cam, observer;

	// Create Transform from VirtualDispaly to Camera
	createaffine(vdisp2cam_rvec, vdisp2cam_tvec, virtualdisp2cam);

	observer = virtualdisp2cam;

	// The Affine transformation from Camera to Mirror
	cv::Matx44d affine_cam_to_mirr;
	createaffine(cam2mirror_rvec, cam2mirror_tvec, affine_cam_to_mirr);

	// Transformation VirtualDispCoord -> CameraCoord -> MirrorCoord
	observer = virtualdisp2cam * affine_cam_to_mirr;

	// In the mirror coordiante System apply the mirroring
	cv::Mat M = cv::Mat::eye(4, 4, CV_64F);

	M.at<double>(2, 2) = -1.0;

	cv::Matx44d householder(M);

	// Transformation: VirtualdispalyCoord -> CameraCoord -> MirrorCoord -> Apply MirrorMatrix 
	// oberserver is no left handed !!!! 
	observer = virtualdisp2cam * affine_cam_to_mirr * householder;

	cv::Matx44d affine_mirror2cam;

	createaffine(mir2cam_rvec, cam2mirror_tvec, affine_mirror2cam);

	// The Transformation to the real coordiante System
	observer = virtualdisp2cam * affine_cam_to_mirr * householder * affine_mirror2cam;

	cv::Vec3d x_axis(observer(0, 0), observer(1, 0), observer(2, 0));
	x_axis /= cv::norm(x_axis);

	cv::Vec3d y_axis(observer(0, 1), observer(1, 1), observer(2, 1));
	y_axis /= cv::norm(y_axis);

	cv::Vec3d z_axis_new = x_axis.cross(y_axis);
	z_axis_new /= cv::norm(z_axis_new);

	// rebuild y so that the frame is orthonormal
	y_axis = z_axis_new.cross(x_axis);
	y_axis /= cv::norm(y_axis);

	cv::Mat R = (cv::Mat_<double>(3, 3) <<
		x_axis[0], y_axis[0], z_axis_new[0],
		x_axis[1], y_axis[1], z_axis_new[1],
		x_axis[2], y_axis[2], z_axis_new[2]);

	cv::Rodrigues(R, rvec_w);
	tvec_w = (cv::Mat_<double>(3, 1) <<
		observer(0, 3), observer(1, 3), observer(2, 3));
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