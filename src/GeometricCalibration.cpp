#include "GeometricCalibration.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "imgProcessing.hpp"
#include <open3d/pipelines/registration/TransformationEstimation.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/visualization/utility/DrawGeometry.h>
#include "cmath"
#include <tuple>
#include <Eigen/Dense>

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
				out(i, j) = R_d.at<double>(i, j);
			}
			out(i, 3) = t_d.at<double>(i, 0);
		}
	};


GeometricCalibrationResult GeometricCalibration::calibrateStereo(
	GeometricCalibrationData_Stereo& data)
{
	if (!data.validData()) throw std::invalid_argument("Dataholder is not valid for geometric Stereo Calibration");

	std::cout << "StereoCalibration start " << std::endl;

	m_impl->mask = data.mask;
	m_impl->working_size = m_impl->mask.size();

	cv::Mat mask_secondary = m_img_processing.createMask(*data.contrast_sec_cam, 0.4);

	cv::Mat biasIntensityPrimary, biasIntensitySecondary,
		contrastIntensityPrimary, contrastIntensitySecondary,
		amplitudePrimary, amplitudeSecondary,
		IntensityPrimary8U, IntensitySecondary8U, homogenuosPoint;

	biasIntensityPrimary = m_img_processing.mean(*data.biasIntensity);
	biasIntensitySecondary = m_img_processing.mean(*data.biasIntensity_sec_cam);
	contrastIntensityPrimary = m_img_processing.mean(*data.contrast);
	contrastIntensitySecondary = m_img_processing.mean(*data.contrast_sec_cam);

	cv::multiply(biasIntensityPrimary, contrastIntensityPrimary, amplitudePrimary);
	cv::multiply(biasIntensitySecondary, contrastIntensitySecondary, amplitudeSecondary);

	cv::normalize(biasIntensityPrimary, IntensityPrimary8U, 0, 255.0, cv::NORM_MINMAX, CV_8U);
	cv::normalize(biasIntensitySecondary, IntensitySecondary8U, 0, 255.0, cv::NORM_MINMAX, CV_8U);

	std::vector<cv::Vec2d> circleImgCoordPrimary, circleImgCoordSecondary,
		circleImgCoordPrimaryUndist, circleImgCoordSecondaryUndist;

	std::vector<cv::Point2f> circleImgCoordPrimary_p, circleImgCoordSecondary_p;

	std::vector<cv::Point2f> circleImgCoordPrimary_sensorcoord, circleImgCoordSecondary_sensor_coord;

	circleImgCoordPrimary = m_img_processing.getCircleCoordinates(
		IntensityPrimary8U,
		m_impl->mask,
		m_impl->m_config.pattern_size,
		data.path + "/primary.png");
	circleImgCoordSecondary = m_img_processing.getCircleCoordinates(
		IntensitySecondary8U,
		mask_secondary,
		m_impl->m_config.pattern_size,
		data.path + "/secondary.png");

	cv::TermCriteria criteria(cv::TermCriteria::EPS, 0, 1e-9);

	// The image Points get undistorted
	for (auto& vec : circleImgCoordPrimary) {

		circleImgCoordPrimary_sensorcoord.push_back(
			cv::Point2f(static_cast<float>(vec[0]),
				static_cast<float>(vec[1])));
		circleImgCoordPrimaryUndist.push_back(
			m_img_processing.undistortImagePts(
				vec, 
				data.camMat, 
				data.distCoeffs));
	}
	
	for (auto& vec : circleImgCoordSecondary) {
		circleImgCoordSecondary_sensor_coord.push_back(
			cv::Point2f(static_cast<float>(vec[0]),
				static_cast<float>(vec[1])));
		circleImgCoordSecondaryUndist.push_back(
			m_img_processing.undistortImagePts(
				vec, 
				data.camMat_secundaryCam, 
				data.distCoeffs_secondaryCam));
	}

	// Normalize to CameraCoordinates (not Sensor)
	for (auto& vec : circleImgCoordPrimaryUndist) {
		vec = normalizePoints(vec, data.camMat);
	}
	for (auto& vec : circleImgCoordSecondaryUndist) {
		vec = normalizePoints(vec, data.camMat_secundaryCam);
	}
	// From cv::Vec2d -> cv::Point2d (needed for all subsequent operation)
	for (const auto& p : circleImgCoordPrimaryUndist)
		circleImgCoordPrimary_p.emplace_back(
			static_cast<float>(p[0]),
			static_cast<float>(p[1])
		);
	for (const auto& p : circleImgCoordSecondaryUndist)
		circleImgCoordSecondary_p.emplace_back(
			static_cast<float>(p[0]),
			static_cast<float>(p[1])
		);
	
	// Projektion Matrixes between the two cameras. 
	cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
	cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
	data.cam2cam_rotMat.copyTo(P2(cv::Rect(0, 0, 3, 3)));
	data.cam2cam_tvec.copyTo(P2(cv::Rect(3, 0, 1, 3)));

	// Triangulate the Points from both cameras
	cv::triangulatePoints(
		P1,
		P2,
		circleImgCoordPrimary_p,
		circleImgCoordSecondary_p,
		homogenuosPoint
	);
	
	std::vector<cv::Point3d> pts3D;

	// Back to 3d
	homogenuosPoint.convertTo(homogenuosPoint, CV_64F);


	for (int i = 0; i < homogenuosPoint.cols; ++i) {
		double x = homogenuosPoint.at<double>(0, i);
		double y = homogenuosPoint.at<double>(1, i);
		double z = homogenuosPoint.at<double>(2, i);
		double w = homogenuosPoint.at<double>(3, i);

		pts3D.emplace_back(x / w, y / w, z / w);
	}

	savePointsToCSV(data.path + "/triangulatedPts.csv", pts3D);

	double errNorm1 = 0.0;
	double errNorm2 = 0.0;


	// Be Carefull ------- Not sure if one undistort Step is missing !!!!!!!
	// (only for the validation off the triangulation)

	// Here we check if a projection back onto the sensor is "good eough"
	// We have the triantulated point in Camera1 coordiantes. 
	// When x/z, y/z, z/z -> back into normalized Camera Coordiantes 
	// x1_hat is in normalized Camera1Coordinates. and undistorted normalized Camera Coordinates
	// From the real meassurment are calcualted above in circieImgCoordPrimary_p. subtrakt and get error
	for (size_t i = 0; i < pts3D.size(); ++i)
	{
		cv::Mat X1 = (cv::Mat_<double>(3, 1) <<
			pts3D[i].x,
			pts3D[i].y,
			pts3D[i].z
			);

		// Cam1 normalized reprojection
		cv::Point2d x1_hat(
			X1.at<double>(0) / X1.at<double>(2),
			X1.at<double>(1) / X1.at<double>(2)
		);

		// Transform into cam2
		cv::Mat X2 = data.cam2cam_rotMat * X1 + data.cam2cam_tvec;

		cv::Point2d x2_hat(
			X2.at<double>(0) / X2.at<double>(2),
			X2.at<double>(1) / X2.at<double>(2)
		);
		
		cv::Point2d x1_meas(circleImgCoordPrimary_p[i].x, circleImgCoordPrimary_p[i].y);
		cv::Point2d x2_meas(circleImgCoordSecondary_p[i].x, circleImgCoordSecondary_p[i].y);

		errNorm1 += cv::norm(x1_hat - x1_meas);
		errNorm2 += cv::norm(x2_hat - x2_meas);
	}

	errNorm1 /= pts3D.size();
	errNorm2 /= pts3D.size();

	std::cout << "Normalized reproj error cam1: " << errNorm1 << "\n";
	std::cout << "Normalized reproj error cam2: " << errNorm2 << "\n";

	std::vector<cv::Point3f> pts3Dfloat;
	for (const auto& point : pts3D) {
		pts3Dfloat.push_back(cv::Vec3f(point.x, point.y, point.z));
	}

	// Here we onyl do a reprojection the sensor coordiante system.
	// It is meassure if the triangulated world points land get, back projected
	// on the normalized & undistorted points on the camera sensor.
	std::vector<cv::Point2f> reproj1;

	cv::projectPoints(
		pts3Dfloat,
		cv::Vec3d(0, 0, 0),                // Kamera 1 = Welt
		cv::Vec3d(0, 0, 0),
		data.camMat,
		data.distCoeffs,
		reproj1
	);

	cv::Mat rvec12;
	cv::Rodrigues(data.cam2cam_rotMat, rvec12);

	std::vector<cv::Point2f> reproj2;

	cv::projectPoints(
		pts3Dfloat,
		rvec12,
		data.cam2cam_tvec,
		data.camMat_secundaryCam,
		data.distCoeffs_secondaryCam,
		reproj2
	);

	double sum = 0.0, sumSq = 0.0;
	double maxErr = 0.0;
	size_t maxIdx = 0;

	for (size_t i = 0; i < pts3D.size(); ++i)
	{
		double e = cv::norm(reproj1[i] - circleImgCoordPrimary_sensorcoord[i]);
		sum += e;
		sumSq += e * e;

		if (e > maxErr) {
			maxErr = e;
			maxIdx = i;
		}
	}

	std::cout << "mean err cam1: " << sum / pts3D.size() << "\n";
	std::cout << "rms err cam1: " << std::sqrt(sumSq / pts3D.size()) << "\n";
	std::cout << "max err cam1: " << maxErr << " at index " << maxIdx << "\n";

	// At this point it would be nice to validate the found points. Also calcualte the standard deviation. 
	// if point are more Error than standard deviation cut them.
	// Not implmented. 
	
	std::vector<cv::Vec3d> patternObjectPoints =
		m_img_processing.createCalibPatternObjectPoints(
			m_impl->m_config.pattern_size,
			m_impl->m_config.point_dist
		);

	CV_Assert(patternObjectPoints.size() == pts3D.size());

	std::vector<Eigen::Vector3d> triangulated_pts, pattern_pts;
	open3d::pipelines::registration::CorrespondenceSet correspondence;
	correspondence.reserve(patternObjectPoints.size());

	for (std::size_t i = 0; i < patternObjectPoints.size(); ++i) {
		triangulated_pts.push_back(Eigen::Vector3d(pts3D[i].x, pts3D[i].y, pts3D[i].z));
		pattern_pts.push_back(Eigen::Vector3d(
			patternObjectPoints[i].val[0],
			patternObjectPoints[i].val[1],
			patternObjectPoints[i].val[2]));
		correspondence.emplace_back(
			static_cast<int>(i),
			static_cast<int>(i));
	}

	open3d::geometry::PointCloud triangulated(triangulated_pts);
	open3d::geometry::PointCloud pattern(pattern_pts);

	calculateDistanceToPlane(triangulated, 10, false);

	// Finds transformation from pattern to Triangulated Points (in Rectified System!!!)
	// cam -> mir
	open3d::pipelines::registration::TransformationEstimationPointToPoint poseEstimation(false);

	// Calculates x_triang = transfrom * x_pattern
	Eigen::Matrix4d transform = poseEstimation.ComputeTransformation(
		pattern,
		triangulated,
		correspondence);

	std::cout << "Transformation Mirr2cam: \n" << 
		transform << std::endl;

	cv::Mat cam2mir_Rot(3, 3, CV_64F), mir2cam_Rot(3, 3, CV_64F), mir2cam_rvec, cam2mir_rvec;
	cv::Mat cam2mir_tvec(3, 1, CV_64F), mir2cam_tvec(3, 1, CV_64F);

	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			cam2mir_Rot.at<double>(i, j) = transform(i, j);

	cam2mir_tvec.at<double>(0, 0) = transform(0, 3);
	cam2mir_tvec.at<double>(1, 0) = transform(1, 3);
	cam2mir_tvec.at<double>(2, 0) = transform(2, 3);


	mir2cam_Rot = cam2mir_Rot.t();
	mir2cam_tvec = -mir2cam_Rot * cam2mir_tvec;
	cv::Rodrigues(mir2cam_Rot, mir2cam_rvec);
	cv::Rodrigues(cam2mir_Rot, cam2mir_rvec);
	
	std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>> calibPoints =
		m_img_processing.do_calibration_Points(
			*data.unwrap,
			m_impl->mask,
			m_impl->m_config.wavelength_phase,
			1,
			1,
			m_impl->m_config.displayPixelPitch,
			false,
			false);

	// Virtual Display Points 
	std::vector<cv::Point3d> objectPointsDisp;
	std::vector<cv::Point2d> imagePointsDisp;

	for (std::size_t i = 0; i < calibPoints.first.size(); ++i) {
		objectPointsDisp.push_back(cv::Point3d(calibPoints.second[i]));
		imagePointsDisp.push_back(cv::Point2d(calibPoints.first[i]));
	}

	cv::Mat virt2cam_rvec, virt2cam_tvec;

	// virt2cam_rvec: rotation of display frame expressed in camera frame
	// virt2cam_tvec: position of display origin expressed in camera frame
	// x_cam = virt2cam_rvec(as matrix)*x_disp + virt2cam_tvec
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
	
	cv::Mat pattern_rot;
	cv::Rodrigues(virt2cam_rvec, pattern_rot); // How is pattern orientated from camera

	cv::Mat virt2cam_tvec_n = -pattern_rot.t() * virt2cam_tvec; // Walk from virt to cam
	cv::Mat virt2cam_rvec_n;
	cv::Rodrigues(pattern_rot.t(), virt2cam_rvec_n);

	result.cam2mir_rvec = cam2mir_rvec;
	result.cam2mir_tvec = cam2mir_tvec;

	//std::cout << "Virt2cam Rvec: \n" << virt2cam_rvec << std::endl;
	//std::cout << "Virt2cam Tvec: \n" << virt2cam_tvec << std::endl;

	backToWorld(
		result.disp2cam_rvec,
		result.disp2cam_tvec,
		virt2cam_rvec,
		virt2cam_tvec,
		mir2cam_rvec,
		mir2cam_tvec,
		result.cam2mir_rvec,
		result.cam2mir_tvec);

	//std::cout << "Cam2mir rvec: \n" << cam2mir_rvec << std::endl;
	//std::cout << "Cam2mir tvec \n " << cam2mir_tvec << std::endl;

	return result;
}

void GeometricCalibration::calculateDistanceToPlane(
	const open3d::geometry::PointCloud& points,
	const double inliers_distance,
	bool visualize)
{
	Eigen::Vector4d surfacePara = fitPlane(points, inliers_distance);

	std::vector<double> distances;
	distances.reserve(points.points_.size());

	Eigen::Vector3d n(surfacePara(0), surfacePara(1), surfacePara(2));

	double n_norm = n.norm();

	for (const auto& p : points.points_) {
		double dist = std::abs(surfacePara(0) * p.x() + surfacePara(1) * p.y()
			+ surfacePara(2) * p.z() + surfacePara(3)) / n_norm;
		distances.push_back(dist);
	}

	double sum = 0.0;
	double max_dist = 0.0;

	for (double dist : distances) {
		sum += dist;
		max_dist = std::max(max_dist, dist);
	}

	double mean_dist = sum / distances.size();

	std::cout << "Mean distance to plane: " << mean_dist << "\n";
	std::cout << "Max distance to plane: " << max_dist << "\n";

	if (visualize) {
		Eigen::Vector3d p0 = -surfacePara(3) * n / n.squaredNorm();
		// zwei orthogonale Richtungen erzeugen
		Eigen::Vector3d v1 = n.unitOrthogonal();
		Eigen::Vector3d v2 = n.cross(v1);

		double size = 400.0;

		std::vector<Eigen::Vector3d> vertices = {
			p0 + size * v1 + size * v2,
			p0 + size * v1 - size * v2,
			p0 - size * v1 - size * v2,
			p0 - size * v1 + size * v2
		};

		// Mesh erstellen
		auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();

		mesh->vertices_ = vertices;
		mesh->triangles_ = {
			Eigen::Vector3i(0,1,2),
			Eigen::Vector3i(0,2,3)
		};

		mesh->ComputeVertexNormals();
		mesh->PaintUniformColor(Eigen::Vector3d(0.8, 0.2, 0.2)); // rot

		auto coord = open3d::geometry::TriangleMesh::CreateCoordinateFrame(500.0);

		open3d::visualization::DrawGeometries({std::make_shared<open3d::geometry::PointCloud>(points), mesh, coord });
	}
}



cv::Vec2d GeometricCalibration::normalizePoints(
	const cv::Vec2d& p,
	const cv::Mat& cam_Mat)
{
	double fx = cam_Mat.at<double>(0, 0);
	double fy = cam_Mat.at<double>(1, 1);
	double cx = cam_Mat.at<double>(0, 2);
	double cy = cam_Mat.at<double>(1, 2);

	return cv::Point2d(
		(p[0] - cx) / fx,
		(p[1] - cy) / fy
	);
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

Eigen::Vector4d GeometricCalibration::fitPlane(
	const open3d::geometry::PointCloud& pts,
	double distance_threshold)
{
	int ransac_n = 200;
	int num_iterations = 100;

	std::tuple< Eigen::Vector4d, std::vector<size_t>> result
		= pts.SegmentPlane(distance_threshold, ransac_n, num_iterations);
	
	Eigen::Vector4d surface_para = std::get<0>(result);
	std::vector<std::size_t> inliers = std::get<1>(result);

	std::cout << inliers.size() << " are treated as inliers for surface calculation"
		<< std::endl;

	return surface_para;
}

void GeometricCalibration::backToWorld(
		cv::Mat& rvec_w,
		cv::Mat& tvec_w,
		const cv::Mat& vdisp2cam_rvec,  // Rotation_vdisp->cam
		const cv::Mat& vdisp2cam_tvec,  // Translation_vdisp->cam
		const cv::Mat& cam2mirror_rvec, // Rotatoin_camera->mirror
		const cv::Mat& cam2mirror_tvec, // translation_camera->mirror
		const cv::Mat& mir2cam_rvec, // Rotation mirror->cam
		const cv::Mat& mir2cam_tvec) // Translation mirror->cam
{
	cv::Matx44d virtualdisp2cam = cv::Matx44d::eye(), observer = cv::Matx44d::eye();;

	// Create Transform from VirtualDispaly to Camera
	createaffine(vdisp2cam_rvec, vdisp2cam_tvec, virtualdisp2cam);

	//observer = virtualdisp2cam;
	observer = virtualdisp2cam;
	// The Affine transformation from Camera to Mirror
	cv::Matx44d affine_cam_to_mirr;
	createaffine(cam2mirror_rvec, cam2mirror_tvec, affine_cam_to_mirr);

	// Transformation VirtualDispCoord -> CameraCoord -> MirrorCoord
	//observer = virtualdisp2cam * affine_cam_to_mirr;
	observer = affine_cam_to_mirr * virtualdisp2cam;
	// In the mirror coordiante System apply the mirroring

	cv::Matx44d householder = cv::Matx44d::eye();
	householder(2, 2) = -1.0;

	// Transformation: VirtualdispalyCoord -> CameraCoord -> MirrorCoord -> Apply MirrorMatrix 
	// oberserver is no left handed !!!! 
	//observer = virtualdisp2cam * affine_cam_to_mirr * householder;

	observer = householder * affine_cam_to_mirr * virtualdisp2cam;

	cv::Matx44d affine_mirror2cam;

	createaffine(mir2cam_rvec, mir2cam_tvec, affine_mirror2cam);

	// The Transformation to the real coordiante System
	//observer = virtualdisp2cam * affine_cam_to_mirr * householder * affine_mirror2cam;

	observer = affine_mirror2cam * householder * affine_cam_to_mirr * virtualdisp2cam;

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
	
	std::cout << "Start Geometric Calibration Test \n";

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

	//std::cout << "Calibration Test: cam2mir_rvec" << data.cam2mir_rvec << std::endl;

	cv::Mat coordiante_mirror_rot =
		m_img_processing.rotateCoordinatedGrid(
			coordiante_mirror,
			data.cam2mir_rvec,
			Rotation::rodrigeuz);

	//std::cout << "Calibration Test: cam2mir_tvec" << data.cam2mir_tvec << std::endl;

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
			data.disp2cam_rvec, 
			//cv::Vec3d(cam2disp_rvec),
			Rotation::rodrigeuz
		);

	cv::Mat unwrap_world_rot_shift =
		m_img_processing.shiftCoordinateGrid(
			unwrap_world_rot,
			data.disp2cam_tvec
			//cam2disp_tvec
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