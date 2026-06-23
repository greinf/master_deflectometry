#include "GeometricCalibration.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "imgProcessing.hpp"
#include <open3d/pipelines/registration/TransformationEstimation.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/visualization/utility/DrawGeometry.h>
#include <open3d/pipelines/registration/Registration.h>
#include <open3d/pipelines/registration/CorrespondenceChecker.h>
#include "cmath"
#include <tuple>
#include <Eigen/Dense>
#include "Triangulate.hpp"
#include <map>
#include <algorithm>

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

	struct Mirror2Cam {
		cv::Mat rvec{};
		cv::Mat tvec{};
	};
	struct Cam2VirtDisp {
		cv::Mat rvec{};
		cv::Mat tvec{};
	};

public:
	explicit Impl(const GeometricCalibrationConfig& config) 
		:m_config{config}{ }
	
	const GeometricCalibrationConfig& m_config;

	cv::Mat working_img;
	cv::Mat mask_ROI_prim;
	cv::Mat mask_ROI_secon;
	cv::Mat mask_contrast_prim;
	cv::Mat mask_contrast_secon;
	cv::Size working_size;
	Cam2Mirror cam2mirror{};
	VirtDisp2Cam virtDisp2cam{};
	Mirror2Cam mirror2cam{};
	Cam2VirtDisp cam2Virtdisp{};

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

StereoCalibrationPoints GeometricCalibration::
getCommonImagePointsfromStereoUnwrap(
	const std::vector<cv::Mat>& unwrap_prim,
	const std::vector<cv::Mat>& unwrap_secon,
	const cv::Mat& mask_prim,
	const cv::Mat& mask_secon,
	const double& wavelength)
{
	CV_Assert(unwrap_prim.size() == unwrap_secon.size());
	CV_Assert(unwrap_prim.size() >= 2);

	// Convert in PixelCoordiantes
	cv::Mat unwrap_primX = unwrap_prim[0].clone();
	unwrap_primX *= (wavelength / CV_2PI); 
	cv::Mat unwrap_primY = unwrap_prim[1].clone();
	unwrap_primY *= (wavelength / CV_2PI);
	cv::Mat unwrap_seconX = unwrap_secon[0].clone();
	unwrap_seconX *= (wavelength / CV_2PI);
	cv::Mat unwrap_seconY = unwrap_secon[1].clone();
	unwrap_seconY *= (wavelength / CV_2PI);

	struct comparator {
		bool operator()(const cv::Point2f& pt1, const cv::Point2f& pt2) const {
			if (pt1.x != pt2.x)
				return pt1.x < pt2.x;
			return pt1.y < pt2.y;
		}
	};

	// unwrapPrim: Key -> Unwrap(x,y), Value -> ImageCoord(u,v)
	std::map<cv::Point2f, cv::Point2f, comparator> primaryCam{};
	// unwarpSecon: Key -> Unwrap(x,y), Value -> ImageCoord(u,v)
	std::map<cv::Point2f, cv::Point2f, comparator> secondaryCam{};

	unwrap_primX.forEach<double>([](double& val, const int* pos) -> void {
		val = std::round(val);
		});
	unwrap_primY.forEach<double>([](double& val, const int* pos) -> void {
		val = std::round(val);
		});
	unwrap_seconX.forEach<double>([](double& val, const int* pos) -> void {
		val = std::round(val);
		});
	unwrap_seconY.forEach<double>([](double& val, const int* pos) -> void {
		val = std::round(val);
		});
	
	for (int row = 0; row < unwrap_primX.rows; ++row) {
		const uchar* mask_prim_ptr = mask_prim.ptr<uchar>(row);
		const double* x_ptr = unwrap_primX.ptr<double>(row);
		const double* y_ptr = unwrap_primY.ptr<double>(row);
		for (int col = 0; col < unwrap_primX.cols; ++col) {
			if (mask_prim_ptr[col] == 0) continue;
			if (std::isnan(x_ptr[col])) continue;
			if (std::isnan(y_ptr[col])) continue;
			primaryCam.insert(
				std::pair<cv::Point2f, cv::Point2f>{
					{static_cast<float>(x_ptr[col]), static_cast<float>(y_ptr[col])},
					{static_cast<float>(col), static_cast<float>(row)}
			});
		}
	}
	for (int row = 0; row < unwrap_seconX.rows; ++row) {
		const uchar* mask_secon_ptr = mask_secon.ptr<uchar>(row);
		const double* x_ptr = unwrap_seconX.ptr<double>(row);
		const double* y_ptr = unwrap_seconY.ptr<double>(row);
		for (int col = 0; col < unwrap_seconY.cols; ++col) {
			if (mask_secon_ptr[col] == 0) continue;
			if (std::isnan(x_ptr[col])) continue;
			if (std::isnan(y_ptr[col])) continue;
			secondaryCam.insert(
				std::pair<cv::Point2f, cv::Point2f>{
					{static_cast<float>(x_ptr[col]), static_cast<float>(y_ptr[col])},
					{static_cast<float>(col), static_cast<float>(row)}
			});
		}
	}

	std::vector<cv::Point2f> commonKeys{};

	auto it1 = primaryCam.begin();
	auto it2 = secondaryCam.begin();

	comparator comp;

	while (it1 != primaryCam.end() && it2 != secondaryCam.end()) {
		const auto& k1 = it1->first;
		const auto& k2 = it2->first;

		if (comp(k1, k2)) {
			++it1;
		}
		else if (comp(k2, k1)) {
			++it2;
		}
		else {
			commonKeys.push_back(k1);
			++it1;
			++it2;
		}
	}

	if (commonKeys.size() == 0) 
		throw std::runtime_error("No common ImagePoints for primary and Secondary Camera \n");

	StereoCalibrationPoints calibPoints{};
	calibPoints.imagePoints_prim.reserve(commonKeys.size());
	calibPoints.imagePoints_secon.reserve(commonKeys.size());

	for (const auto& key : commonKeys) {
		calibPoints.imagePoints_prim.push_back(primaryCam.at(key));
		calibPoints.imagePoints_secon.push_back(secondaryCam.at(key));
	}

	return calibPoints;
}



GeometricCalibrationResult GeometricCalibration::calibrateStereo(
	GeometricCalibrationData_Stereo& data)
{
	if (!data.validData()) throw std::invalid_argument("Dataholder is not valid for geometric Stereo Calibration");

	std::cout << "StereoCalibration start " << std::endl;

	m_impl->mask_ROI_prim = data.mask_ROI;
	// Try to create here a Mask only via Base intensity. in the hole ROI marked. 
	// If not possible we need to check if Graycode must be applied also for the secon Camera. 
	//m_impl->mask_ROI_secon = data.mask_ROI_secon;

	cv::Mat amplx_prim, amplx_secon, amply_prim, amply_secon;

	cv::multiply((*data.contrast)[0], (*data.biasIntensity)[0], amplx_prim);
	cv::multiply((*data.contrast)[1], (*data.biasIntensity)[1], amply_prim);

	cv::multiply((*data.contrast_sec_cam)[0], (*data.biasIntensity_sec_cam)[0], amplx_secon);
	cv::multiply((*data.contrast_sec_cam)[1], (*data.biasIntensity_sec_cam)[1], amply_secon);

	std::vector<cv::Mat> amplitude_prim{ amplx_prim, amply_prim };
	std::vector<cv::Mat> amplitude_secon{ amplx_secon, amply_secon };

	m_impl->working_size = m_impl->mask_ROI_prim.size();

	m_impl->mask_ROI_secon = m_img_processing.createMask(*data.contrast_sec_cam, 0.5);
	m_impl->mask_contrast_prim = m_img_processing.createAdaptiveMask(amplitude_prim, 30, 0.95);
	m_impl->mask_contrast_secon = m_img_processing.createAdaptiveMask(amplitude_secon, 30, 0.95);

	cv::Mat biasIntensityPrimary, biasIntensitySecondary,
		amplitudePrimary, amplitudeSecondary,
		IntensityPrimary8U, IntensitySecondary8U, homogenuosPoint;


	biasIntensityPrimary = m_img_processing.mean(*data.biasIntensity);   // amplitude_prim
	biasIntensitySecondary = m_img_processing.mean(*data.biasIntensity_sec_cam);  //amplitude_secon
	
	cv::normalize(biasIntensityPrimary, IntensityPrimary8U, 0, 255.0, cv::NORM_MINMAX, CV_8U);
	cv::normalize(biasIntensitySecondary, IntensitySecondary8U, 0, 255.0, cv::NORM_MINMAX, CV_8U);

	std::vector<cv::Vec2d> circleImgCoordPrimary, circleImgCoordSecondary;

	std::vector<cv::Point2f> circleImgCoordPrimary_p, circleImgCoordSecondary_p;

	std::vector<cv::Point3f> triangulatedPts;

	circleImgCoordPrimary = m_img_processing.getCircleCoordinates(
		IntensityPrimary8U,
		m_impl->mask_ROI_prim,
		m_impl->m_config.pattern_size,
		data.path + "/primary.png");
	circleImgCoordSecondary = m_img_processing.getCircleCoordinates(
		IntensitySecondary8U,
		m_impl->mask_ROI_secon,
		m_impl->m_config.pattern_size,
		data.path + "/secondary.png");

	cv::TermCriteria criteria(cv::TermCriteria::EPS, 0, 1e-9);

	if (circleImgCoordPrimary.size() != circleImgCoordSecondary.size())
		throw std::runtime_error("DifferentPoint sizes in both containers \n");

	circleImgCoordPrimary_p.reserve(circleImgCoordPrimary.size());
	circleImgCoordSecondary_p.reserve(circleImgCoordSecondary.size());
	triangulatedPts.reserve(circleImgCoordSecondary.size());

	Triangulate triangulate(
		data.camMat,
		data.distCoeffs,
		data.camMat_secundaryCam,
		data.distCoeffs_secondaryCam,
		data.cam2cam_rotMat,
		data.cam2cam_tvec
	);

	for (std::size_t i = 0; i < circleImgCoordPrimary.size(); ++i)
	{
		circleImgCoordPrimary_p.push_back({
			static_cast<float>(circleImgCoordPrimary[i][0]),
			static_cast<float>(circleImgCoordPrimary[i][1])
			});
		circleImgCoordSecondary_p.push_back({
			static_cast<float>(circleImgCoordSecondary[i][0]),
			static_cast<float>(circleImgCoordSecondary[i][1])
			});
		triangulatedPts.push_back(
			triangulate.calculate(
				circleImgCoordPrimary_p[i],
				circleImgCoordSecondary_p[i]
			)
		);
	}

	auto reprojectionErr =
		triangulate.calculateError(
			triangulatedPts,
			circleImgCoordPrimary_p,
			circleImgCoordSecondary_p);
	
	triangulate.saveFullProtocoll(
		data.path + "/triangulatedPtsMirror.csv",
		circleImgCoordPrimary_p,
		circleImgCoordSecondary_p,
		triangulatedPts,
		std::make_optional(reprojectionErr)
	);
	
	std::vector<cv::Vec3d> patternObjectPoints =
		m_img_processing.createCalibPatternObjectPoints(
			m_impl->m_config.pattern_size,
			m_impl->m_config.point_dist
		);

	// Create The point Clouds
	std::vector<Eigen::Vector3d> triangulated_pts_eigen_mirr, 
		pattern_pts_eigen_mirr;

	triangulated_pts_eigen_mirr.reserve(patternObjectPoints.size());
	pattern_pts_eigen_mirr.reserve(patternObjectPoints.size());

	open3d::pipelines::registration::CorrespondenceSet correspondence;
	correspondence.reserve(patternObjectPoints.size());

	for (std::size_t i = 0; i < patternObjectPoints.size(); ++i) {
		triangulated_pts_eigen_mirr.push_back(Eigen::Vector3d(
			static_cast<double>(triangulatedPts[i].x),
			static_cast<double>(triangulatedPts[i].y),
			static_cast<double>(triangulatedPts[i].z)));
		pattern_pts_eigen_mirr.push_back(Eigen::Vector3d(
			patternObjectPoints[i].val[0],
			patternObjectPoints[i].val[1],
			patternObjectPoints[i].val[2]));
	}

	std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>> calibPoints =
		m_img_processing.do_calibration_Points(
			*data.unwrap,
			m_impl->mask_contrast_prim,
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

	m_impl->virtDisp2cam.rvec = (cv::Mat_<double>(3, 1) << 0.0, CV_PI, 0.0);
	m_impl->virtDisp2cam.tvec = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 2500.0);

	std::vector<int> inliers_d;

	bool ok = cv::solvePnPRansac(
		objectPointsDisp,
		imagePointsDisp,
		data.camMat,
		data.distCoeffs,
		m_impl->virtDisp2cam.rvec,
		m_impl->virtDisp2cam.tvec,
		true,                          // useExtrinsicGuess
		2000,                           // iterationsCount
		2.0,                            // reprojectionError [px]
		0.99,                           // confidence
		inliers_d,
		cv::SOLVEPNP_IPPE               // good for planar object points
	);

	std::cout << inliers_d.size() << " Points are inliers for Resektion from Ransac from " << objectPointsDisp.size() <<
		" availabel points" << std::endl;

	if (!ok || inliers_d.size() < 10)
	{
		std::cout << "solvePnPRansac failed for virtual display pose\n";
		return {};
	}

	std::vector<cv::Point3d> objectInliers;
	std::vector<cv::Point2d> imageInliers;

	objectInliers.reserve(inliers_d.size());
	imageInliers.reserve(inliers_d.size());

	for (int idx : inliers_d)
	{
		objectInliers.push_back(objectPointsDisp[idx]);
		imageInliers.push_back(imagePointsDisp[idx]);
	}

	ok = cv::solvePnP(
		objectInliers,
		imageInliers,
		data.camMat,
		data.distCoeffs,
		m_impl->virtDisp2cam.rvec,
		m_impl->virtDisp2cam.tvec,
		true,
		cv::SOLVEPNP_IPPE
	);

	if (!ok)
	{
		std::cout << "solvePnP IPPE refinement failed\n";
		return {};
	}

	cv::solvePnPRefineLM(
		objectInliers,
		imageInliers,
		data.camMat,
		data.distCoeffs,
		m_impl->virtDisp2cam.rvec,
		m_impl->virtDisp2cam.tvec
	);

	open3d::geometry::PointCloud triangulated_mirr(triangulated_pts_eigen_mirr);
	open3d::geometry::PointCloud pattern_mirr(pattern_pts_eigen_mirr);

	// auto index = calculateDistanceToPlane(triangulated_mirr, 2, false);

	// Now create the correspondence
	for (std::size_t i = 0; i < triangulated_pts_eigen_mirr.size(); ++i) {
		correspondence.emplace_back(
			static_cast<int>(i),
			static_cast<int>(i));
	}

	// Finds transformation from pattern to Triangulated Points (in Rectified System!!!)
	// cam -> mir
	open3d::pipelines::registration::TransformationEstimationPointToPoint poseEstimation(false);
	// open3d::pipelines::registration::RANSACConvergenceCriteria converg(1e7, 0.9999);
	// open3d::pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength checker(0.8);
	// std::vector<std::reference_wrapper<const open3d::pipelines::registration::CorrespondenceChecker>> ref_checker{ checker };


	// Calculates x_triang = transfrom * x_pattern
	open3d::pipelines::registration::RegistrationResult result_registration_mirr = 
		open3d::pipelines::registration::RegistrationRANSACBasedOnCorrespondence(
			pattern_mirr,
			triangulated_mirr,
			correspondence,
			0.5,
			poseEstimation,
			5
			// ref_checker
			//converg
		);

	const open3d::pipelines::registration::CorrespondenceSet& inliers = result_registration_mirr.correspondence_set_;

	std::cout << inliers.size() << " Points are inliers in Registration \n";

	Eigen::Matrix4d mir2cam_affine = poseEstimation.ComputeTransformation(
		pattern_mirr,
		triangulated_mirr,
		inliers
	);

	std::cout << "Transformation Mirr2cam: \n" << 
		mir2cam_affine << std::endl;

	cv::Mat mir2cam_Rot(3, 3, CV_64F);
	cv::Mat mir2cam_tvec(3, 1, CV_64F);

	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			mir2cam_Rot.at<double>(i, j) = mir2cam_affine(i, j);

	mir2cam_tvec.at<double>(0, 0) = mir2cam_affine(0, 3);
	mir2cam_tvec.at<double>(1, 0) = mir2cam_affine(1, 3);
	mir2cam_tvec.at<double>(2, 0) = mir2cam_affine(2, 3);

	cv::Mat cam2mir_Rot = mir2cam_Rot.t();
	cv::Mat cam2mir_tvec = -cam2mir_Rot * mir2cam_tvec;

	cv::Mat mir2cam_rvec, cam2mir_rvec;
	cv::Rodrigues(mir2cam_Rot, mir2cam_rvec);
	cv::Rodrigues(cam2mir_Rot, cam2mir_rvec);
	
	m_impl->cam2mirror.rvec = cam2mir_rvec;
	m_impl->cam2mirror.tvec = cam2mir_tvec;
	m_impl->mirror2cam.rvec = mir2cam_rvec;
	m_impl->mirror2cam.tvec = mir2cam_tvec;


	GeometricCalibrationResult result{};

	result.mir2cam_rvec = m_impl->mirror2cam.rvec;
	result.mir2cam_tvec = m_impl->mirror2cam.tvec;
	result.virtDisp2cam_rvec = m_impl->virtDisp2cam.rvec;
	result.virtDisp2cam_tvec = m_impl->virtDisp2cam.tvec;

	std::cout << "Reach this. \n";

	backToWorld(
		result.disp2cam_rvec,
		result.disp2cam_tvec,
		m_impl->virtDisp2cam.rvec,
		m_impl->virtDisp2cam.tvec,
		m_impl->mirror2cam.rvec,
		m_impl->mirror2cam.tvec,
		m_impl->cam2mirror.rvec,
		m_impl->cam2mirror.tvec
	);

	std::cout << "Reachd end \n";
	return result;

	//// Here the Detection of the Virtual Display!!! 
	//StereoCalibrationPoints virtualDisp_Stereo =
	//	getCommonImagePointsfromStereoUnwrap(
	//		*data.unwrap,
	//		*data.unwrap_sec_cam,
	//		m_impl->mask_contrast_prim,
	//		m_impl->mask_contrast_secon,
	//		m_impl->m_config.wavelength_phase
	//);

	//if (virtualDisp_Stereo.imagePoints_prim.size() !=
	//	virtualDisp_Stereo.imagePoints_secon.size())
	//	throw std::runtime_error("Different Containersizes for the UnwrapImagePoints \n");
	//
	//const std::size_t virtualDispImagePt_sz =
	//	virtualDisp_Stereo.imagePoints_prim.size();

	//std::vector<cv::Point3f> triangualtedVirtDisp;
	//triangualtedVirtDisp.reserve(virtualDispImagePt_sz);

	//for (std::size_t i = 0; i < virtualDispImagePt_sz; ++i)
	//{
	//	triangualtedVirtDisp.push_back(
	//		triangulate.calculate(
	//			virtualDisp_Stereo.imagePoints_prim[i],
	//			virtualDisp_Stereo.imagePoints_secon[i]
	//		)
	//	);
	//}

	//reprojectionErr =
	//	triangulate.calculateError(
	//		triangualtedVirtDisp,
	//		virtualDisp_Stereo.imagePoints_prim,
	//		virtualDisp_Stereo.imagePoints_secon);

	//triangulate.saveFullProtocoll(
	//	data.path + "/triangulatedPtsMirror.csv",
	//	virtualDisp_Stereo.imagePoints_prim,
	//	virtualDisp_Stereo.imagePoints_secon,
	//	triangualtedVirtDisp,
	//	std::make_optional(reprojectionErr)
	//);

	//// CreatePoint Cloud from Triangulated Pts and Image Pts. 
	//std::vector<Eigen::Vector3d> triangulatedVirtDisp_Eigen;
	//std::vector<Eigen::Vector3d> unwrapImagePts_Eigen;
	//triangulatedVirtDisp_Eigen.reserve(virtualDispImagePt_sz);
	//unwrapImagePts_Eigen.reserve(virtualDispImagePt_sz);

	//for (std::size_t i = 0; i < virtualDispImagePt_sz; ++i) {
	//	triangulatedVirtDisp_Eigen.push_back(
	//		Eigen::Vector3d{
	//			static_cast<double>(triangualtedVirtDisp[i].x),
	//			static_cast<double>(triangualtedVirtDisp[i].y),
	//			static_cast<double>(triangualtedVirtDisp[i].z)
	//			}
	//	);
	//	unwrapImagePts_Eigen.push_back(
	//		Eigen::Vector3d{
	//			static_cast<double>(virtualDisp_Stereo.imagePoints_prim[i].x),
	//			static_cast<double>(virtualDisp_Stereo.imagePoints_prim[i].y),
	//			0.0
	//		}
	//	);
	//}
	//
	//open3d::geometry::PointCloud triangulated_Disp(triangulatedVirtDisp_Eigen);
	//open3d::geometry::PointCloud pattern_Disp(unwrapImagePts_Eigen);


	//correspondence.clear();
	//// Now create the correspondence
	//for (std::size_t i = 0; i < virtualDispImagePt_sz; ++i) {
	//	correspondence.emplace_back(
	//		static_cast<int>(i),
	//		static_cast<int>(i));
	//}

	////open3d::pipelines::registration::RANSACConvergenceCriteria converg(1e2, 0.9999);
	////open3d::pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength checker(0.99);
	////std::vector<std::reference_wrapper<const open3d::pipelines::registration::CorrespondenceChecker>> ref_checker{ checker };

	//// Calculates x_triang = transfrom * x_pattern
	//open3d::pipelines::registration::RegistrationResult result_registration =
	//	open3d::pipelines::registration::RegistrationRANSACBasedOnCorrespondence(
	//		pattern_Disp,
	//		triangulated_Disp,
	//		correspondence,
	//		0.2,
	//		poseEstimation,
	//		3
	//	);

	//const open3d::pipelines::registration::CorrespondenceSet& inliers_disp = result_registration.correspondence_set_;

	//Eigen::Matrix4d virtDisp2cam_affine =
	//	open3d::pipelines::registration::TransformationEstimationPointToPoint(false).ComputeTransformation(
	//		pattern_Disp,
	//		triangulated_Disp,
	//		inliers_disp
	//	);

	//cv::Mat virt2cam_rvec, virt2cam_tvec(3,1, CV_64F, cv::Scalar(0)), cam2virt_rvec, cam2virt_tvec;

	//cv::Mat virt2camRot(cv::Size(3, 3), CV_64F, cv::Scalar(0.0));
	//for (int row = 0; row < 3; ++row) {
	//	for (int col = 0; col < 3; ++col) {
	//		virt2camRot.at<double>(row, col) = virtDisp2cam_affine(row, col);
	//	}
	//}
	//
	//virt2cam_tvec.at<double>(0, 0) = virtDisp2cam_affine(0, 3);
	//virt2cam_tvec.at<double>(1, 0) = virtDisp2cam_affine(1, 3);
	//virt2cam_tvec.at<double>(2, 0) = virtDisp2cam_affine(2, 3);
	//
	//cv::Rodrigues(virt2camRot, virt2cam_rvec);
}

std::vector<std::size_t> GeometricCalibration::calculateDistanceToPlane(
	const open3d::geometry::PointCloud& points,
	const double inliers_distance,
	bool visualize)
{
	Eigen::Vector4d surfacePara = fitPlane(points, inliers_distance);

	std::vector<double> distances;
	distances.reserve(points.points_.size());

	Eigen::Vector3d n(surfacePara(0), surfacePara(1), surfacePara(2));
	const double n_norm = n.norm();

	for (const auto& p : points.points_) {
		double dist = std::abs(
			surfacePara(0) * p.x() +
			surfacePara(1) * p.y() +
			surfacePara(2) * p.z() +
			surfacePara(3)
		) / n_norm;

		distances.push_back(dist);
	}

	double sum = 0.0;
	double max_dist = 0.0;

	for (double dist : distances) {
		sum += dist;
		max_dist = std::max(max_dist, dist);
	}

	const double mean_dist = sum / distances.size();

	double variance = 0.0;
	for (double dist : distances) {
		const double diff = dist - mean_dist;
		variance += diff * diff;
	}

	const double std_dev = std::sqrt(variance / distances.size());

	const double threshold = mean_dist + 2 * std_dev;

	std::vector<std::size_t> outlier_indices;

	for (std::size_t i = 0; i < distances.size(); ++i) {
		if (distances[i] > threshold) {
			outlier_indices.push_back(i);
		}
	}

	std::cout << "Mean distance to plane: " << mean_dist << "\n";
	std::cout << "Max distance to plane: " << max_dist << "\n";
	std::cout << "Std deviation: " << std_dev << "\n";
	std::cout << "Outlier threshold: " << threshold << "\n";
	std::cout << "Number of outliers: " << outlier_indices.size() << "\n";

	if (visualize) {
		Eigen::Vector3d p0 = -surfacePara(3) * n / n.squaredNorm();

		Eigen::Vector3d v1 = n.unitOrthogonal();
		Eigen::Vector3d v2 = n.cross(v1).normalized();

		double size = 400.0;

		std::vector<Eigen::Vector3d> vertices = {
			p0 + size * v1 + size * v2,
			p0 + size * v1 - size * v2,
			p0 - size * v1 - size * v2,
			p0 - size * v1 + size * v2
		};

		auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();

		mesh->vertices_ = vertices;
		mesh->triangles_ = {
			Eigen::Vector3i(0, 1, 2),
			Eigen::Vector3i(0, 2, 3)
		};

		mesh->ComputeVertexNormals();
		mesh->PaintUniformColor(Eigen::Vector3d(0.8, 0.2, 0.2));

		auto coord = open3d::geometry::TriangleMesh::CreateCoordinateFrame(500.0);

		open3d::visualization::DrawGeometries({
			std::make_shared<open3d::geometry::PointCloud>(points),
			mesh,
			coord
			});
	}

	return outlier_indices;
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
	// The mask from GrayCode defining an valid ROI
	m_impl->mask_ROI_prim = data.mask_ROI;
	m_impl->working_size = m_impl->mask_ROI_prim.size();

	std::vector<cv::Mat> masking;
	for (std::size_t i = 0; i < 2; ++i) {
		cv::Mat img;
		cv::multiply((*data.contrast)[i], (*data.biasIntensity)[i], img);
		masking.push_back(img);
	}

	m_impl->mask_contrast_prim = m_img_processing.createAdaptiveMask(masking, 20, 1);

	// Virtual Display Pose Estimation 
	// Through Unwrap Pictures. 
	std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>> calibPoints =
		m_img_processing.do_calibration_Points(
			*data.unwrap,
			m_impl->mask_contrast_prim,
			m_impl->m_config.wavelength_phase,
			1,
			1,
			m_impl->m_config.displayPixelPitch);

	
	cv::Mat mean = m_img_processing.mean(*data.biasIntensity);

	cv::normalize(mean, m_impl->working_img, 0, 255, cv::NORM_MINMAX, CV_8U);

	// Mirror Pose estimation 
	// imagePoints - Circles in the images position
	std::vector<cv::Vec2d> imagePoints =
		m_img_processing.getCircleCoordinates(
			m_impl->working_img,
			m_impl->mask_ROI_prim,
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
	 
	//std::vector<cv::Mat> mir2cam_allSolutions_rvec, mir2cam_allSolutions_tvec;
	// mir2cam_allSolutions_rvec,
	// mir2cam_allSolutions_tvec,

	m_impl->mirror2cam.rvec = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
	m_impl->mirror2cam.tvec = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1000);

	bool solvePnP = cv::solvePnP(
		object,
		image,
		data.camMat,
		data.distCoeffs,
		m_impl->mirror2cam.rvec,
		m_impl->mirror2cam.tvec,
		true,
		cv::SOLVEPNP_IPPE);

	//if (!solvePnP || mir2cam_allSolutions_rvec.empty()) {
	//	std::cout << "SolvePnP failed for calculating the mirror pose\n";
	//	return {};
	//}

	//for (std::size_t i = 0; i < mir2cam_allSolutions_rvec.size(); ++i) {
	//	// It is assumed that there is a Mirror Coordiante System with nearly no Rotation
	//	// with respect to Cammera coordinate System
	//	if (cv::norm(mir2cam_allSolutions_rvec[i]) < CV_PI) {
	//		m_impl->mirror2cam.rvec = mir2cam_allSolutions_rvec[i].clone();
	//		m_impl->mirror2cam.tvec = mir2cam_allSolutions_tvec[i].clone();
	//		break;
	//	}
	//}
	
	cv::solvePnPRefineLM(
		object,
		image,
		data.camMat,
		data.distCoeffs,
		m_impl->mirror2cam.rvec,
		m_impl->mirror2cam.tvec
	);

	cv::Mat mir2cam_R;
	cv::Rodrigues(m_impl->mirror2cam.rvec, mir2cam_R);

	//// If Slave Camera was used one must transform back to primary
	//// This is just an extension for the normal method where we claculate via Master Camera !
	//cv::Mat primary2secondaryRot = data.cam2camRot;
	//cv::Mat primary2secondaryTvec = data.cam2camTvec;
	//cv::Mat secondary2primaryRot = data.cam2camRot.t();
	//cv::Mat secondary2primaryTvec = -secondary2primaryRot * primary2secondaryTvec;

	//cv::Mat mirror2primcamR = secondary2primaryRot * mir2cam_R;

	//cv::Mat t_obj2primary = secondary2primaryRot * m_impl->mirror2cam.tvec + secondary2primaryTvec;

	//cv::Rodrigues(mirror2primcamR, m_impl->mirror2cam.rvec);
	//m_impl->cam2mirror.tvec = t_obj2primary;

	//mir2cam_R = mirror2primcamR;

	cv::Mat cam2mir_R = mir2cam_R.t();
	cv::Mat cam2mir_t = -cam2mir_R * m_impl->mirror2cam.tvec;

	m_impl->cam2mirror.tvec = cam2mir_t;
	cv::Rodrigues(cam2mir_R, m_impl->cam2mirror.rvec);

	m_impl->virtDisp2cam.rvec = (cv::Mat_<double>(3, 1) << 0.0, CV_PI, 0.0);
	m_impl->virtDisp2cam.tvec = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 2000);
	 
	std::vector<int> inliers;
	
	bool ok = cv::solvePnPRansac(
		objectPointsDisp,
		imagePointsDisp,
		data.camMat,
		data.distCoeffs,
		m_impl->virtDisp2cam.rvec,
		m_impl->virtDisp2cam.tvec,
		true,                          // useExtrinsicGuess
		2000,                           // iterationsCount
		2.0,                            // reprojectionError [px]
		0.99,                           // confidence
		inliers,
		cv::SOLVEPNP_IPPE               // good for planar object points
	);

	if (!ok || inliers.size() < 4)
	{
		std::cout << "solvePnPRansac failed for virtual display pose\n";
		return {};
	}

	std::vector<cv::Point3f> objectInliers;
	std::vector<cv::Point2f> imageInliers;

	objectInliers.reserve(inliers.size());
	imageInliers.reserve(inliers.size());

	for (int idx : inliers)
	{
		objectInliers.push_back(objectPointsDisp[idx]);
		imageInliers.push_back(imagePointsDisp[idx]);
	}

	ok = cv::solvePnP(
		objectInliers,
		imageInliers,
		data.camMat,
		data.distCoeffs,
		m_impl->virtDisp2cam.rvec,
		m_impl->virtDisp2cam.tvec,
		true,                 
		cv::SOLVEPNP_IPPE
	);

	if (!ok)
	{
		std::cout << "solvePnP IPPE refinement failed\n";
		return {};
	}

	cv::solvePnPRefineLM(
		objectInliers,
		imageInliers,
		data.camMat,
		data.distCoeffs,
		m_impl->virtDisp2cam.rvec,
		m_impl->virtDisp2cam.tvec
	);

	GeometricCalibrationResult result{};

	/*cv::Mat virtdisp2SlaveR;
	cv::Rodrigues(m_impl->virDisp2cam.rvec, virtdisp2SlaveR);

	cv::Mat virtdisp2MasterR = secondary2primaryRot * virtdisp2SlaveR;

	cv::Mat virtdisp2Master_tvec = -virtdisp2MasterR * m_impl->virDisp2cam.tvec;*/


	//// Another extension for the Slave Camera:
	//cv::Mat secondary2primaryRot = data.cam2camRot.t();
	//cv::Mat secondary2primaryTvec = -secondary2primaryRot * primary2secondaryTvec;

	//cv::Mat mirror2primcamR = secondary2primaryRot * mir2cam_R;

	//cv::Mat t_obj2primary = secondary2primaryRot * m_impl->mirror2cam.tvec + secondary2primaryTvec;

	//cv::Rodrigues(mirror2primcamR, m_impl->mirror2cam.rvec);
	//m_impl->cam2mirror.tvec = t_obj2primary;

	//mir2cam_R = mirror2primcamR;

	result.virtDisp2cam_rvec = m_impl->virtDisp2cam.rvec;
	result.virtDisp2cam_tvec = m_impl->virtDisp2cam.tvec;
	result.mir2cam_rvec = m_impl->mirror2cam.rvec;
	result.mir2cam_tvec = m_impl->mirror2cam.tvec;

	backToWorld(
		result.disp2cam_rvec,
		result.disp2cam_tvec,
		m_impl->virtDisp2cam.rvec,
		m_impl->virtDisp2cam.tvec,
		m_impl->mirror2cam.rvec,
		m_impl->mirror2cam.tvec,
		m_impl->cam2mirror.rvec,
		m_impl->cam2mirror.tvec
	);

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
		const cv::Mat& virtdisp2cam_rvec,  
		const cv::Mat& virtDisp2cam_tvec,  
		const cv::Mat& mirror2cam_rvec, 
		const cv::Mat& mirror2cam_tvec, 
		const cv::Mat& cam2mirror_rvec,
		const cv::Mat& cam2mirror_tvec) 
{
	cv::Matx44d virtDisp2cam = cv::Matx44d::eye(), observer = cv::Matx44d::eye();;

	createaffine(virtdisp2cam_rvec, virtDisp2cam_tvec, virtDisp2cam);

	observer = virtDisp2cam;

	cv::Matx44d affine_cam2mirror;
	createaffine(cam2mirror_rvec, cam2mirror_tvec, affine_cam2mirror);

	observer = affine_cam2mirror * virtDisp2cam;
	
	cv::Matx44d householder = cv::Matx44d::eye();
	householder(2, 2) = -1.0;

	observer = householder * affine_cam2mirror * virtDisp2cam;

	cv::Matx44d affine_mirror2cam;
	createaffine(mirror2cam_rvec, mirror2cam_tvec, affine_mirror2cam);

	observer = affine_mirror2cam * householder * affine_cam2mirror * virtDisp2cam;

	std::cout << "Disp 2 Camera: \n" << observer << std::endl;

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

	m_impl->mask_contrast_prim = m_img_processing.createMask(
		*data.contrast,
		0.3);

	cv::Size sz = m_impl->mask_contrast_prim.size();

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
			data.mir2cam_rvec,
			Rotation::rodrigeuz);

	//std::cout << "Calibration Test: cam2mir_tvec" << data.cam2mir_tvec << std::endl;

	cv::Mat coordiante_mirror_shift =
		m_img_processing.shiftCoordinateGrid(
			coordiante_mirror_rot,
			data.mir2cam_tvec
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
		m_impl->mask_contrast_prim
	);

	return surface_normals;
}


GeometricCalibration::~GeometricCalibration() = default;