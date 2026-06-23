#ifndef TRIANGULATE_HPP
#define TRIANGULATE_HPP
#include <opencv2/opencv.hpp>
#include <fstream>
#include <cmath>
#include <optional>
#include <filesystem>


class Triangulate {
	struct TriangulationErr {
	private:
		struct ReprojectionErr {
			double meanErr{ 0.0 };
			double sqrtErr{ 0.0 };
			double maxErr{ 0.0 };
		};
	public:
		ReprojectionErr cam1{};
		ReprojectionErr cam2{};
	};
public:
	Triangulate(
		const cv::Mat& cameraMatrixPrim,
		const cv::Mat& distCoeffsPrim,
		const cv::Mat& cameraMatrixSecon,
		const cv::Mat& distCoeffsSecon, 
		const cv::Mat& rotationExtrin,
		const cv::Mat& translationExtrin)
		:m_cameraMatPrim{ cameraMatrixPrim }
		,m_distCoeffsPrim{distCoeffsPrim}
		,m_cameraMatSecon{cameraMatrixSecon}
		,m_distCoeffsSecon{distCoeffsSecon}
		,m_RotationExtrin{rotationExtrin}
		,m_TranslationExtrin{translationExtrin}
	{}

	cv::Point3f calculate(
		cv::Point2f primär,
		cv::Point2f sekundär)
	{
		cv::Mat homogenuos;

		std::vector<cv::Point2f> pt_prim{ primär };
		std::vector<cv::Point2f> pt_secon{ sekundär };

		std::vector<cv::Point2f> primär_undist, sekundär_undist;
		cv::Point3f world_pt;

		cv::Mat P1 = cv::Mat::eye(3, 4, CV_32F);
		cv::Mat P2 = cv::Mat::zeros(3, 4, CV_32F);
		m_RotationExtrin.copyTo(P2(cv::Rect(0, 0, 3, 3)));
		m_TranslationExtrin.copyTo(P2(cv::Rect(3, 0, 1, 3)));


		cv::undistortPoints(pt_prim, primär_undist, m_cameraMatPrim, m_distCoeffsPrim);
		cv::undistortPoints(pt_secon, sekundär_undist, m_cameraMatSecon, m_distCoeffsSecon);

		cv::triangulatePoints(P1, P2, primär_undist, sekundär_undist, homogenuos);
		
		homogenuos.convertTo(homogenuos, CV_64F);

		double x = homogenuos.at<double>(0, 0);
		double y = homogenuos.at<double>(1, 0);
		double z = homogenuos.at<double>(2, 0);
		double w = homogenuos.at<double>(3, 0);

		world_pt = cv::Point3f(
			static_cast<float>(x / w), 
			static_cast<float>(y / w),
			static_cast<float>(z / w));

		return world_pt;
	}

	TriangulationErr calculateError(
		const std::vector<cv::Point3f>& triangulated,
		const std::vector<cv::Point2f>& prim_cam,
		const std::vector<cv::Point2f>& secon_cam)
	{
		if (triangulated.size() != prim_cam.size() ||
			triangulated.size() != secon_cam.size())
			throw std::invalid_argument("Input containers have different for Triangulation Error");

		std::vector<cv::Point2f> reproj1;

		const std::size_t sz{ triangulated.size() };

		cv::projectPoints(
			triangulated,
			cv::Vec3d(0, 0, 0),                // Kamera 1 = Welt
			cv::Vec3d(0, 0, 0),
			m_cameraMatPrim,
			m_distCoeffsPrim,
			reproj1
		);

		cv::Mat rvec12;
		cv::Rodrigues(m_RotationExtrin, rvec12);

		std::vector<cv::Point2f> reproj2;

		cv::projectPoints(
			triangulated,
			rvec12,
			m_TranslationExtrin,
			m_cameraMatSecon,
			m_distCoeffsSecon,
			reproj2
		);
		double sum_prim = 0.0, sumSq_prim = 0.0, sum_secon = 0.0, sumSq_secon = 0.0;
		double maxErr_prim = 0.0, maxErr_secon = 0.0;
		size_t maxIdx_prim = 0, maxIdx_secon = 0;

		for (size_t i = 0; i < sz; ++i)
		{
			double e = cv::norm(reproj1[i] - prim_cam[i]);
			sum_prim += e;
			sumSq_prim += e * e;

			if (e > maxErr_prim) {
				maxErr_prim = e;
				maxIdx_prim = i;
			}
			e = cv::norm(reproj2[i] - secon_cam[i]);
			sum_secon += e;
			sumSq_secon += e * e;

			if (e > maxErr_secon) {
				maxErr_secon = e;
				maxIdx_secon = i;
			}
		}
		TriangulationErr Err{};
		Err.cam1.maxErr = maxErr_prim;
		Err.cam1.meanErr = sum_prim / sz;
		Err.cam1.sqrtErr = std::sqrt(sumSq_prim / sz);

		Err.cam2.maxErr = maxErr_secon;
		Err.cam2.meanErr = sum_secon / sz;
		Err.cam2.sqrtErr = std::sqrt(sumSq_secon / sz);

		return Err;
	}

	void saveFullProtocoll(
		const std::string& path,
		const std::vector<cv::Point2f>& imagePoints_master,
		const std::vector<cv::Point2f>& imagePoints_slave,
		const std::vector<cv::Point3f>& triangulated_pts,
		const std::optional<TriangulationErr> Err = std::optional<TriangulationErr>())
	{
		if (imagePoints_master.size() != imagePoints_slave.size() ||
			imagePoints_master.size() != triangulated_pts.size() || path.empty())
			throw std::invalid_argument("Different Size of the containers in Triangulated::saveFullProtocoll() \n");

		std::filesystem::path path_file(path);

		if (std::filesystem::exists(path_file)) 
		{
			std::cout << "The file " << std::filesystem::relative(path_file).c_str() <<
				"does already exists and will get overridden \n";
			std::filesystem::remove(path);
		}

		for (std::size_t i = 0; i < imagePoints_master.size(); ++i) {
			savePoint(
				path, 
				imagePoints_master[i], 
				imagePoints_slave[i], 
				triangulated_pts[i]);
		}

		if (Err.has_value()) {
			saveError(
				path,
				Err.value());
		}
	}

	void saveError(
		const std::string& path,
		const TriangulationErr& err)
	{
		if (path.empty()) throw std::invalid_argument("Path is empty \n");

		std::ofstream file(path, std::ios_base::app);
		if (!file.is_open()) {
			throw std::runtime_error("Could not open file");
		}

		file << '\n' << "Reprojection Error Primary Camera \n";
		file << "Max Err: " << err.cam1.maxErr << '\n';
		file << "Mean Err: " << err.cam1.meanErr << '\n';
		file << "Sqrt Err: " << err.cam1.sqrtErr << '\n';
		file << '\n' << "Reprojection Error Secondary Camera \n";
		file << "Max Err: " << err.cam2.maxErr << '\n';
		file << "Mean Err: " << err.cam2.meanErr << '\n';
		file << "Sqrt Err: " << err.cam2.sqrtErr << '\n';

		file.close();
	}


	void savePoint(
		const std::string& path,
		cv::Point2f image_master,
		cv::Point2f image_slave,
		cv::Point3f worldPoint)
	{
		std::ofstream file(path, std::ios_base::app);
		if (!file.is_open()) {
			throw std::runtime_error("Could not open file");
		}

		if(std::streampos(0) == file.tellp())
			file << "System, x,y,z \n";

		file << "ImageMaster," << image_master.x << "," << image_master.y << '\n';
		file << "ImageSlave," << image_slave.x << "," << image_slave.y << '\n';
		file << "Triangulated," << worldPoint.x << "," << worldPoint.y << "," << worldPoint.z << '\n';

		file.close();
	}

private:
	cv::Mat m_cameraMatPrim{};
	cv::Mat m_distCoeffsPrim{};
	cv::Mat m_cameraMatSecon{};
	cv::Mat m_distCoeffsSecon{};
	cv::Mat m_RotationExtrin{};
	cv::Mat m_TranslationExtrin{};

	
};


#endif