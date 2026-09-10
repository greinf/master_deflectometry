#ifndef CAMERAMATRIX_HPP
#define CAMERAMATRIX_HPP

#include "Object.hpp"
#include "Utils.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <ios>
#include <Eigen/dense>


// Everything is stored in (x,y,z) 
using Rays = Eigen::Matrix<Eigen::Vector3d, Eigen::Dynamic, Eigen::Dynamic>;
using Sensor = Eigen::Matrix<Eigen::Vector2d, Eigen::Dynamic, Eigen::Dynamic>;



class CameraMatrix: public Object {
	struct CamData {
		Eigen::Matrix3d* camMat{ nullptr };
		Eigen::VectorXd* distCoeffs{ nullptr };
	};
public:
	CameraMatrix() = default;
	CameraMatrix(
		const Eigen::Matrix3d& camMat,
		const Eigen::VectorXd& distCoeffs)
		: Object()
		, m_camMat{camMat}
		, m_camMat_inv{camMat.inverse()}
		, m_distCoeffs{distCoeffs}
	{
		check();
	}

	CameraMatrix(
		const Eigen::Matrix3d& camMat,
		const Eigen::VectorXd& distCoeffs,
		const Eigen::Matrix4d& pos)
		: Object(pos)
		, m_camMat{ camMat }
		, m_camMat_inv{camMat.inverse()}
		, m_distCoeffs{ distCoeffs }
	{
		check();
	}

	CameraMatrix(
		const cv::Mat& camMat,
		const cv::Mat& distCoeffs,
		const Eigen::Matrix4d& pos = Eigen::Matrix4d::Identity())
		: Object(pos)
	{
		if (camMat.size() != cv::Size(3, 3)) 
			throw std::invalid_argument("CameraMatrix wrong size");
		if (camMat.type() != CV_64F) 
			throw std::invalid_argument("CameraMatrix wrong type");
		if (!distCoeffs.empty() && distCoeffs.type() != CV_64F)
			throw std::invalid_argument("Distortion coefficients must use CV_64F");

		camMat.forEach<double>(
			[&](const double& value, const int* pos) -> void {
				m_camMat(pos[0], pos[1]) = value;
			}
		);

		m_camMat_inv = m_camMat.inverse();

		const Eigen::Index nDistCoeffs{
			static_cast<Eigen::Index>(distCoeffs.total())
		};
		m_distCoeffs.resize(nDistCoeffs);

		if (!distCoeffs.empty()) {
			Eigen::Index coeffIndex{ 0 };
			distCoeffs.forEach<double>(
				[this, &coeffIndex](const double& value, const int*) -> void {
					m_distCoeffs(coeffIndex++) = value;
				}
			);
		}
		check();
	}

	static Eigen::Matrix3d generateIntrinsicMatrix(
		const Protocoll::CameraData::ObjectiveData* objective,
		const double& f_number,
		const double& image_scale_goal_abs,
		const double& pixel_pitch,
		const double& px,
		const double& py,
		std::pair<int, int>& sensor_dimension = std::pair<int, int>(2464, 2056),
		const double& fx_fy_ratio = 1.0 /*fx/fy*/ )
	{
		if (fx_fy_ratio != 1.0) std::cout << "WARNING: Ratio fx to fy is != 0 \n";
		if (objective == nullptr) throw std::invalid_argument("Objective Data is nullptr");
		if (image_scale_goal_abs > 1.0 || image_scale_goal_abs <= 0.0) throw 
			std::invalid_argument("ImageScale must be a positive value smaller than 1.0");
		if (pixel_pitch <= 0.0) throw std::invalid_argument("Pixel Pitch must greater than zero");

		if (!objective->valid_Iris(f_number)) throw std::invalid_argument("F-Number not allowed");

		const double object_length{ (image_scale_goal_abs + 1) * objective->focal_length / image_scale_goal_abs };

		auto focusPoints = objective->generateFocusPoints(
			image_scale_goal_abs,
			f_number,
			pixel_pitch,
			object_length);
		
		if (focusPoints.FarPoint <= object_length || focusPoints.NearPoint >= object_length)
			throw std::invalid_argument("Object lies outside the focus area");

		// Debugging
		/*{
			std::cout << "Object to Hauptebene: " << object_length << '\n';
			std::cout << "Nahpunkt: " << focusPoints.NearPoint << '\n';
			std::cout << "Fernpunkt: " << focusPoints.FarPoint << '\n';
		}*/
		
		const double normalized_focal{
			objective->focal_length / pixel_pitch
		};

		const double normalized_fx{
			normalized_focal * fx_fy_ratio
		};

		Eigen::Matrix3d intrinsic =
			(Eigen::Matrix3d() <<
				normalized_fx, 0.0, px,
				0.0, normalized_focal, py,
				0.0, 0.0, 1.0).finished();

		return intrinsic;
	}


	static bool isSparseIntrinsic(const Eigen::Matrix3d& K) noexcept {
		// Define the expected non-zero pattern (true where elements CAN be set)
		const Eigen::Matrix<bool, 3, 3> mask{
			{false, true, false},	
			{true, false, false},
			{true, true, false}
		};

		// Check that every element where mask is false is exactly 0.0
		return (mask).select(K, 0.0).isZero(0.0);
	}


	// Creates Random Translation matrices for a given Intrinsic Matrix and Distortion Coefficients
	static std::vector<Eigen::Matrix4d> generateTranslationMatrixForCalibrationBoard_SyntheticCalibration(
		const Eigen::Matrix3d& intrinsic_matrix,
		const Eigen::Vector6d& distortion_Coefficients,
		const double& camera_pixel_pitch,
		const std::pair<int, int> pixel_xy,
		const std::pair<int, int> pattern_x_y,
		const double& pattern_width,
		const std::size_t n_positions)
	{
		/*if (intrinsic_matrix(0, 0) <= 0.0 || intrinsic_matrix(1, 1) <= 0.0)
			throw std::invalid_argument("The focus length must be bigger than 0");*/

		if (isSparseIntrinsic(intrinsic_matrix))
			throw std::invalid_argument("Only fx fy px and py must be set ");

		const bool has_distortion{ distortion_Coefficients.isZero() };

		if (camera_pixel_pitch <= 0)
			throw std::invalid_argument("Pixel_pitch must be bigger than 0");
		if (pattern_x_y.first <= 0 || pattern_x_y.second <= 0)
			throw std::invalid_argument("Calibration Board size must be bigger than 0");
		if (pattern_width <= 0)
			throw std::invalid_argument("Pattern size must be bigger than zero");
		if (n_positions <= 0)
			throw std::invalid_argument("...");
		
		std::size_t n_translation{};

		std::vector<Eigen::Matrix4d> translationMatrices(n_positions);

		while (true) {

		}

		


	}



	static Sensor generateSensorCoords(
		const double x_start = 0.0,
		const double y_start = 0.0,
		const double width = 2464.0,
		const double height = 2056.0,
		const double stepwidth_x = 1.0,
		const double stepwidth_y = 1.0)
	{
		if (width <= 0.0 || height <= 0.0) {
			throw std::invalid_argument("Width and height must be positive");
		}

		if (stepwidth_x <= 0.0 || stepwidth_y <= 0.0) {
			throw std::invalid_argument("Step widths must be positive");
		}

		if (x_start < 0.0 || x_start >= width ||
			y_start < 0.0 || y_start >= height) {
			throw std::invalid_argument("Sensor start coordinate lies outside the sensor");
		}

		const Eigen::Index rows = static_cast<Eigen::Index>(
			std::ceil((height - y_start) / stepwidth_y)
			);

		const Eigen::Index cols = static_cast<Eigen::Index>(
			std::ceil((width - x_start) / stepwidth_x)
			);

		Sensor sensor_coords(rows, cols);

		for (Eigen::Index row = 0; row < rows; ++row) {
			for (Eigen::Index col = 0; col < cols; ++col) {
				const double u =
					x_start + static_cast<double>(col) * stepwidth_x;

				const double v =
					y_start + static_cast<double>(row) * stepwidth_y;

				sensor_coords(row, col) = Eigen::Vector2d{ u, v };
			}
		}

		return sensor_coords;
	}

	virtual void castRays(
		Rays& rays,
		const Sensor*,
		const SamplingSetting& sample) const = 0;
	
	virtual ~CameraMatrix() = default;

	CamData getCameraData() {
		return CamData{
			{&m_camMat},
			{&m_distCoeffs}
		};
	}

protected:
	Eigen::Matrix3d m_camMat{};
	Eigen::Matrix3d m_camMat_inv{};
	Eigen::VectorXd m_distCoeffs{};

private:
	void check() {
		if (m_distCoeffs.size() > 5) {
			std::cout << "Warning: More than 5 Distortion Coefficients \n";
			std::cout << m_distCoeffs.size() << " - Coefficients provided \n";
			m_distCoeffs.conservativeResize(5);
		}
		if (m_distCoeffs.size() < 5) {
			std::cout << "Warning: Less than 5 Distortion Coefficients provided \n";
			const Eigen::Index wanted_length{ 5 };
			const Eigen::Index current_length{ m_distCoeffs.size() };
			m_distCoeffs.conservativeResize(wanted_length);
			m_distCoeffs.tail(wanted_length - current_length).setZero();
		}
	}
};


class OpenCvMatrix : public CameraMatrix {
public:
	explicit OpenCvMatrix(
		const cv::Mat& camMat,
		const cv::Mat& distCoeffs)
		:CameraMatrix(
			camMat,
			distCoeffs)
	{}
	
	// Return normalized direction Vectors for each pixel 
	void castRays(
		Rays& rays,
		const Sensor* sensor_coords,
		const SamplingSetting& sample) const override {
		if (sensor_coords == nullptr)
			throw std::invalid_argument("Sensor pointer must not be nullptr");

		if (sensor_coords->size() == 0) throw std::runtime_error("The SensorCoords must be Available");

		if (rays.rows() != sensor_coords->rows() * sample.samples_y ||
			rays.cols() != sensor_coords->cols() * sample.samples_x)
		{
			rays.resize(
				sensor_coords->rows() * sample.samples_y,
				sensor_coords->cols() * sample.samples_x
			);
		}

		Sensor sensor_Sampling{generateSensorCoords(
			0.0,
			0.0,
			2464.0,
			2056.0,
			1.0/sample.samples_x,
			1.0/sample.samples_y)
		};

		rays = sensor_Sampling.unaryExpr(
			[this](const Eigen::Vector2d& coords) -> Eigen::Vector3d
			{
				Eigen::Vector3d homogeneousCoords(coords[0], coords[1], 1.0);
				Eigen::Vector3d camera_coord{ m_camMat_inv * homogeneousCoords };
				//return camera_coord;
				Eigen::Vector3d distorted{ newtonSolverdistort(camera_coord, m_distCoeffs) };
				return distorted.normalized();
			}
		);
	}

	static Eigen::Vector3d undistort(
		const Eigen::Vector3d& image_pt,
		const Eigen::VectorXd& dist_Coeffs)
	{
		if (std::abs(image_pt.z()) < 1e-15)
			throw std::invalid_argument("Cannot undistort homogeneous point with z == 0");

		const Eigen::Vector3d normalized{
			image_pt.x() / image_pt.z(),
			image_pt.y() / image_pt.z(),
			1.0
		};

		return newtonSolverdistort(normalized, dist_Coeffs);
	}

	static Eigen::Vector3d newtonSolverdistort(
		const Eigen::Vector3d& pixelCoords,   // (u_d, v_d) verzerrt in Pixeln
		const Eigen::VectorXd& dist_coeffs)
	{
		const double* dptr = dist_coeffs.data();
		const double k1 = (dist_coeffs.size() > 0) ? dptr[0] : 0.0;
		const double k2 = (dist_coeffs.size() > 1) ? dptr[1] : 0.0;
		const double p1 = (dist_coeffs.size() > 2) ? dptr[2] : 0.0;
		const double p2 = (dist_coeffs.size() > 3) ? dptr[3] : 0.0;
		const double k3 = (dist_coeffs.size() > 4) ? dptr[4] : 0.0;

		if (dist_coeffs.isZero()) return pixelCoords;

		// --- Distorted pixel coordinates (u_d, v_d) ---
		const double x_d = pixelCoords[0];
		const double y_d = pixelCoords[1];


		// --- Initial guess for undistorted normalized coordinates (x_u, y_u) ---
		double x_u = x_d;
		double y_u = y_d;

		// --- Newton iteration ---
		for (int iter = 0; iter < 1000; ++iter) {
			// Radius
			const double r2 = x_u * x_u + y_u * y_u;
			const double r4 = r2 * r2;
			const double r6 = r4 * r2;

			// Radial term
			const double L = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
			const double dLdr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;
			const double Lx = 2.0 * x_u * dLdr2;
			const double Ly = 2.0 * y_u * dLdr2;

			// Tangential distortion
			const double tx = 2.0 * p1 * x_u * y_u + p2 * (r2 + 2.0 * x_u * x_u);
			const double ty = p1 * (r2 + 2.0 * y_u * y_u) + 2.0 * p2 * x_u * y_u;

			// Tangential derivatives
			const double dtxdx = 2.0 * p1 * y_u + 6.0 * p2 * x_u;
			const double dtxdy = 2.0 * p1 * x_u + 2.0 * p2 * y_u;
			const double dtydx = 2.0 * p1 * x_u + 2.0 * p2 * y_u;
			const double dtydy = 6.0 * p1 * y_u + 2.0 * p2 * x_u;

			// Forward distortion of current (x_u, y_u)
			const double fx = x_u * L + tx;
			const double fy = y_u * L + ty;

			// Residual: f(x_u, y_u) - (x_d, y_d) = 0
			const double R_x = fx - x_d;
			const double R_y = fy - y_d;

			// Jacobian matrix entries
			const double a = L + x_u * Lx + dtxdx;   // df_x / dx_u
			const double b = x_u * Ly + dtxdy;       // df_x / dy_u
			const double c = y_u * Lx + dtydx;       // df_y / dx_u
			const double d = L + y_u * Ly + dtydy;   // df_y / dy_u

			const double det = a * d - b * c;
			if (std::abs(det) < 1e-12) {
				//std::cout << "Reached termination cirterium \n";
				break; // numerisch instabil -> abbrechen
			}

			const double dx = (d * R_x - b * R_y) / det;
			const double dy = (-c * R_x + a * R_y) / det;

			x_u -= dx;
			y_u -= dy;

			if (dx * dx + dy * dy < 1e-18) {
				//std::cout << "Reached termination cirterium \n";
				break; // konvergiert
			}
		}
		return Eigen::Vector3d(x_u, y_u, 1.0);
	}
};

class ManualMatrix : public CameraMatrix {
public:
	ManualMatrix(
		const Eigen::Matrix3d& camMat,
		const Eigen::VectorXd& distCoeffs)
		: CameraMatrix(
			camMat,
			distCoeffs) 
	{ }

	void castRays(
		Rays& rays,
		const Sensor* sensor_coord,
		const SamplingSetting& sample) const override {
		throw std::runtime_error("Not implemented for Manual Camera Matrix");
	}

	// Here we need a full creation of the Manual CameraMatrix - but not today
};


#endif