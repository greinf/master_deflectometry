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
#include <Eigen/Dense>


// Everything is stored in (x,y,z) 
using Rays = Eigen::Matrix<Eigen::Vector3d, Eigen::Dynamic, Eigen::Dynamic>;
using Sensor = Eigen::Matrix<Eigen::Vector2d, Eigen::Dynamic, Eigen::Dynamic>;



class CameraMatrix: public Object {
	struct CamData {
		const Eigen::Matrix3d* camMat{ nullptr };
		const Eigen::VectorXd* distCoeffs{ nullptr };
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
			if (distCoeffs.rows != 1 && distCoeffs.cols != 1)
				throw std::invalid_argument("Distortion must be a vector");
			const cv::Mat flat = distCoeffs.clone().reshape(1, 1);
			for (Eigen::Index i = 0; i < nDistCoeffs; ++i)
				m_distCoeffs(i) = flat.at<double>(0, static_cast<int>(i));
		}
		check();
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

	static Sensor generateSensorCoords(
		const double x_start = 0.0,
		const double y_start = 0.0,
		const double width = 2464.0,
		const double height = 2056.0,
		const double stepwidth_x = 1.0,
		const double stepwidth_y = 1.0)
	{
		if (!std::isfinite(x_start) || !std::isfinite(y_start) ||
			!std::isfinite(width) || !std::isfinite(height) ||
			!std::isfinite(stepwidth_x) || !std::isfinite(stepwidth_y))
			throw std::invalid_argument("Non-finite sensor coordinates");
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

		const double row_count = std::ceil((height - y_start) / stepwidth_y);
		const double col_count = std::ceil((width - x_start) / stepwidth_x);
		const double max_count = static_cast<double>(std::numeric_limits<Eigen::Index>::max());
		if (!std::isfinite(row_count) || !std::isfinite(col_count) ||
			row_count >= max_count || col_count >= max_count ||
			row_count * col_count >= max_count / sizeof(Eigen::Vector2d))
			throw std::overflow_error("Sensor coordinate array size overflow");

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

	CamData getCameraData() const {
		return CamData{
			{&m_camMat},
			{&m_distCoeffs}
		};
	}

protected:
	Eigen::Matrix3d m_camMat = Eigen::Matrix3d::Zero();
	Eigen::Matrix3d m_camMat_inv = Eigen::Matrix3d::Zero();
	Eigen::VectorXd m_distCoeffs{};

private:
	void check() {
		if (!m_camMat.allFinite() || !isSparseIntrinsic(m_camMat) ||
			m_camMat(0, 0) <= 0 || m_camMat(1, 1) <= 0 || std::abs(m_camMat(2, 2) - 1) > 1e-12)
			throw std::invalid_argument("Invalid camera matrix");
		m_camMat_inv = m_camMat.inverse();
		if (!m_camMat_inv.allFinite() || !m_distCoeffs.allFinite() ||
			(m_distCoeffs.size() != 0 && m_distCoeffs.size() != 4 && m_distCoeffs.size() != 5))
			throw std::invalid_argument("Only empty/4/5 coefficient Brown vectors are supported");
		if (m_distCoeffs.size() < 5) {
			const Eigen::Index wanted_length{ 5 };
			const Eigen::Index current_length{ m_distCoeffs.size() };
			m_distCoeffs.conservativeResize(wanted_length);
			m_distCoeffs.tail(wanted_length - current_length).setZero();
		}
	}
};


class OpenCvMatrix : public CameraMatrix {
public:
	OpenCvMatrix(
		const cv::Mat& camMat,
		const cv::Mat& distCoeffs)
		:CameraMatrix(
			camMat,
			distCoeffs)
	{ }
	
	OpenCvMatrix(
		const Eigen::Matrix3d& camMat,
		const Eigen::VectorXd& distCoeffs)
		:CameraMatrix(
			camMat,
			distCoeffs)
	{ }
	
	// Return normalized direction Vectors for each pixel 
	void castRays(Rays& rays, const Sensor* sensor, const SamplingSetting& sample) const override {
		if (!sensor || sensor->size() == 0 || sample.samples_x == 0 || sample.samples_y == 0)
			throw std::invalid_argument("Empty sensor or zero sampling");
		const auto limit = static_cast<std::uint64_t>(std::numeric_limits<Eigen::Index>::max());
		if (sample.samples_x > limit / static_cast<std::uint64_t>(sensor->cols()) ||
			sample.samples_y > limit / static_cast<std::uint64_t>(sensor->rows()))
			throw std::overflow_error("Ray array dimensions overflow");
		const Eigen::Index sx = static_cast<Eigen::Index>(sample.samples_x);
		const Eigen::Index sy = static_cast<Eigen::Index>(sample.samples_y);
		const Eigen::Index rows = sensor->rows() * sy, cols = sensor->cols() * sx;
		if (static_cast<std::uint64_t>(rows) > limit / static_cast<std::uint64_t>(cols) / sizeof(Eigen::Vector3d))
			throw std::overflow_error("Ray array storage overflow");
		rays.resize(rows, cols);
		// Integer coordinates denote pixel CENTERS. Each sensor entry describes
		// a unit-sized pixel in K's coordinate system (ROI offsets are allowed).
		for (Eigen::Index col = 0; col < sensor->cols(); ++col)
			for (Eigen::Index row = 0; row < sensor->rows(); ++row) {
				const auto& center = (*sensor)(row, col);
				if (!center.allFinite()) throw std::invalid_argument("Non-finite sensor coordinate");
				for (Eigen::Index ax = 0; ax < sx; ++ax)
					for (Eigen::Index ay = 0; ay < sy; ++ay) {
						const double u = center.x() + (ax + 0.5) / sx - 0.5;
						const double v = center.y() + (ay + 0.5) / sy - 0.5;
						const Eigen::Vector3d q = m_camMat_inv * Eigen::Vector3d(u, v, 1.0);
						rays(row * sy + ay, col * sx + ax) = newtonSolverdistort(q, m_distCoeffs).normalized();
					}
			}
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
		const Eigen::Vector3d& pixelCoords,   // distorted NORMALIZED coordinates (xd, yd, 1)
		const Eigen::VectorXd& dist_coeffs)
	{
		if (!pixelCoords.allFinite() || !dist_coeffs.allFinite() ||
			(dist_coeffs.size() != 0 && dist_coeffs.size() != 4 && dist_coeffs.size() != 5))
			throw std::invalid_argument("Invalid inverse distortion input");
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
		const double r2 = x_u * x_u + y_u * y_u;
		const double radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
		const double ex = x_u * radial + 2 * p1 * x_u * y_u + p2 * (r2 + 2 * x_u * x_u) - x_d;
		const double ey = y_u * radial + p1 * (r2 + 2 * y_u * y_u) + 2 * p2 * x_u * y_u - y_d;
		if (!std::isfinite(ex) || !std::isfinite(ey) || ex * ex + ey * ey > 1e-20)
			throw std::runtime_error("Inverse Brown distortion failed to converge");
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
