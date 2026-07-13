#ifndef CAMERAMATRIX_HPP
#define CAMERAMATRIX_HPP

#include <Eigen/dense>
#include <opencv2/opencv.hpp>
#include <algorithm>

// Everything is stored in (x,y,z) 
using Rays = Eigen::Matrix<Eigen::Vector3d, Eigen::Dynamic, Eigen::Dynamic>;
using Sensor = Eigen::Matrix<Eigen::Vector2d, Eigen::Dynamic, Eigen::Dynamic>;

class CameraMatrix {
public:
	CameraMatrix() = default;
	CameraMatrix(
		const Eigen::Matrix3d& camMat,
		const Eigen::VectorXd& distCoeffs)
		: m_camMat{camMat}
		, m_camMat_inv{camMat.inverse()}
		, m_distCoeffs{distCoeffs}
	{
		check();
	}

	CameraMatrix(
		const Eigen::Matrix3d& camMat,
		const Eigen::VectorXd& distCoeffs,
		const Eigen::Matrix4d& pos)
		: m_camMat{ camMat }
		, m_camMat_inv{camMat.inverse()}
		, m_distCoeffs{ distCoeffs }
	{
		check();
	}

	CameraMatrix(
		const cv::Mat& camMat,
		const cv::Mat& distCoeffs,
		const Eigen::Matrix4d& pos = Eigen::Matrix4d::Identity())
	{
		if (camMat.size() != cv::Size(3, 3)) 
			throw std::invalid_argument("CameraMatrix wrong size");
		if (camMat.type() != CV_64F) 
			throw std::invalid_argument("CameraMatrix wrong type");

		camMat.forEach<double>(
			[&](const double& value, const int* pos) -> void {
				m_camMat(pos[0], pos[1]) = value;
			}
		);

		m_camMat_inv = m_camMat.inverse();

		m_distCoeffs.resize(
			static_cast<Eigen::Index>(std::max(distCoeffs.size[0], distCoeffs.size[1])),
			Eigen::NoChange
		);

		distCoeffs.forEach<double>(
			[this](const double& value, const int* pos) -> void {
				m_distCoeffs(pos[0], pos[1]) = value;
			}
		);
		check();
	}

	static Sensor generateSensorCoords(
		const double x_start = 0.0,
		const double y_start = 0.0,
		const double width = 2464.0,
		const double height = 2056 ,
		const double stepwidth_x = 1.0,
		const double stepwidth_y = 1.0)
	{
		if (width < 0 || height < 0) throw std::invalid_argument("Width and height must be > 0");
		if (stepwidth_x > 9 * width) throw std::invalid_argument("stepwith_x to high");
		if (stepwidth_y > 9 * height) throw std::invalid_argument("stepthwidth_y to high");

		const Eigen::Index rows{ 
			static_cast<Eigen::Index>(std::round(width / stepwidth_x))};
		const Eigen::Index cols{ 
			static_cast<Eigen::Index>(std::round(height / stepwidth_y))};

		Sensor sensor_coords{};
		sensor_coords.resize(rows, cols);

		for (Eigen::Index row = static_cast<Eigen::Index>(x_start);
			row < rows; ++row) {
			for (Eigen::Index col = static_cast<Eigen::Index>(y_start);
				col < cols; ++col) {
				sensor_coords(row, col) = { row, col };
			}
		}
		
		return sensor_coords;
	}

	virtual void castRays(
		Rays& rays,
		const Sensor*) const = 0;
	
	virtual ~CameraMatrix() = default;

protected:
	Eigen::Matrix3d m_camMat{};
	Eigen::Matrix3d m_camMat_inv{};
	Eigen::VectorXd m_distCoeffs{};

private:
	void check() {
		if (m_distCoeffs.size() > 5) {
			std::cout << "Warning: More than 5 Distortion Coefficients \n";
			std::cout << m_distCoeffs.size() << " - Coefficients provided \n";
		}
		if (m_distCoeffs.size() < 4) {
			std::cout << "Warning: Less than 4 Distortion Coefficient provided \n";
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
	
	void castRays(
		Rays& rays,
		const Sensor* sensor_coords) const override {
		if (sensor_coords->size() == 0) throw std::runtime_error("The SensorCoords must be Available");
		if (rays.size() == 0) rays.resize(sensor_coords->rows(), sensor_coords->cols());

		rays = sensor_coords->unaryExpr(
			[this](const Eigen::Vector2d& coords) -> Eigen::Vector3d
			{
				Eigen::Vector3d homogeneousCoords(coords[0], coords[1], 1.0);
				Eigen::Vector3d camera_coord{ m_camMat_inv * homogeneousCoords };
				//return camera_coord;
				return newtonSolverdistort(camera_coord, m_distCoeffs);
			}
		);
	}

	static Eigen::Vector3d newtonSolverdistort(
		const Eigen::Vector3d& pixelCoords,   // (u_d, v_d) verzerrt in Pixeln
		const Eigen::VectorXd& dist_coeffs)
	{
		const double* dptr = dist_coeffs.data();
		const double k1 = dptr[0];
		const double k2 = dptr[1];
		const double p1 = dptr[2];
		const double p2 = dptr[3];
		const double k3 = (dist_coeffs.size() > 4) ? dptr[4] : 0.0;

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
		const Sensor* sensor_coord) const override {
		throw std::runtime_error("Not implemented for Manual Camera Matrix");
	}

	// Here we need a full creation of the Manual CameraMatrix - but not today
};


#endif