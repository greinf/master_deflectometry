#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "CameraMatrix.hpp"
#include <memory>
#include <Eigen/dense>

class Camera {
public:
	Camera(
		std::unique_ptr<CameraMatrix>&& camMat,
		Sensor sensor_coords = CameraMatrix::generateSensorCoords(),
		const SamplingSetting setting = {})
		: m_cameraMatrix{std::move(camMat)}
		, m_sensorCoords{std::make_unique<Sensor>(std::move(sensor_coords))}
		, m_setting{setting}
	{
		if (m_cameraMatrix == nullptr)
			throw std::invalid_argument("Camera received nullptr CameraMatrix");
	}

	Camera(
		std::unique_ptr<CameraMatrix>&& camMat,
		std::unique_ptr<Sensor>&& sensor_coords,
		const SamplingSetting setting = {})
		: m_cameraMatrix{ std::move(camMat) }
		, m_sensorCoords{ std::move(sensor_coords) }
		, m_setting{setting}
	{
		if (m_cameraMatrix == nullptr || m_sensorCoords == nullptr)
			throw std::invalid_argument("Camera received nullptr input");
	}

	Camera(
		std::unique_ptr<CameraMatrix>&& camMat,
		const SamplingSetting setting = {})
		: m_cameraMatrix{ std::move(camMat) }
		, m_setting{ setting }
	{
		m_sensorCoords = std::make_unique<Sensor>(CameraMatrix::generateSensorCoords());
		if (m_cameraMatrix == nullptr)
			throw std::invalid_argument("Camera received nullptr input");
	}

	Camera(
		std::unique_ptr<CameraMatrix>&& camMat,
		const cv::Mat_<cv::Vec2d>& sensor,
		const SamplingSetting setting = {})
		: m_cameraMatrix{ std::move(camMat) }
		, m_setting{setting}
	{
		if (m_cameraMatrix == nullptr)
			throw std::invalid_argument("Camera received nullptr CameraMatrix");

		m_sensorCoords = std::make_unique<Sensor>();
		m_sensorCoords->resize(
			static_cast<Eigen::Index>(sensor.rows),
			static_cast<Eigen::Index>(sensor.cols));

		for (int row = 0; row < sensor.rows; ++row)
		{
			for (int col = 0; col < sensor.cols; ++col)
			{
				const cv::Vec2d& coord = sensor(row, col);

				(*m_sensorCoords)(
					static_cast<Eigen::Index>(row),
					static_cast<Eigen::Index>(col)) =
					Eigen::Vector2d{
						coord[0],
						coord[1]
				};
			}
		}
	}
	
	[[nodiscard]] void generateRays(Rays& rays) const {
		m_cameraMatrix->castRays(rays, m_sensorCoords.get(), m_setting);
	}

	[[nodiscard]] const Eigen::Matrix4d& getTransform() const {
		return m_cameraMatrix->m_transform;
	}

	std::unique_ptr<Sensor> m_sensorCoords{ nullptr };

	SamplingSetting m_setting{};
private:
	std::unique_ptr<CameraMatrix> m_cameraMatrix{ nullptr };
};



#endif // CAMERA_HPP