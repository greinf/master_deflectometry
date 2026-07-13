#ifndef CAMERA_HPP
#define CAMERA_HPP
#include "CameraMatrix.hpp"
#include "Object.hpp"

#include <Eigen/dense>
#include <memory>

class Camera: public Object {
public:
	Camera(
		std::unique_ptr<CameraMatrix>&& camMat,
		Sensor& sensor_coords = CameraMatrix::generateSensorCoords())
		: Object()
		, m_cameraMatrix{std::move(camMat)}
		, m_sensorCoords{std::make_unique<Sensor>(std::move(sensor_coords))}
	{}

	Camera(
		std::unique_ptr<CameraMatrix>&& camMat,
		std::unique_ptr<Sensor>&& sensor_coords)
		: Object()
		, m_cameraMatrix{ std::move(camMat) }
		, m_sensorCoords{ std::move(sensor_coords) }
	{}

	Camera(
		std::unique_ptr<CameraMatrix>&& camMat,
		const cv::Mat_<cv::Vec2d>& sensor)
		: Object()
		, m_cameraMatrix{ std::move(camMat) }
	{
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
	
	void generateRays(Rays& rays) const {
		m_cameraMatrix->castRays(rays, m_sensorCoords.get());
	}

	const Eigen::Matrix4d& getTransform() const {
		return m_transform;
	}

	std::unique_ptr<Sensor> m_sensorCoords{ nullptr };

private:
	std::unique_ptr<CameraMatrix> m_cameraMatrix{ nullptr };
};



#endif // "CAMERA_HPP"