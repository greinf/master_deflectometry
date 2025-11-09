#ifndef CAMERA_H
#define CAMERA_H

#include <cstddef>
#include <iostream>
#include <memory>
#include <peak/peak.hpp>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <string>
#include <limits>
#include <cctype>
#include <array>

class Screen;

class Camera
{
public:
	Camera();
	Camera(const Camera&) = delete;
	Camera& operator=(const Camera&) = delete;
	Camera(Camera&&) = delete;
	Camera& operator=(Camera&&) = delete;
	~Camera();
	
	void setUpAcquisition(std::size_t i = 0);

	void grayValueCalibration(std::shared_ptr<Screen>);
	std::vector<std::shared_ptr<peak::core::Device>> m_device; //normally only one camera is used
	std::vector<std::shared_ptr<peak::core::DataStream>> m_dataStream;
	std::vector<std::shared_ptr<peak::core::NodeMap>> m_nodemapRemoteDevice;
	bool adjustSettings(std::size_t i);

private:
	bool PrepareAcquisition(std::size_t i=0);
	bool SetRoi(std::int64_t x, std::int64_t y, std::int64_t width, std::int64_t height, std::size_t i=0);
	bool AllocAndAnnounceBuffers(std::size_t i = 0 );
	bool StartAcquisition(std::size_t i = 0);

};

#endif