#ifndef CAMERA_H
#define CAMERA_H

#include <cstddef>
#include <iostream>
#include <memory>

#include <peak/peak.hpp>

class Camera
{
public:
	Camera();
	Camera(const Camera&) = delete;
	Camera& operator=(const Camera&) = delete;
	Camera(Camera&&) = default;
	Camera& operator=(Camera&&) = default;
	~Camera();
	
	void getFrames(std::size_t i=1);
private:
	std::vector<std::shared_ptr<peak::core::Device>> m_device; //normally only one camera is used
	std::vector<std::shared_ptr<peak::core::DataStream>> m_dataStream;
	std::vector<std::shared_ptr<peak::core::NodeMap>> m_nodemapRemoteDevice;
	bool PrepareAcuqisition(std::size_t i=1);
	bool SetRoi(std::int64_t x, std::int64_t y, std::int64_t width, std::int64_t height, std::size_t i);

};

#endif