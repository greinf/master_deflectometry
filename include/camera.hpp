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

private:
	std::vector<std::shared_ptr<peak::core::Device>> camera_ptr; //normally only one camera is used

};

#endif