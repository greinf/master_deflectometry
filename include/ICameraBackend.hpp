#ifndef ICAMERABACKEND
#define ICAMERABACKEND
#include <memory>
#include <iostream>
#include <vector>
#include <config/CameraConfig.hpp>
#include <opencv2/opencv.hpp>



class ICameraBackend {
public:
    
    virtual ~ICameraBackend() = default;
    // = 0 indication that in the base class the function is set to zero und must be overridden. 
    virtual bool open(std::size_t ) = 0;
    virtual void close(std::size_t) = 0;

    virtual bool isRunning(std::size_t) = 0;

    virtual std::vector<std::shared_ptr<defl::CameraConfig>> getCameraConfig() = 0;

    // Blocking grab. Returns empty on timeout/error.
    virtual cv::Mat grab(int timeout_ms) = 0;
};

#endif


