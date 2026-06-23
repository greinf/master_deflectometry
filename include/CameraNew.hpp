#ifndef CAMERANEW_HPP
#define CAMERANEW_HPP

#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "ICameraBackend.hpp"
#include "config/CameraConfig.hpp"


class CameraN {
public:
    enum class Backend { IDS, VIMBA };

    explicit CameraN(Backend b);

    // Does the setup for to get the camera in Acuqisition mode
    bool open(std::size_t i = 0) { return m_backend->open(i); };

    // Puts the camera out of the Acuqisition setup.  
    void close(std::size_t i = 0) { m_backend->close(i); }
    
    // If camera is in Acquisition mode grabs frames from the buffer.
    std::vector<cv::Mat> grab(int timeout_ms = 5000);

    // Checks for active acquisition 
    bool isRunning(std::size_t i = 0) { return m_backend->isRunning(i); };

    std::vector<defl::CameraConfig> getCamConfig() {
        return m_backend->getCameraConfig();
    }

    CameraN& operator=(const CameraN&) = delete;
    CameraN& operator=(CameraN&&) = delete;
    CameraN(const CameraN&) = delete;
    CameraN(CameraN&&) = delete;

    
private:
    std::unique_ptr<ICameraBackend> m_backend;
};

#endif
