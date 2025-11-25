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
#include <vector>
#include <config/CameraConfig.hpp>


class Camera
{
public:
    // optional: explizit Device-Index angeben, default = 0
    explicit Camera(std::size_t preferredIndex = 0);
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;
    Camera(Camera&&) = delete;
    Camera& operator=(Camera&&) = delete;
    ~Camera();

    // richtet ROI, Buffer, Acquisition ein
    void setUpAcquisition(std::size_t streamIndex = 0);

    // Einstellungen (Exposure/Gain/Framerate)
    bool adjustSettings(std::size_t deviceIndex);

    bool isacquisitionRunning(std::size_t deviceIndex = 0); 

    // NEU: sauberer Zugriff für AcquisitionWorker via Composition ===
    std::shared_ptr<peak::core::DataStream> dataStream(std::size_t i = 0) const;
    std::shared_ptr<peak::core::NodeMap> nodeMap(std::size_t i = 0) const;
    std::shared_ptr<peak::core::Device> device(std::size_t i = 0) const;

    
    std::vector<std::shared_ptr<defl::CameraConfig>> m_camera_data;

    std::vector<std::shared_ptr<defl::CameraConfig>>
        getCameraConfig() { return m_camera_data; }

private:
    std::size_t deviceCount() const noexcept { return m_devices.size(); }
    void logging_data(std::size_t cameraIndex);
    std::vector<std::shared_ptr<defl::CameraConfig>>::iterator 
        check_for_config(std::size_t index);

    bool PrepareAcquisition(std::size_t i = 0);
    bool SetRoi(std::int64_t x, std::int64_t y,
        std::int64_t width, std::int64_t height,
        std::size_t i = 0);
    bool AllocAndAnnounceBuffers(std::size_t i = 0);
    bool StartAcquisition(std::size_t i = 0);

    // NEU: gekapselt ===
    std::vector<std::shared_ptr<peak::core::Device>>   m_devices;
    std::vector<std::shared_ptr<peak::core::DataStream>> m_dataStreams;
    std::vector<std::shared_ptr<peak::core::NodeMap>>  m_nodeMaps;

    // Welches Device ist „aktiv“
    std::size_t m_activeDeviceIndex{ 0 };
};

#endif
