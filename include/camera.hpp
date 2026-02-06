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
#include "ICameraBackend.hpp"

// This class build the IDS Backend for the image Acquisition
// The class does inherit from the ICameraBackend which provide the necessary function
// to controll the cameras. 
class Camera: public ICameraBackend
{
public:
    // The class tries to connect to available cameras at the moment of initialisation
    // The argument std::size_t let´s the user choose if a ceratain camera should be used.
    // The agrument is defaultet with 0 since this class will be mostly used wiht a single cmaera.
    Camera(std::size_t preferredIndex = 0); //

    bool open(std::size_t preferredIndex = 0) override;
    bool isRunning(std::size_t i = 0) override { return isacquisitionRunning(i); }

    void close(std::size_t preferredIndex = 0) override;



    // --- Safety Checks for coping moing the class --- 
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;
    Camera(Camera&&) = delete;
    Camera& operator=(Camera&&) = delete;
    ~Camera() override;

    

    // Einstellungen (Exposure/Gain/Framerate)
    bool adjustSettings(std::size_t deviceIndex);

   

    // NEU: sauberer Zugriff für AcquisitionWorker via Composition ===
    std::shared_ptr<peak::core::DataStream> dataStream(std::size_t i = 0) const;
    std::shared_ptr<peak::core::NodeMap> nodeMap(std::size_t i = 0) const;
    std::shared_ptr<peak::core::Device> device(std::size_t i = 0) const;

    


    // return the hole camera config. This vector should normale have size() = 1 
    // If multiple camera from the same manufacturer are used this could be > 1.

    std::vector<defl::CameraConfig>
        getCameraConfig() override
    {
        std::vector<defl::CameraConfig> config_vec;
        for (const auto& cfg : m_camera_data) {
            // Creates a copy of each instance of m_camera_data (porpably only one)
            config_vec.emplace_back(*cfg);
        }
        return config_vec;
    }

    std::vector<cv::Mat> grab(int timeout);

private:
    // Does look in the according logging file if the camera "i" and return the "acquisition_mode_active" member
    bool isacquisitionRunning(std::size_t deviceIndex = 0);

    // richtet ROI, Buffer, Acquisition ein
    void setUpAcquisition(std::size_t streamIndex = 0);

    // Just return the ammount of available devices found by the IDS-peak
    std::size_t deviceCount() const noexcept { return m_devices.size(); }

    // Logging Data, does check for a camera config file.
    // If it does not exists it is created and all the data is locked in the file
    void logging_data(std::size_t cameraIndex);

    // Checks for a config file given the Camera Number (index) 
    // If this does not exist it gets created and a iterator to this element returned.
    // The Camera config has a member that "active Index", this is the numerator. 
    std::vector<std::shared_ptr<defl::CameraConfig>>::iterator 
        check_for_config(std::size_t index);

    // The datastream instance is created for the given camera int i
    bool PrepareAcquisition(std::size_t i = 0);

    // An ROI must be set before the Acquisition is start
    // The int value is there to use different found cameras
    bool SetRoi(std::int64_t x, std::int64_t y,
        std::int64_t width, std::int64_t height,
        std::size_t i = 0);
    
    // The ammount of Buffers needed for the given Datatype is set.
    // Method that is done when the acquisition is about to start.
    // datastream method needs to exist at this point
    // The int value is there to use different found cameras
    bool AllocAndAnnounceBuffers(std::size_t i = 0);

    // This method does take the exisiting Datastream, locks the Nodes and start the ->execute()
    // The int value is there to use different found cameras
    bool StartAcquisition(std::size_t i = 0);

    // NEU: gekapselt ===
    std::vector<std::shared_ptr<peak::core::Device>>   m_devices;
    std::vector<std::shared_ptr<peak::core::DataStream>> m_dataStreams;
    std::vector<std::shared_ptr<peak::core::NodeMap>>  m_nodeMaps;


    // Propably bad decission at the start but ok for now to make this shared ptr. 
    // The return value for when getting a config is copy of the instance and not a shared ptr
    std::vector<std::shared_ptr<defl::CameraConfig>> m_camera_data;

    // Welches Device ist „aktiv“
    std::size_t m_activeDeviceIndex{ 0 };
};

#endif
