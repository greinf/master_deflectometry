#ifndef CAMERACONFIG_H
#define CAMERACONFIG_H
#include <chrono>
#include <string>


namespace defl {

    struct CameraConfig {
        // --- Core Parameter ---
        int pixel_x = 0;
        int pixel_y = 0;
        double exposure_time = 0.0;
        double frame_rate = 0.0;
        double gain = 0.0;
        bool acquisition_mode_active{ false };

        // --- Optional Parameter ---
        std::size_t active_device_index = { 0 };
        std::string model_name{};
        std::string version{};
        std::string parent_system{};

        std::chrono::system_clock::time_point timestamp;

        CameraConfig(int pix_x, int pix_y, double expo_time, double framerate, double gain) {
            timestamp = std::chrono::system_clock::now();
        }

        CameraConfig() {
            timestamp = std::chrono::system_clock::now();
        }

        
        
        void print() {
            std::cout << "----- Camera Config ----- \n" <<
                "Pixel x: " << pixel_x << '\n' <<
                "Pixel y: " << pixel_y << '\n' <<
                "Exposure Time: " << exposure_time << '\n' <<
                "Frame Rate: " << frame_rate << '\n' <<
                "Gain: " << gain << '\n' <<
                "Acquisition mode " << (acquisition_mode_active ? "is active\n" : "is not active \n") <<
                "Model Name: " << model_name << '\n' <<
                "Version: " << version << '\n' <<
                "Parent: " << parent_system << '\n' <<
                "----------------------------\n";
        
        }
    };

} // namespace defl

#endif