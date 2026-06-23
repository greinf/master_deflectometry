#ifndef GRAYCALIBRATIONCONFIG_HPP
#define GRAYCALIBRATIONCONFIG_HPP

#include <array>
#include <string>  
#include <opencv2/opencv.hpp>

namespace gr_calib {
    struct NoCalibration {};
    struct ActiveCalibration {};
    struct PassiveCalibration {};
    struct LUTCalibration {};
}


namespace defl {
    struct GrayCalibrationConfig {
        std::string method_name;

        template<typename T>
        typename std::enable_if<std::is_class<T>::value, void>::type
        assign_method(T) {
            method_name = method(T{});
        }


        save_to_xml(const std::string& path) {
            try {
                cv::FileStorage fs(path, cv::FileStorage::WRITE);
                
                if (!fs.isOpened()) {
                    std::cerr << "Error: Could not open file for writing: " << path << "\n";
                    return false;
                }

                fs << "GrayValueCalibration " << "{";
                fs << "Method " << method_name;

                // Fill in here 

                fs << "}";
                fs.release();
                return true;
            }
            catch (const cv::Exception& e) {
                std::cerr << "OpenCV Exception in save_to_XML: " << e.what() << "\n";
                return false;
            }
        }

    private:
        
        static constexpr const char* method(gr_calib::NoCalibration) const { return method_names[0]; }
        static constexpr const char* method(gr_calib::ActiveCalibration) const { return method_names[1];}
        static constexpr const char* method(gr_calib::PassiveCalibration) const { return method_names[2]; }
        static constexpr const char* method(gr_calib::LUTCalibration) const { return method_names[3]; }

        static constexpr std::array<const char*, 4> method_names{
            "No Calibration",
            "Active Calibration",
            "Passive Calibration",
            "LUT Calibration"
        };
    };
}


#endif