#ifndef PHASESHIFTCONFIG_H
#define PHASESHIFTCONFIG_H
#include <chrono>
#include <filesystem>


namespace defl {

    struct PhaseShiftConfig {
        // --- Core Parameter ---
        int pixel_x = 0;
        int pixel_y = 0;
        int steps = 0;               // number of shifts
        int periods_in_y = 0;        // number of periods in y direction
        double wavelength = 0.0;     // pixels per 2π
        double shift_length = 0.0;   // radians between shifts


        // --- Optional Parameter ---
        double amplitude = 127.5;
        double mean_value = 127.5;

        bool lut_available = false;
        std::vector<std::pair<double, double>> lut_data{};

        // --- Metadata ---
        std::string algorithm_name { "FourPhaseShift" };
        
        std::chrono::system_clock::time_point timestamp;

        // --- Constructor ---
        PhaseShiftConfig() {
            timestamp = std::chrono::system_clock::now();
        }

        void print() const;
        bool save_to_XML(const std::string& path) const;

        static PhaseShiftConfig load_from_XML(const std::string& path);

        friend std::ostream& operator<<(std::ostream& out, const defl::PhaseShiftConfig& conf) {
            conf.print();
            return out;
        }
    };

    void inline PhaseShiftConfig::print() const {
        std::cout << "----- PhaseShiftConfig -----\n"
            << "Resolution: " << pixel_x << " x " << pixel_y << "\n"
            << "Periods in Y: " << periods_in_y << "\n"
            << "Steps: " << steps << "\n"
            << "Wavelength: " << wavelength << "\n"
            << "Shift length: " << shift_length << "\n"
            << "Amplitude / Mean: " << amplitude << " / " << mean_value << "\n"
            << "LUT available: " << (lut_available ? "yes" : "no") << "\n"
            << "Algorithm: " << algorithm_name << "\n"
            << "----------------------------\n";
    }

    bool inline PhaseShiftConfig::save_to_XML(const std::string& path) const {
        try {
            cv::FileStorage fs(path, cv::FileStorage::WRITE);

            if (!fs.isOpened()) {
                std::cerr << "Error: Could not open file for writing: " << path << "\n";
                return false;
            }

            fs << "PhaseShiftConfig" << "{";
            fs << "pixel_x" << pixel_x;
            fs << "pixel_y" << pixel_y;
            fs << "steps" << steps;
            fs << "periods_in_y" << periods_in_y;
            fs << "wavelength" << wavelength;
            fs << "shift_length" << shift_length;
            fs << "amplitude" << amplitude;
            fs << "mean_value" << mean_value;
            fs << "lut_available" << lut_available;

            fs << "}";  // Ende des Blocks

            fs.release();
            return true;
        }
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV Exception in save_to_XML: " << e.what() << "\n";
            return false;
        }
    }


    inline static PhaseShiftConfig load_from_XML(const std::string& path) {
        PhaseShiftConfig cfg;

        try {
            cv::FileStorage fs(path, cv::FileStorage::READ);
            if (!fs.isOpened()) {
                throw std::runtime_error("Cannot open XML: " + path);
            }

            cv::FileNode n = fs["PhaseShiftConfig"];
            if (n.empty()) {
                throw std::runtime_error("XML does not contain PhaseShiftConfig node.");
            }

            n["pixel_x"] >> cfg.pixel_x;
            n["pixel_y"] >> cfg.pixel_y;
            n["steps"] >> cfg.steps;
            n["periods_in_y"] >> cfg.periods_in_y;
            n["wavelength"] >> cfg.wavelength;
            n["shift_length"] >> cfg.shift_length;
            n["amplitude"] >> cfg.amplitude;
            n["mean_value"] >> cfg.mean_value;
            n["lut_available"] >> cfg.lut_available;
            n["algorithm_name"] >> cfg.algorithm_name;

       

            return cfg;

        }
        catch (const std::exception& e) {
            std::cerr << "load_from_XML failed: " << e.what() << "\n";
            throw;  // let caller decide
        }
    }

} // namespace defl

#endif