#include "CameraNew.hpp"
#include <iostream>
#include <stdexcept>

#include "camera.hpp"          // IDS backend (Camera : ICameraBackend)
#include "VimbaBackend.hpp"    // Vimba backend (VimbaBackend : ICameraBackend)

CameraN::CameraN(Backend b) {
    switch (b) {
    case Backend::IDS:
        m_backend = std::make_unique<Camera>();
        std::cout << "Try to connect to IDS Backend\n";
        break;

    case Backend::VIMBA:
        m_backend = std::make_unique<VimbaBackend>();
        std::cout << "Try to connect to Vimba Backend\n";
        break;

    default:
        throw std::runtime_error("Unknown backend");
    }
}

std::vector<cv::Mat> CameraN::grab(int timeout_ms) {
    return m_backend->grab(timeout_ms);
}
