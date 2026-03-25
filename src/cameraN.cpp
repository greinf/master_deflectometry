#include "CameraNew.hpp"
#include <iostream>
#include <stdexcept>

//#include "camera.hpp"          // IDS backend (Camera : ICameraBackend)
#include "VimbaBackend.hpp"    // Vimba backend (VimbaBackend : ICameraBackend)

CameraN::CameraN(Backend b) {
    switch (b) {
    case Backend::IDS:
        // IDS backend was delted because the hole project was continued with Mako Cameras. 
        std::cout << "Ids Backend depracted \n";
        throw std::runtime_error("Not possible at this point");
       
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
