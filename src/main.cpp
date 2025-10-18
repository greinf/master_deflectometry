#define VERSION "1"

#include <cstddef>
#include <iostream>
#include "deflectometry.hpp"
#include "enums.hpp"
#include <opencv2/opencv.hpp>


int main()
{
    //Supress open CV Information -only warnings are logged. 
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    Deflectometry meassure{};

    meassure.start_meassurement(Shift_mode::user_defined, DisplayMode::Automatic, 0);

    
    meassure.show_acquistion();
    

    /*
    Camera camera1{};
    //camera.getFrames();
    std::thread frame_catcher([&]() { camera1.getFrames(); });
    Screen screen{};
    screen.generate_phaseShift(Screen::Shift_mode::four_phase_shift);
    screen.displayPattern();
    
    if (frame_catcher.joinable()) frame_catcher.join();
    
    //Camera camera{};
    //camera.getFrames();
    */ 
    
}
