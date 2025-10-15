#define VERSION "1"

#include <cstddef>
#include <iostream>
#include "deflectometry.hpp"
#include "enums.hpp"


int main()
{
    Deflectometry meassure{};

    meassure.start_meassurement(Shift_mode::four_phase_shift, DisplayMode::Automatic, 0);
    

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
