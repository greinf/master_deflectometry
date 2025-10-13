#define VERSION "1"

#include <cstddef>
#include <iostream>

#include "camera.hpp"
#include "screen.hpp"

int main()
{

    Screen screen{};
    screen.generate_phaseShift(Screen::Shift_mode::four_phase_shift);
    screen.displayPattern();
    
    
    //Camera camera{};
    //camera.getFrames();
     

}
