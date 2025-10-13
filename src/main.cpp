#define VERSION "1"

#include <cstddef>
#include <iostream>

#include "camera.hpp"
#include "screen.hpp"

int main()
{

    Screen screen{};
    screen.generate_phaseShift();
    screen.displayPattern();
    
    
    //Camera camera{};
    //camera.getFrames();
     

}
