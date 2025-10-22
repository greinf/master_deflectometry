#define VERSION "1"

#include <cstddef>
#include <iostream>
#include "deflectometry.hpp"
#include "enums.hpp"
#include <opencv2/opencv.hpp>
#include "camera_calib.hpp"
#include <filesystem>

//#include "imgProcessing.hpp"

int main()
{
    //Supress open CV Information -only warnings are logged. 
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    std::vector<cv::Mat> vector{};
	std::string settings_path("C:\\Users\\grein\\Desktop\\Master\\Project\\deflectometrie\\data\\in_VID5.xml");
    std::string images("C:\\Users\\grein\\Desktop\\OpenCV_Example_CheckboardImages");
	std::filesystem::path image_files(images), settings_files(settings_path); 
    if(!std::filesystem::exists(image_files)) {
		std::cerr << "Settings file does not exist: " << image_files << '\n';
	}
    std::filesystem::directory_iterator dir_iter(image_files);
    for (const auto& entry : dir_iter) {
		std::cout << "Dir entry: " << entry.path() << '\n'; 
		vector.push_back(cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE).clone());
    }
    

	runCameraCalibration(vector, true, settings_files.string());

    /*
    Deflectometry meassure{};

    meassure.start_meassurement(Shift_mode::four_phase_shift, DisplayMode::Automatic, 0);


    meassure.phase_unwrap();
    */

    
    //ImageProcessing proessor{};
	//processor.load_images("C:\\Users\\grein\\Desktop\\Master\\Project\\deflectometrie\\out\\test_images\\four_phase_shift\\vertical\\");
    

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
