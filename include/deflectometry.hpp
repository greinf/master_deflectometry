#ifndef	DEFLECTOMETRY_H
#define DEFLECTOMETRY_H

#include <memory>
#include <vector>
#include <exception>
#include <iostream>
#include <cassert>
#include <thread>
#include <opencv2/opencv.hpp>

//Forward Decleration Enums + Class
enum class AcquisitionMode;
enum class Shift_mode;
enum class DisplayMode;

class Screen;
class AcquisitionWorker;
class ImageProcessing;

class Deflectometry {
public:
	 Deflectometry();

	 void start_meassurement(Shift_mode, DisplayMode, int);
	 void show_acquistion();
	 void phase_unwrap();

	 void camera_calibration(int camera);
	 
	 void Deflectometry::save_unwrap(std::string&);

	 void save_frames(std::vector<cv::Mat>& frames, const std::string& path);

private:
	std::shared_ptr<Screen> m_screen{nullptr};
	std::shared_ptr<AcquisitionWorker> m_acquisition_worker{ nullptr };
	std::shared_ptr<ImageProcessing> m_img_processing{ nullptr };
	std::thread img_handler_thread;

	// Contoller_thread that is used to oversee the acquisition of camera frames
	// druing the meassurment. This is done by UserInput.
	void controller_userInput();


	
	// Controller thread for automatic acquisaition. Default argument is ammount of pictures taken per 
	// Meassurment. Information about ammount of shift_steps is saved in flagHandler.hpp.
	void controller_automatic();

	
	
};

#endif