#ifndef	DEFLECTOMETRY_H
#define DEFLECTOMETRY_H

#include <memory>
#include <vector>
#include <exception>
#include <iostream>
#include <cassert>
#include <thread>

//Forward Decleration Enums + Class
enum class AcquisitionMode;
enum class Shift_mode;
enum class DisplayMode;

class Screen;
class AcquisitionWorker;

class Deflectometry {
public:
	 Deflectometry();

	 void start_meassurement(Shift_mode, DisplayMode, int);

private:
	std::shared_ptr<Screen> m_screen{nullptr};
	std::shared_ptr<AcquisitionWorker> m_acquisition_worker{ nullptr };
	std::thread img_handler_thread;

	// Contoller_thread that is used to oversee the acquisition of camera frames
	// druing the meassurment. This is done by UserInput.
	void controller_userInput();

	// Controller thread for automatic acquisaition. Default argument is ammount of pictures taken per 
	// Meassurment. Information about ammount of shift_steps is saved in flagHandler.hpp.
	void controller_automatic();
};

#endif