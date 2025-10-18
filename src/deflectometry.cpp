#pragma once
#include "deflectometry.hpp"
#include "screen.hpp"
#include "acquisitionworker.hpp"
#include "cassert"
#include "enums.hpp"
#include "imageHandler.hpp"
#include "flagHandler.hpp"

cv::Mat rotImage180(const cv::Mat&);

//if image should be saved, assign here a handler function that can save the image. 
void showRawImage(const cv::Mat& mat) {
	cv::Mat rotated = rotImage180(mat);
	imgHandler.imshow_Camera(rotated);
}

cv::Mat rotImage180(const cv::Mat& mat) {
	int width = mat.cols; //no function just public member variable
	int height = mat.rows;
	cv::Vec<float, 2> center(float(width / 2.), float(height / 2.));
	cv::Mat rot_Matrix = cv::getRotationMatrix2D(center, 180., 1.);
	cv::Mat destination(height, width, CV_8UC1);
	cv::warpAffine(mat, destination, rot_Matrix, destination.size());
	return destination;
}



// Constructor Deflectometry() takes no argument. Automatically creates Camera class with ids::peak library. Acuqistionworker inherits from that. 
// If multiple cameras are used these can be choosen by the input Argument of Acquisitionworker. 
// For each camera, a new Acuistionworker instance must be created with the according index. 
// Screen class is created for Fringe projection
Deflectometry::Deflectometry() {
	m_screen = std::make_shared<Screen>();
	m_acquisition_worker = std::make_shared<AcquisitionWorker>(0); 
}

void Deflectometry::show_acquistion() {
	std::cout << m_acquisition_worker->m_frames.size() << std::endl;
	for (const auto& frame : m_acquisition_worker->m_frames) {
		cv::namedWindow("Raw Phase", cv::WINDOW_NORMAL);
		cv::setWindowProperty("Raw Phase", cv::WINDOW_NORMAL, cv::WINDOW_FREERATIO);
		cv::imshow("Raw Phase", frame);
		cv::waitKey(0);
	}
}

void Deflectometry::controller_automatic() {
	//Check for user if Setup is correct
	std::cout << "Press any button to start meassurment if camera sees full fringe pattern \n";
	char u{ '\0' };
	std::cin.ignore(1000, '\n');
	while (!u) {
		u = std::cin.get();
		if (!std::cin) {
			std::cin.clear();
			std::cin.ignore(1000, '\n');
		}
		if (u == EOF) {
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			u = '\0';
		}
	}
	runtime_flags.set_number_of_pictures_per_pattern(5);
	
	for (int i = 0; i < runtime_flags.get_number_of_shifts()*2; ++i) { //times two for vertikal and horizontal
		for (int j = 0; j < runtime_flags.get_number_of_pictures_per_pattern(); ++j) {
			runtime_flags.set_true_imSave_flag();
			std::this_thread::sleep_for(std::chrono::milliseconds(300));
		}
		runtime_flags.set_next_fringe_pattern_flag_true();
	}
	std::cout << "Finished :D \n";
	if (runtime_flags.get_finished_fringe_Iteration()) {
		std::cout << "finished? \n";
		runtime_flags.set_false_acquisition_flag();
		runtime_flags.set_stop_fringe_projection_flag_true();
		imgHandler.stop();
		
	}
}


void Deflectometry::controller_userInput() {
	std::cout << "Press S to save image. \n" <<
		"Press P for next fringe pattern \n" <<
		"Press E to abort \n";

	for (;;) {
		int ch = std::cin.get();          // use int to detect EOF
		if (ch == EOF) {
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			//m_controll_variable = command::stop;
			continue;
		}

		char c = static_cast<char>(ch);

		// skip whitespace (space, tab, enter, etc.)
		if (std::isspace(static_cast<unsigned char>(c)))
			continue;

		switch (std::toupper(static_cast<unsigned char>(c))) {
		case 'P':
			runtime_flags.set_next_fringe_pattern_flag_true();
			std::cout << "Next fringe pattern gets projected \n";
			std::cin.ignore(1000, '\n');
			break;

		case 'S':
			runtime_flags.set_true_imSave_flag();
			std::cout << "Image Saved!\n";
			std::cin.ignore(1000, '\n');
			break;
		case 'E':
			runtime_flags.set_false_acquisition_flag();
			runtime_flags.set_stop_fringe_projection_flag_true();
			imgHandler.stop();
			return;
		default:
			std::cout << "Not a valid input! (Use S, P or E )\n";
			break;
		}
	}
}


void Deflectometry::start_meassurement(Shift_mode shift_mode, DisplayMode disp_mode, int camera) {
	
	// m_screen & m_acquisition_worker must not be nullptr. Also camera running must be set. 
	assert(m_screen && m_acquisition_worker && runtime_flags.get_camera_running_flag());
	//After setUpAcuqisition Datastream is available
	m_acquisition_worker->setUpAcquisition(camera);
	m_acquisition_worker->getDatastream(camera);

	// All image dispalying is running via the image handler class
	m_acquisition_worker->assignImageHandler(showRawImage);
	//Shift mode is neede to generate the Pattern
	m_screen->generate_phaseShift(shift_mode);
	
	runtime_flags.set_next_fringe_pattern_flag_false(); //First set "next image flag" to false
	runtime_flags.set_stop_fringe_projection_flag_false(); //Set stop fringe projection flag to false
	
	// Run the Imagehandler class
	img_handler_thread = std::thread(&ImageHandler::run, &imgHandler, 2);
	
	// Lauch 
	std::thread controller;
	if (disp_mode == DisplayMode::UserInput) {
		controller = std::thread(&Deflectometry::controller_userInput, this);
	}
	if (disp_mode == DisplayMode::Automatic) {
		controller = std::thread(&Deflectometry::controller_automatic, this);// some function for automatic handling !!! 
	}

	std::thread fringe_pattern_thread(&Screen::displayPatterns_multi_thread, m_screen.get()); 
	
	std::thread camera_thread(&AcquisitionWorker::start, m_acquisition_worker.get(), disp_mode);
	
	// First thread to finish, should be Camera_thread. 
	// runtime_flags.acquisition_flag -> first set to false
	if (camera_thread.joinable()) camera_thread.join();

	// Second thread to finish, should be fringe_pattern thread
	// runtime_flags().stop_fringe_projection set to true;
	if (fringe_pattern_thread.joinable()) fringe_pattern_thread.join();

	// Third is ImageHandler img_handler->stop();
	if (img_handler_thread.joinable()) img_handler_thread.join();

	//Controller thread .join() call should block 
	if (controller.joinable()) controller.join();
	
	std::cout << "Function *Start Meassurement* Exit \n";

	return;
	
}