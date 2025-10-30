#pragma once
#include "screen.hpp"
#include "enums.hpp"
#include "flagHandler.hpp"
#include "imageHandler.hpp"
#include <condition_variable>
#include <mutex>
/*
Screen::Screen(std::int32_t pixel_x, std::int32_t pixel_y, std::int32_t pixel_pitch, float numberPeriods):
	m_pixel_x{pixel_x}, m_pixel_y{pixel_y}, m_pixel_pitch{pixel_pitch}, m_numberPeriods { numberPeriods },
	m_mode{Shift_mode::max_value}
{ }
*/
Screen::Screen(int n_shifts) {
	getfromFlag_H(n_shifts);
}

void Screen::getfromFlag_H(int n_shifts) {
	m_pixel_x = runtime_flags.disp.width;
	m_pixel_y = runtime_flags.disp.height;
	m_pixel_pitch = runtime_flags.disp.pixelptich_mm;
	m_numberPeriods = n_shifts;
	runtime_flags.set_number_of_shifts(n_shifts);
	m_mode = Shift_mode::max_value;
}


void Screen::gray_value_calib() {
	// Sets flags
	runtime_flags.set_finished_fringe_Iteration_false();
	runtime_flags.set_stop_fringe_projection_flag_false();
	// Allocate memory for array
	cv::Mat gray_image(runtime_flags.disp.height, runtime_flags.disp.width, CV_8UC1);
	assert(runtime_flags.calib.stepwidth && "Stepwidth is not defined \n");
	for (size_t counter = 0; counter < 256; counter += runtime_flags.calib.stepwidth ) {
		//std::this_thread::sleep_for(std::chrono::milliseconds(500));
		std::unique_lock<std::mutex> lk_pattern(runtime_flags.pattern_mutex);
		//in theory this runtime_flags should not be necessary
		gray_image.setTo(cv::Scalar(static_cast<int>(counter)));
		imgHandler.imshow_Pattern(gray_image);
		
		if (runtime_flags.get_stop_fringe_projection_flag()) {
			runtime_flags.set_stop_fringe_projection_flag_false();
			std::cout << "Gray value projection interrupted\n";
			lk_pattern.unlock();
			runtime_flags.cv.notify_all();
			return;
		}
		//unlocks the std::mutex and notify the controller automatic thread. 
		lk_pattern.unlock();
		runtime_flags.cv.notify_one();
		std::cout << "Pattern Screen function " << counter << "\n";
		// Reset flag and continue
	}
	std::cout << "Reached last gray value\n";
	runtime_flags.set_finished_fringe_Iteration_true();
	return;
}

bool Screen::prepareShiftParameters() {
	try {
		runtime_flags.set_number_of_shifts(m_steps);
		CV_Assert(m_pixel_x > 0 && m_pixel_y > 0);
		CV_Assert(m_numberPeriods >= 1);
		m_wavelength = static_cast<float>(m_pixel_y) / m_numberPeriods; //Number of periods is bound to the y-Axis here! 

		//runtime_flags
		m_shift_length = ((CV_2PI) / m_steps);
		runtime_flags.disp.wavelength = m_wavelength;
	}
	catch (std::exception& e) {
		std::cout << e.what() << " Parameter generation failed \n ";
		return false;
	}
	return true;
}
//m_patterns.reserve(2 * 4);

bool Screen::generateSinusPatterns() {	
	try {
		double two_pi_overlambda{ CV_2PI / m_wavelength };
		int axisLen{};
		/*
		if (m_mode == Shift_mode::four_phase_shift) {
			m_steps = 4;
		}

		if (m_mode == Shift_mode::user_defined) {
			std::cout << "Input the needed ammount of phase steps during meassurment (4 < N < 12): \n";
			bool valid_input{ true };
			m_steps = 0;
			while (!(m_steps > 4 && m_steps < 13)) {
				std::cin >> m_steps;
				if (!std::cin) {
					std::cin.clear();
					std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				}
			}
			std::cout << "You enterd " << m_steps << " steps. '\n";
		}
		*/

		for (bool horizontal : {true, false}) {

			switch (horizontal) {
			case(true): {
				//two_pi_overlambda = CV_2PI / m_wavelength_horizontal;
				axisLen = m_pixel_x;
				break;
			}
			case(false): {
				//two_pi_overlambda = CV_2PI / m_wavelength_vertical;
				axisLen = m_pixel_y;
				break;
			}
			}
			cv::Mat line(axisLen, 1, CV_32F); // column vector for convenience *** Constructor Mat (int rows, int cols, int type)

			for (int k = 0; k < m_steps; ++k) {
				const double phaseShift = 2.0 * CV_PI * (static_cast<double>(k) / m_steps);

				// sin( 2π * i / λ + φ_k )
				for (int i = 0; i < axisLen; ++i) {
					float s = std::sin(two_pi_overlambda * static_cast<double>(i) + phaseShift);
					line.at<float>(i, 0) = static_cast<float>(s);
				}

				// Map [-1, +1] -> [mean-amp, mean+amp]
				cv::Mat lineScaled;
				lineScaled = line * static_cast<float>(m_amp);
				lineScaled = lineScaled + static_cast<float>(m_mean);

				// Build the 2D image by repeating along the constant axis
				cv::Mat img;
				if (horizontal) {
					// horizontal stripes -> vary along rows (Y). Repeat the column across width.
					cv::Mat row; cv::transpose(lineScaled, row);  // 1 x H
					img = cv::repeat(row, m_pixel_y, 1); // 1xH -> 1xW repeated
				}
				else {
					// vertical stripes -> vary along columns (X). Repeat the column across width.
					img = cv::repeat(lineScaled, 1, m_pixel_x); // Hx1 -> HxW
				}

				// Convert to 8-bit for display/projection
				cv::Mat img8;
				img.convertTo(img8, CV_8U);
				m_patterns.push_back(std::move(img8));
			}
		}
		return true;
	}
	catch (std::exception& e) { std::cout << "EXCEPTION " << e.what() << std::endl; return false; }
}

void Screen::generate_phaseShift(Shift_mode mode) {
	m_mode = mode;
	if (mode == Shift_mode::four_phase_shift) {
		m_steps = 4;
		if (!prepareShiftParameters()) {
			std::cout << "Parameter generation failed. \n";
		}
		if (!generateSinusPatterns()) {
			std::cout << "Sinsu generation failed. \n";
		}
	}
	else if (mode == Shift_mode::user_defined) {
		std::cout << "Enter an integer for the ammount of shifts (4 < x <= 12) \n";
		m_steps = 0;
		do {
			std::cin >> m_steps;
			if (!std::cin) {
				std::cin.clear();
				std::cin.ignore(1000, '\n');
				std::cout << "Not a valid Input. Try again \n";
				continue;
			}
			std::cout << m_steps << '\n';
		} while ((m_steps < 5) || (m_steps > 11));
		if (!prepareShiftParameters()) {
			std::cout << "Parameter generation failed. \n";
		}
		if (!generateSinusPatterns()) {
			std::cout << "Sinsu generation failed. \n";
		}
	}
	else {
		std::cout << "Not implemented \n"; 
		return;
	}
}

void Screen::showImage(const cv::Mat& img) {
	m_keepDisplaying.store(true);
	cv::namedWindow("PhaseShift", cv::WINDOW_NORMAL);
	cv::setWindowProperty("PhaseShift", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
	while (m_keepDisplaying.load()) {
		// Blocking this thread is necessary because multiple calls to cv::waitKey from different 
		// thread occur. Becasue that pattern is not necessary to be updatet within milliseconds
		// it was decided to block this thread. 
		cv::imshow("PhaseShift", img);
		std::this_thread::sleep_for(std::chrono::milliseconds(300));
		int key = cv::waitKey(10);
		if (key == 27) {  // ESC key for example
			m_keepDisplaying = false;
		}
	}
}

void Screen::displayPatterns_single_thread() {
	cv::namedWindow("PhaseShift", cv::WINDOW_NORMAL);
	cv::setWindowProperty("PhaseShift", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
	std::cout << "Press P for next Image. \n";
	for (const auto& m_pattern : m_patterns) {
		while (true) {
			cv::imshow("PhaseShift", m_pattern);
			int key = cv::waitKey(50); // Poll every 20 ms to keep window responsive

			if (key == 'p' || key == 'P') {
				break;  // show next image
			}
			else if (key == 27) { // ESC
				std::cout << "Display interrupted by user.\n";
				return;  // exit the function early
			}
		}
	}
}


void Screen::displayPatterns_multi_thread() {
	for (const auto& m_pattern : m_patterns) {
		runtime_flags.next_fringe_process_finished_true();
		while (true) {
			//Fringe Pattern update 
			runtime_flags.set_finished_fringe_Iteration_false();
			imgHandler.imshow_Pattern(m_pattern);
			if (runtime_flags.get_next_fringe_pattern_flag()) {
				runtime_flags.set_next_fringe_pattern_flag_false();
				break;  // show next image
			}
			if (&m_pattern == &m_patterns.back()) {
				std::cout << "Reached Last fringe Pattern \n";
				runtime_flags.set_finished_fringe_Iteration_true();
				return;
			}
			//If stop_fringe_projection_flag is set -> early return
			else if (runtime_flags.get_stop_fringe_projection_flag()) { 
				runtime_flags.set_stop_fringe_projection_flag_false();
				std::cout << "Fringe Pattern interrupt \n";
				return;  // exit the function early
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}
}

