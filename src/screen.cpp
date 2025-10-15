#pragma once
#include "screen.hpp"
#include "enums.hpp"

Screen::Screen(std::int32_t pixel_x, std::int32_t pixel_y, std::int32_t pixel_pitch, float numberPeriods):
	m_pixel_x{pixel_x}, m_pixel_y{pixel_y}, m_pixel_pitch{pixel_pitch}, m_numberPeriods { numberPeriods },
	m_mode{Shift_mode::max_value}
{ }

bool Screen::preparefourShiftParameters() {
	try {
		m_mode = Shift_mode::four_phase_shift;
		CV_Assert(m_pixel_x > 0 && m_pixel_y > 0);
		CV_Assert(m_numberPeriods >= 1);
		m_wavelength = static_cast<float>(m_pixel_y) / m_numberPeriods;
		m_shift_length = ((CV_2PI) / 4);
		
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
		if (!preparefourShiftParameters()) {
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

//shift through all the images with a dealy of 1s;
void Screen::displayPatterns() {
	cv::namedWindow("PhaseShift", cv::WINDOW_NORMAL);
	cv::setWindowProperty("PhaseShift", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
	std::cout << "Press P for next Image. \n";
	for (const auto& m_pattern : m_patterns) {
		// Wait until 'P' or 'ESC' is pressed
		while (true) {
			// Blocking this thread is necessary because multiple calls to cv::waitKey from different 
			// thread occur. Becasue that pattern is not necessary to be updatet within milliseconds
			// it was decided to block this thread. 
			std::this_thread::sleep_for(std::chrono::milliseconds(300));
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

