#include "screen.hpp"


Screen::Screen(std::int32_t pixel_x, std::int32_t pixel_y, std::int32_t pixel_pitch, float numberPeriods):
	m_pixel_x{pixel_x}, m_pixel_y{pixel_y}, m_pixel_pitch{pixel_pitch}, m_numberPeriods { numberPeriods },
	m_mode{Shift_mode::max_values}
{ }

bool Screen::preparefourShiftParameters() {
	try {
		m_mode = Shift_mode::four_phase_shift;
		CV_Assert(m_pixel_x > 0 && m_pixel_y > 0);
		CV_Assert(m_numberPeriods >= 1);
		//m_wavelength_horizontal = static_cast<float>(m_pixel_x) / m_numberPeriods;
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
		for (bool horizontal : {true, false}) {
			double two_pi_overlambda{ CV_2PI / m_wavelength };
			int axisLen{};
			int steps{};

			//here input any other mode with maybe more steps. 
			if (m_mode == Shift_mode::four_phase_shift) {
				steps = 4;

				switch (horizontal) {
				case(true): {
					//two_pi_overlambda = CV_2PI / m_wavelength_horizontal;
					axisLen = m_pixel_x ;
					break;
				}
				case(false): {
					//two_pi_overlambda = CV_2PI / m_wavelength_vertical;
					axisLen = m_pixel_y;
					break;
				}
				default: return false;
				}
				cv::Mat line(axisLen, 1, CV_32F); // column vector for convenience *** Constructor Mat (int rows, int cols, int type)

				for (int k = 0; k < steps; ++k) {
					const double phaseShift = 2.0 * CV_PI * (static_cast<double>(k) / steps);

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
						img = cv::repeat(row, /*ny=*/m_pixel_y, /*nx=*/1); // 1xH -> 1xW repeated
						//cv::transpose(img, img); // back to HxW
					}
					else {
						// vertical stripes -> vary along columns (X). Repeat the column across width.
						img = cv::repeat(lineScaled, /*ny=*/1, /*nx=*/m_pixel_x); // Hx1 -> HxW
					}

					// Convert to 8-bit for display/projection
					cv::Mat img8;
					img.convertTo(img8, CV_8U);
					m_patterns.push_back(std::move(img8));
				}
			}
		}
	}
	catch (std::exception& e) { std::cout << "EXCEPTION " << e.what() << std::endl; }
}

void Screen::generate_phaseShift(Shift_mode mode) {
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


void Screen::displayPattern() {
	cv::namedWindow("PhaseShift", cv::WINDOW_NORMAL);
	cv::setWindowProperty("PhaseShift", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
	for (const auto& m_pattern : m_patterns) {
		cv::imshow("PhaseShift", m_pattern);
		cv::waitKey(0);
	}

}

/*
void Screen::generate_phaseShift() {
	// run horizontal then vertical explicitly
	for (bool horizontal : {true, false}) {
		if (!preparePhaseshift(horizontal)) {
			std::cout << "Prepare Phaseshift did fail\n";
			continue;
		}

		// (RE)create the generator with current params
		auto m_sinus = cv::structured_light::SinusoidalPattern::create(
			cv::makePtr<cv::structured_light::SinusoidalPattern::Params>(m_params)
		);

		m_patterns.clear(); 

		if (!m_sinus->generate(m_patterns)) {
			std::cout << "Sinus pattern generation failed\n";
			continue;
		}

		displayPattern();
	}
}

bool Screen::preparePhaseshift(bool horizontal_shift, const Shift_mode mode) {
	try {
		m_params.width = m_pixel_x;
		m_params.height = m_pixel_y;
		m_params.nbrOfPeriods = 10;
		m_params.setMarkers = false;
		m_params.horizontal = horizontal_shift;
		if (mode == Shift_mode::four_phase_shift) {
			m_params.shiftValue = static_cast<float>(2 * CV_PI / 4);
		}
		m_params.nbrOfPixelsBetweenMarkers = 70; 
	}
	catch (std::exception& e) { 
		std::cout << "EXCPETION :" << e.what() << std::endl;
		return false;
	}
	return true;
}
*/

