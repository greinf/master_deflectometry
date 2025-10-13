#include "screen.hpp"


Screen::Screen(std::int32_t pixel_x, std::int32_t pixel_y, std::int32_t pixel_pitch):
	m_pixel_x{pixel_x}, m_pixel_y{pixel_y}, m_pixel_pitch{pixel_pitch}
{ }

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

		m_patterns.clear(); // clear, don’t call .empty()

		if (!m_sinus->generate(m_patterns)) {
			std::cout << "Sinus pattern generation failed\n";
			continue;
		}

		displayPattern();
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

