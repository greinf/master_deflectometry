#include "screen.hpp" // Be carefull renamed to Pattern
#include "enums.hpp"
#include "imageHandler.hpp"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <vector>
#include "imageStore.hpp"
#include <opencv2/core.hpp>


Pattern::Pattern(int heigth, int width, std::shared_ptr<ImageStore> img_store) :
	m_pixel_y{ heigth },
	m_pixel_x{ width },
	m_img_store { std::move(img_store) }
{  }



//void Pattern::gray_value_calib() {
//	// Sets flags
//	runtime_flags.set_finished_fringe_Iteration_false();
//	runtime_flags.set_stop_fringe_projection_flag_false();
//	// Allocate memory for array
//	cv::Mat gray_image(runtime_flags.disp.height, runtime_flags.disp.width, CV_8UC1);
//	assert(runtime_flags.calib.stepwidth && "Stepwidth is not defined \n");
//	for (size_t counter = 0; counter <= std::numeric_limits<uchar>::max(); counter += runtime_flags.calib.stepwidth ) {
//		//std::this_thread::sleep_for(std::chrono::milliseconds(500));
//		std::unique_lock<std::mutex> lk_pattern(runtime_flags.pattern_mutex);
//
//		// linearisatoin of gray value if m_LUT got value
//		if (m_LUT.has_value()) {
//			gray_image.setTo(linear_gray(static_cast<double>(counter)));
//		}
//		else { gray_image.setTo(cv::Scalar(static_cast<int>(counter))); }
//		imgHandler.imshow_Pattern(gray_image);
//		
//		if (runtime_flags.get_stop_fringe_projection_flag()) {
//			runtime_flags.set_stop_fringe_projection_flag_false();
//			std::cout << "Gray value projection interrupted\n";
//			lk_pattern.unlock();
//			runtime_flags.cv.notify_all();
//			return;
//		}
//		//unlocks the std::mutex and notify the controller automatic thread. 
//		lk_pattern.unlock();
//		runtime_flags.cv.notify_one();
//		std::cout << "Pattern Screen function " << counter << "\n";
//		// Reset flag and continue
//	}
//	std::cout << "Reached last gray value\n";
//	runtime_flags.set_finished_fringe_Iteration_true();
//	return;
//}

// n_checkersize in pixel
cv::Mat Pattern::generateCheckerboard(
	const int pixel_x,
	const int pixel_y,
	const int n_checkersize)
{
	CV_Assert(pixel_x > 0 && pixel_y > 0);
	CV_Assert(n_checkersize >= 0);

	cv::Mat pattern(pixel_y, pixel_x, CV_64F, cv::Scalar(0.0));

	const int W = pixel_x;
	const int H = pixel_y;

	// ---------- SPECIAL CASE: symmetric 2x2 checkerboard ----------
	if (n_checkersize == 0) {

		const int mx = W / 2;
		const int my = H / 2;

		for (int y = 0; y < H; ++y) {
			double* ptr = pattern.ptr<double>(y);
			const bool bottom = (y >= my);
			for (int x = 0; x < W; ++x) {
				const bool right = (x >= mx);
				const bool white = right ^ bottom;
				ptr[x] = white ? 255.0 : 0.0;
			}
		}
		/*cv::Mat gray;
		cv::normalize(pattern, gray, 0, 255, cv::NORM_MINMAX, CV_8U);
		cv::imshow("pattern", gray);
		cv::waitKey(0);
		cv::imwrite("C:/Users/grein/Desktop/josepha.jpg", gray);*/
		return pattern;
	}

	// ---------- GENERAL CASE: regular checkerboard ----------
	CV_Assert(n_checkersize < W && n_checkersize < H);

	for (int y = 0; y < H; ++y) {
		double* ptr = pattern.ptr<double>(y);
		const int by = y / n_checkersize;
		for (int x = 0; x < W; ++x) {
			const int bx = x / n_checkersize;
			const bool white = (bx + by) & 1;
			ptr[x] = white ? 255.0 : 0.0;
		}
	}

	return pattern;
}




cv::Mat Pattern::generateCross(
	const int pixel_x,
	const int pixel_y,
	const double mid_x,
	const double mid_y,
	const int thickness)
{
	// checks if the value is odd. We want even values vor symmetrie
	// This function only works for even pixel_x and even even pixely
	CV_Assert(!(thickness & 1) && !(pixel_x & 1) && !(pixel_y & 1));
	CV_Assert((pixel_x > 0) && (pixel_y > 0));
	CV_Assert((mid_x > 0) && (mid_x < pixel_x));
	CV_Assert((mid_y > 0) && (mid_y < pixel_y));
	CV_Assert(thickness > 1);
	
	int x_start, x_end, y_start, y_end;
	x_start = std::ceil(mid_x - (thickness/2.0));
	x_end = std::floor(mid_x + (thickness / 2.0));

	y_start = std::ceil(mid_y - (thickness / 2.0));
	y_end = std::floor(mid_y + (thickness / 2.0));

	cv::Mat pattern(pixel_y, pixel_x, CV_64F, cv::Scalar(0));
	for (int row = 0; row < pattern.rows; ++row) {
		double* const ptr = pattern.ptr<double>(row);
		for (int cols = 0; cols < pattern.cols; ++cols) {
			if ((row >= y_start) && (row <= y_end)) {
				ptr[cols] = 255;
				continue;
			}
			if ((cols >= x_start) && (cols <= x_end)) {
				ptr[cols] = 255;
			}
		}
	}
	return pattern;
}

std::vector<cv::Mat> Pattern::generateGrayCalibrationSequence(int stepwidth)
{
	std::vector<cv::Mat> grayFrames;
	grayFrames.reserve(256);     // worst case

	int H = m_pixel_y;
	int W = m_pixel_x;

	if (stepwidth <= 0) {
		throw std::runtime_error("Gray-value calibration stepwidth not defined.");
	}

	for (int val = 0; val <= 255; val += stepwidth)
	{
		cv::Mat frame(H, W, CV_8UC1);

		if (m_LUT.has_value() && m_lut_ready) {
			double linear_val = linear_gray(static_cast<double>(val));
			frame.setTo(static_cast<uchar>(linear_val));
		}
		else {
			frame.setTo(static_cast<uchar>(val));
		}

		// store in ImageStore
		m_img_store->add(FrameRole::GrayCalibrationGT, frame);

		grayFrames.push_back(frame);
	}

	return grayFrames;
}

cv::Mat Pattern::generateCartesian(int gridX, int gridY) {
	//Use the member varialles m_pixel_x and m_pixel_y
	CV_Assert((gridX > 0 )&& (gridY > 0));
	CV_Assert((m_pixel_x > 0) && (m_pixel_y > 0));
	cv::Mat cartesian = createCartesian(m_pixel_y, m_pixel_x, gridY, gridX);
	return cartesian;
}

cv::Mat Pattern::createCartesian(
	const int pixelY,
	const int pixelX,
	const int GridY,
	const int GridX) {
	CV_Assert((pixelX > 0) && (pixelY > 0));
	CV_Assert((GridY > 0) && (GridX > 0));
	CV_Assert((GridY <= pixelY) && (GridX <= pixelX));
	cv::Mat empty = cv::Mat::zeros(pixelY, pixelX, CV_8U);
	std::vector<cv::Vec2i> gridPoints = getCartesianGridpoints(empty.size(), GridX, GridY);

	for (std::size_t count = 0; count < gridPoints.size(); ++count) {
		cv::drawMarker(empty, cv::Point2i(gridPoints[count][0], gridPoints[count][1]), cv::Scalar(255), 0);
	}

	return empty;
}

cv::Mat Pattern::createCoordinateImg(int pixel_x, int pixel_y) {
	CV_Assert(pixel_x > 0 && pixel_y > 0);

	cv::Mat coord(pixel_y, pixel_x, CV_64FC2);

	for (int row = 0; row < coord.rows; ++row) {
		double* ptr = coord.ptr<double>(row);
		// --- Also possible and safer ---
		// One can use this instead of a double* which points to the first channel of the first column. 
		// cv::Vec2f* ptr = mat.ptr<cv::Vec2f>(row);
		// ptr[col] == cv::Vec2f(x,y)
		// Be Carefull .cols still refers to the REAL amount of cols and therefore must be multplied times the channels (as below)
		for (int col = 0; col < coord.cols; ++col) {
			// index = 2 * col because we have 2 channels per pixel
			ptr[2 * col] = static_cast<double>(col); // x
			ptr[2 * col + 1] = static_cast<double>(row); // y
		}
	}

	return coord;
}

cv::Mat Pattern::generatecoordianteImg() {
	cv::Mat coordImg = createCoordinateImg(m_pixel_x, m_pixel_y);
	return coordImg;
}

std::vector<cv::Vec2i> Pattern::getCartesianGridpoints(
	const cv::Size& sz,
	const int gridPointsX,
	const int gridPointsY)
{
	CV_Assert(sz.area() > 0);
	CV_Assert(gridPointsX <= sz.width);
	CV_Assert(gridPointsY <= sz.height);

	const int stepx = sz.width / gridPointsX;
	const int stepy = sz.height / gridPointsY;

	std::vector<cv::Vec2i> coordinates;

	for (int row = 0; row < sz.height; row += stepy) {
		for (int cols = 0; cols < sz.width; cols += stepx) {
			coordinates.push_back(cv::Vec2i(cols, row));
		}
	}
	return coordinates;
}


bool Pattern::prepareShiftParameters(int n_periods_in_y, int steps) {
	try {
		CV_Assert(m_pixel_x > 0 && m_pixel_y > 0);
		CV_Assert(n_periods_in_y >= 1);
		CV_Assert(steps >= 1);
		m_steps = steps;
		m_wavelength = static_cast<double>(m_pixel_y) / static_cast<double>(n_periods_in_y); //Number of periods is bound to the y-Axis here! 
		m_shift_length = ((CV_2PI) / static_cast<double>(steps));
		
		////runtime_flags Shoudl Be deleted in short time !!!!!
		//runtime_flags.disp.wavelength = m_wavelength;
		//runtime_flags.phase_shift.n_shifts = steps;
	}
	catch (std::exception& e) {
		std::cout << e.what() << " Parameter generation failed \n ";
		return false;
	}
	return true;
}
//m_patterns.reserve(2 * 4);


std::vector<cv::Mat> Pattern::generatePhase(int pixel_x, int pixel_y, double wavelength) {
	try {
		CV_Assert(wavelength > 0);

		cv::Mat row = generateRowPhase(pixel_x, wavelength);       // 1 × pixel_x
		cv::Mat col = generateColumnPhase(pixel_y, wavelength);    // 1 × pixel_y

		// Now repeat to 2D
		std::vector<cv::Mat> phase;
		phase.emplace_back(cv::repeat(row, pixel_y, 1));  // horizontal
		phase.emplace_back(cv::repeat(col, 1, pixel_x));  // vertical

		//Store the files in Image_storage File
		m_img_store->add(FrameRole::RawPhase, phase[0]);
		m_img_store->add(FrameRole::RawPhase, phase[1]);

		return phase;
	}
	catch (std::exception& e) {
		std::cout << "EXCEPTION :" << e.what() << std::endl;
		return {};
	}
}



std::vector<cv::Mat> Pattern::generateSinusPatternFromPhase(
	const std::vector<cv::Mat>& phaseMaps,
	int steps,
	double amplitude,
	double mean_value
)
{
	assert(phaseMaps.size() == 2);
	assert(phaseMaps[0].size() == phaseMaps[1].size());
	assert(steps > 0);

	std::vector<cv::Mat> outPatterns;
	outPatterns.reserve(phaseMaps.size() * steps);

	double shift_step = CV_2PI / static_cast<double>(steps);

	for (const auto& phi : phaseMaps) {

		for (int k = 0; k < steps; ++k) {
			double phase_shift = k * shift_step;

			cv::Mat shifted, pattern;

			cv::add(phi, phase_shift, shifted, cv::noArray(), CV_64F);

			cv::Mat cosine(shifted.size(), CV_64F);

			for (int y = 0; y < shifted.rows; ++y) {
				const double* src = shifted.ptr<double>(y);
				double* dst = cosine.ptr<double>(y);

				for (int x = 0; x < shifted.cols; ++x) {
					dst[x] = std::cos(src[x]);
				}
			}
			
			double mean = mean_value / amplitude;
			pattern = mean * (1.0 + cosine);
			pattern *= amplitude;
			if (m_LUT.has_value()) {
				cv::parallel_for_(cv::Range(0, pattern.rows),
					[&](const cv::Range& r) {
						for (int y = r.start; y < r.end; ++y) {
							double* ptr = pattern.ptr<double>(y);
							for (int x = 0; x < pattern.cols; ++x) {
								ptr[x] = Pattern::linear_gray(ptr[x]);
							}
						}
					});
			}
			m_img_store->add(FrameRole::PatternDouble, pattern.clone());
			pattern.convertTo(pattern, CV_8UC1);
			outPatterns.push_back(pattern);
			m_img_store->add(FrameRole::Pattern, pattern);
			m_mean_value = mean;
			m_amplitude = amplitude;
		}
	}
	return outPatterns;
}

cv::Mat Pattern::generateColumnPhase(int pixel_y, double wave_length) {
	try {
		CV_Assert(wave_length > 0 && pixel_y > 0);

		cv::Mat column(pixel_y, 1, CV_64F);
		

		double step = CV_2PI / static_cast<double>(wave_length);

		for (int y = 0; y < pixel_y; ++y) {
			double* ptr = column.ptr<double>(y);
			*ptr = CV_2PI * (static_cast<double>(y) / wave_length);
		}

		return column;
	}
	catch (std::exception& e) {
		std::cout << "EXCEPTION: " << e.what() << " in Columnphase generation \n";
	}
}

cv::Mat Pattern::generateRowPhase(int pixel_x, double wave_length) {
	try {
		CV_Assert(wave_length > 0 && pixel_x > 0);

		cv::Mat row(1, pixel_x, CV_64F);
		double* ptr = row.ptr<double>(0);

		double step = CV_2PI / static_cast<double>(wave_length);

		for (int x = 0; x < pixel_x; ++x) {
			//ptr[x] = step * x;
			ptr[x] = CV_2PI * (static_cast<double>(x) / wave_length);
		}

		return row;
	}
	catch (std::exception& e) {
		std::cout << "EXCEPTION: " << e.what() << " in Rowphase generation \n";
	}

}


double Pattern::linear_gray(double s)
{
	if (!m_lut_ready || m_sortedLUT.empty())
		return s; // fallback: no LUT active

	// transform 0..255 range into LUT-range
	double target = s * m_lut_scale_factor + m_lut_offset;

	// binary search on "second" values
	auto it = std::lower_bound(
		m_sortedLUT.begin(),
		m_sortedLUT.end(),
		target,
		[](const auto& a, double val) {
			return a.second < val;
		}
	);

	if (it == m_sortedLUT.begin())
		return it->first;

	if (it == m_sortedLUT.end())
		return std::prev(it)->first;

	// choose closer of the two neighbors
	double hi_dist = std::abs(it->second - target);
	double lo_dist = std::abs(std::prev(it)->second - target);

	if (lo_dist < hi_dist)
		return std::prev(it)->first;
	else
		return it->first;
}


void Pattern::prepareLUT()
{
	if (!m_LUT.has_value()) {
		m_lut_ready = false;
		return;
	}

	m_sortedLUT = m_LUT.value();
	std::sort(m_sortedLUT.begin(), m_sortedLUT.end(),
		[](auto& a, auto& b) { return a.second < b.second; });

	double minv = m_sortedLUT.front().second;
	double maxv = m_sortedLUT.back().second;

	double range = maxv - minv;
	double range_safety = range * 0.9;      // keep 10% margin

	m_lut_scale_factor = range_safety / 255.0;
	m_lut_offset = minv + range * 0.05;

	m_lut_ready = true;
}

void Pattern::generate_optimalPhase() {
	generate_phaseShift(Shift_mode::four_phase_shift);
	
}

//n_perdios_in_y defaulted to 10
void Pattern::generate_phaseShift(
	Shift_mode mode, 
	int n_periods_in_y, 
	int pixelX,
	int pixelY) {

	m_numberPeriods = n_periods_in_y;
	m_pixel_x = pixelX;
	m_pixel_y = pixelY;

	if (mode == Shift_mode::four_phase_shift) {
		m_steps = 4;

		if (!prepareShiftParameters(n_periods_in_y, m_steps)) {
			std::cout << "Parameter generation failed.\n";
			return;
		}

		auto phaseMaps = generatePhase(m_pixel_x, m_pixel_y, m_wavelength);
		auto patterns = generateSinusPatternFromPhase(phaseMaps, m_steps);
		/*for (const auto& m : phaseMaps) {
			cv::Mat norm;
			cv::normalize(m, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
			cv::imshow("norm", norm);
			cv::waitKey(0);
		}*/

		logging();
	}
	else if (mode == Shift_mode::user_defined) {
		std::cout << "Enter an integer for the amount of shifts (4 < x <= 100)\n";
		int steps = 0;
		do {
			std::cin >> steps;
			if (!std::cin) {
				std::cin.clear();
				std::cin.ignore(1000, '\n');
				std::cout << "Not a valid Input. Try again.\n";
				continue;
			}
		} while (steps < 5 || steps > 100);

		m_steps = steps;
		if (!prepareShiftParameters(n_periods_in_y, m_steps)) {
			std::cout << "Parameter generation failed.\n";
			return;
		}

		auto phaseMaps = generatePhase(m_pixel_x, m_pixel_y, m_wavelength);
		auto patterns = generateSinusPatternFromPhase(phaseMaps, m_steps);

		logging();
		
	}
	else {
		std::cout << "Not implemented\n";
	}
}

void Pattern::logging() {
	std::shared_ptr<defl::PhaseShiftConfig> logging_ptr = std::make_shared<defl::PhaseShiftConfig>();
	// logging when finished 
	if (logging_ptr) {
		logging_ptr->pixel_x = m_pixel_x;
		logging_ptr->pixel_y = m_pixel_y;
		logging_ptr->steps = m_steps;
		logging_ptr->periods_in_y = m_numberPeriods;
		logging_ptr->wavelength = m_wavelength;
		logging_ptr->shift_length = m_wavelength / m_steps;
		logging_ptr->amplitude = m_amplitude;
		logging_ptr->mean_value = m_mean_value;
		logging_ptr->lut_available = m_lut_ready;
		logging_ptr->lut_data = m_sortedLUT;
		logging_ptr->algorithm_name = std::to_string(m_steps) + " Shift Algorithm";
		std::cout << *logging_ptr;
		m_cfg.push_back(std::move(logging_ptr));
		
	}
	else std::cout << "Logging class Generation failed\n";
	m_numberPeriods = 0;
	m_steps = 0;
	m_wavelength = 0;
}
//
//void Pattern::displayPatterns_single_thread() {
//	cv::namedWindow("PhaseShift", cv::WINDOW_NORMAL);
//	cv::setWindowProperty("PhaseShift", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
//	std::cout << "Press P for next Image. \n";
//	std::vector<cv::Mat> pattern = m_img_store->get(FrameRole::Pattern);
//	for (const auto& m_pattern : pattern) {
//		while (true) {
//			cv::imshow("PhaseShift", m_pattern);
//			int key = cv::waitKey(50); // Poll every 20 ms to keep window responsive
//
//			if (key == 'p' || key == 'P') {
//				break;  // show next image
//			}
//			else if (key == 27) { // ESC
//				std::cout << "Display interrupted by user.\n";
//				return;  // exit the function early
//			}
//		}
//	}
//}
//
//
//void Pattern::displayPatterns_multi_thread() {
//	runtime_flags.set_finished_fringe_Iteration_false();
//	runtime_flags.set_stop_fringe_projection_flag_false();
//	assert(runtime_flags.phase_shift.n_pics_per_Phase && "n_pics is not defined \n");
//	int count{};
//	std::vector<cv::Mat> pattern = m_img_store->get(FrameRole::Pattern);
//	for (const auto& m_pattern : pattern) {
//		std::cout << "Fringe pattern: " << count++ << "\n";
//		std::unique_lock<std::mutex> lk_pattern(runtime_flags.pattern_mutex);
//		runtime_flags.next_fringe_process_finished_true();
//		imgHandler.imshow_Pattern(m_pattern);
//
//		if (runtime_flags.get_stop_fringe_projection_flag()) {
//			runtime_flags.set_stop_fringe_projection_flag_false();
//			std::cout << "Gray value projection interrupted\n";
//			lk_pattern.unlock();
//			runtime_flags.cv.notify_all();
//			return;
//		}
//
//		lk_pattern.unlock();
//		runtime_flags.cv.notify_one();
//
//	}
//	std::cout << "Reaches last pattern \n";
//	runtime_flags.set_finished_fringe_Iteration_true();
//	return;
//	
//}

