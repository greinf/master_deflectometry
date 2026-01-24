#include "screen.hpp" // Be carefull renamed to Pattern
#include "enums.hpp"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <vector>
#include "imageStore.hpp"
#include <opencv2/core.hpp>
#include "config/PhaseShiftConfig.hpp"
#include "RowPolicy.hpp"


Pattern::Pattern(ImageStore& img_store) :
	m_img_store { img_store }
{  }


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
	x_start = static_cast<int>(std::ceil(mid_x - (thickness/2.0)));
	x_end = static_cast<int>(std::floor(mid_x + (thickness / 2.0)));

	y_start = static_cast<int>(std::ceil(mid_y - (thickness / 2.0)));
	y_end = static_cast<int>(std::floor(mid_y + (thickness / 2.0)));

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

std::vector<cv::Mat> Pattern::generateGrayCalibrationSequence(
	const int stepwidth,
	const int pixel_x,
	const int pixel_y)
{
	CV_Assert(stepwidth >= 1);
	CV_Assert(pixel_x >= 1);
	CV_Assert(pixel_y >= 1);
	std::vector<cv::Mat> grayFrames;
	grayFrames.reserve(256);     // worst case

	int H{}, W{};
	
	(pixel_x != m_pixel_x) ? (W = pixel_x) : (W = m_pixel_x);
	(pixel_y != m_pixel_y) ? (H = pixel_y) : (H = m_pixel_y);

	for (int val = 0; val <= 255; val += stepwidth)
	{
		cv::Mat frame(H, W, CV_8UC1);

		frame.setTo(static_cast<uchar>(val));

		// store in ImageStore
		m_img_store.add(FrameRole::GrayCalibrationGT, frame);

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

cv::Mat Pattern::generatecoordianteImg(
	const int pixel_x, 
	const int pixel_y) {
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
	}
	catch (std::exception& e) {
		std::cout << e.what() << " Parameter generation failed \n ";
		return false;
	}
	return true;
}
//m_patterns.reserve(2 * 4);


std::vector<cv::Mat> Pattern::generatePhase(
	int pixel_x, 
	int pixel_y, 
	double wavelength) 
{
	try {
		CV_Assert(wavelength > 0);

		cv::Mat row = generateRowPhase(pixel_x, wavelength);       // 1 × pixel_x
		cv::Mat col = generateColumnPhase(pixel_y, wavelength);    // pixely x 1

		// Now repeat to 2D
		std::vector<cv::Mat> phase;
		phase.emplace_back(cv::repeat(row, pixel_y, 1));  // horizontal
		phase.emplace_back(cv::repeat(col, 1, pixel_x));  // vertical

		CV_Assert(phase[0].size() == phase[1].size());
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
	double mean_value,
	bool uniformRow_Cols)
{
	assert(phaseMaps.size() == 2);
	assert(phaseMaps[0].size() == phaseMaps[1].size());
	assert(steps > 0);
	CV_Assert(phaseMaps[0].type() == CV_64F);
	CV_Assert((amplitude + mean_value) < 256);
	
	m_amplitude = amplitude;
	m_mean_value = mean_value;

	std::vector<cv::Mat> outPatterns;
	outPatterns.reserve(phaseMaps.size() * steps);

	double shift_step = CV_2PI / static_cast<double>(steps);

	shift_axis axis;
	for (std::size_t i = 0; i < phaseMaps.size(); ++i) {
		axis = (i == 0) ? shift_axis::horizontal : shift_axis::vertikal;
		cv::Mat phi = phaseMaps[i];

		for (int k = 0; k < steps; ++k) {
			double phase_shift = k * shift_step;

			cv::Mat shifted, pattern;

			cv::add(phi, phase_shift, shifted, cv::noArray(), CV_64F);

			cv::Mat cosine(shifted.size(), CV_64F);

			// Much cheaper to caclulate
			if (uniformRow_Cols) {
				cv::Mat working;
				switch (axis) {
				case(shift_axis::horizontal): {
					working.create(cv::Size(shifted.cols, 1), CV_64F);
					double* dst = working.ptr<double>(0);
					double* src = shifted.ptr<double>(shifted.rows / 2);
					for (int cols = 0; cols < shifted.cols; ++cols) {
						dst[cols] = std::cos(src[cols]);
					}
					cosine = cv::repeat(working, shifted.rows, 1);
					break;
				}
				case(shift_axis::vertikal): {
					working.create(cv::Size(1, shifted.rows), CV_64F);
					//double* dst = working.ptr<double>(0);
					for (int rows = 0; rows < shifted.rows; ++rows) {
						*(working.ptr<double>(rows)) = std::cos(shifted.ptr<double>(rows)[shifted.cols / 2]);
					}
					cosine = cv::repeat(working, 1, shifted.cols);
					break;
				}
				}
			}

			else {
				// Very expensive computation. 
				// This could be done for only one row and reapeated over the image
				// This approach was taken to allow that rows may differ
				cv::parallel_for_(cv::Range(0, shifted.rows),
					[&](const cv::Range& range) {
						for (int row = range.start; row < range.end; ++row) {
							const double* src = shifted.ptr<double>(row);
							double* dst = cosine.ptr<double>(row);

							for (int x = 0; x < shifted.cols; ++x) {
								dst[x] = std::cos(src[x]);
							}
						}
					}
				);
			}

			// cosine contains cos(phi + shift) in [-1,1]
			cv::Mat pattern64 = mean_value + amplitude * cosine;  // range [mean-ampl, mean+ampl]

			m_img_store.add(FrameRole::PatternDouble, pattern64.clone());
			cv::Mat pattern8;
			//cv::normalize(pattern64, pattern8, 0, 255, cv::NORM_MINMAX, CV_8U);
			pattern64.convertTo(pattern8, CV_8U, 1.0, 0.0);
			outPatterns.push_back(pattern8);
			m_img_store.add(FrameRole::Pattern, pattern8);

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
		return {};
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
		return {};
	}
}

void Pattern::generate_optimalPhase() {
	generate_phaseShift(Shift_mode::four_phase_shift);
	
}


std::vector<cv::Mat> Pattern::generate_phaseShift(
	Shift_mode mode, 
	int n_periods_in_y, 
	double mean,
	double ampl,
	int pixelX,
	int pixelY,
	bool uniformRow)
{
	m_numberPeriods = n_periods_in_y;
	m_pixel_x = pixelX;
	m_pixel_y = pixelY;
	m_mean_value = mean;
	m_amplitude = ampl;

	if (mode == Shift_mode::four_phase_shift) {
		m_steps = 4;

		if (!prepareShiftParameters(n_periods_in_y, m_steps)) {
			std::cout << "Parameter generation failed.\n";
			return{};
		}

		auto phaseMaps = generatePhase(m_pixel_x, m_pixel_y, m_wavelength);
		
		auto patterns = generateSinusPatternFromPhase(
			phaseMaps,
			m_steps,
			127.5,
			127.5,
			uniformRow);
		
		//Store the files in Image_storage File
		m_img_store.add(FrameRole::RawPhase, phaseMaps);
		m_img_store.add(FrameRole::RawPhase, patterns);

		logging();

		return patterns;
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
			return{};
		}

		auto phaseMaps = generatePhase(m_pixel_x, m_pixel_y, m_wavelength);
		auto patterns = generateSinusPatternFromPhase(phaseMaps, m_steps);

		//Store the files in Image_storage File
		m_img_store.add(FrameRole::RawPhase, phaseMaps);
		m_img_store.add(FrameRole::RawPhase, patterns);

		logging();

		return patterns;
	}
}

std::vector<defl::PhaseShiftConfig> Pattern::getPhaseConfig() const
{
	std::vector<defl::PhaseShiftConfig> config;
	for (const auto& cfg : m_cfg) {
		config.push_back(*cfg);
	}
	return config;
}


void Pattern::logging() {
	std::shared_ptr<defl::PhaseShiftConfig> logging_ptr = std::make_shared<defl::PhaseShiftConfig>();
	// logging when finished 
	if (logging_ptr) {
		logging_ptr->pixel_x = m_pixel_x;
		logging_ptr->pixel_y = m_pixel_y;
		logging_ptr->steps = m_steps;
		logging_ptr->periods_in_y = static_cast<int>(m_numberPeriods);
		logging_ptr->wavelength = m_wavelength;
		logging_ptr->shift_length = m_wavelength / m_steps;
		logging_ptr->amplitude = m_amplitude;
		logging_ptr->mean_value = m_mean_value;
		logging_ptr->algorithm_name = std::to_string(m_steps) + " Shift Algorithm";
		std::cout << *logging_ptr;
		m_cfg.push_back(std::move(logging_ptr));
		
	}
	else std::cout << "Logging class Generation failed\n";
	m_numberPeriods = 0;
	m_steps = 0;
	m_wavelength = 0;
}

