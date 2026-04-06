// Creates syntehical images of the scene by doing a homography and smoothing acoording to the
	 // circle of confusion. 
	 // If no smoothing set smooth to false
	 // If no homogrpahy should be done. set warp to false!
	 // If Gamma is set to smooth to false
	 // The overall order in which computation is done differs for homorgraphy and Raycasting.
	 // ------------------------------Homogrpahy path ------------------------------------------------
	 // If Homography is used -> Create Pattern -> Gamma -> (if smoothing = true) Smoothing (with a fixed Kernel with respect
	 // to the distance to the object and Focuslength) -> if (luminance = true) calculate luminance (lambda emitter assumed) for different angles of dispaly and positions ->
	 // Warp the image with fixed RoiBorders (calculate Homography Matrix and warp into the Camera image (Camera pixel_width and pixel_height needed)
	 // Be carefull the homography, luminance and smoothing are not connected -> Therefore the kernel (smoothing) stays fixed and homography and luminance are 
	 // totally unrealated in this case 
	 // 
	 // ------------------------------Raycasting path ------------------------------------------------
	 // If Raycasting is used -> create Pattern -> Gamma 
	 // 
	 //
std::vector<cv::Mat> createSyntheticalImages(
	const Warping,
	std::optional< std::vector<cv::Mat>> camera_matrix,
	const double gamma = 2.2,
	const bool display_qunatize = true,
	const bool camera_quantization = true,
	const bool luminance = true,
	const bool smoothing = true,
	const bool warp = true,
	const double image_height = 400,  // Mirror circumference  
	const int dest_width = 2464,      // Mako G-507-B width
	const int dest_height = 2056,     // Mako G-507-B height
	const double display_pixel_pitch = 0.2745,                //PixelPitch  FH 0.277    BMZ: 
	const double display_shift_x = 0.0,
	const double display_shift_y = 0.0,
	const double display_tilt_x = 0.0,
	const double display_tilt_y = 0.0,
	const Shift_mode mode = Shift_mode::four_phase_shift,
	const _defl_::GrayCal::Method method = _defl_::GrayCal::Method::None,
	const int pattern_width = 1920,
	const int pattern_height = 1080,
	const int n_periods_in_y = 10,
	const double aperture_number = 2.4,
	const double distance = 3200,     // f = 1600 distance 2*f
	const double object_height = 6.6, // 2/3" Sensor 8,8 * 6,6 
	const RoiBorders<double> destination = {
	{200, 200},   // left up corner
	{1800, 180},  // left down corner,
	{210, 2200},  // right up corner,
	{1900, 1900}, // right down corner
	},
	const std::string& calibPath = ""
	);


std::vector<cv::Mat> Deflectometry::createSyntheticalImages(
	const Warping operation,
	std::optional<std::vector<cv::Mat>> cameraMatrix,
	const double gamma,
	const bool display_quantization,
	const bool camera_quantization,
	const bool luminance,
	const bool smoothing,
	const bool warping,
	const double image_height,  // Mirror circumference  
	const int dest_width,      // Mako G-507-B width 
	const int dest_height,     // Mako G-507-B height
	const double dispaly_pixel_pitch,
	const double display_shift_x,
	const double display_shift_y,
	const double display_tilt_x,
	const double display_tilt_y,
	const Shift_mode mode,
	const _defl_::GrayCal::Method method,
	const int pattern_width,
	const int pattern_height,
	const int n_periods_in_y,
	const double aperture_number,
	const double distance,     // f = 1600 distance 2*f
	const double object_height, // 2/3" Sensor 8,8 * 6,6 
	const RoiBorders<double> homography_points,
	const std::string& calib_path)
{
	CV_Assert(dispaly_pixel_pitch >= 0);
	CV_Assert(pattern_width > 0 && pattern_height > 0);
	CV_Assert(aperture_number >= 0 && n_periods_in_y > 0);
	CV_Assert(distance >= 0 && object_height > 0);
	CV_Assert(image_height >= 0);
	CV_Assert(dest_width >= 0 && dest_height >= 0);
	CV_Assert(gamma >= 0);
	CV_Assert(display_tilt_x < CV_PI / 4);
	CV_Assert(display_tilt_y < CV_PI / 4);


	setupPattern(*m_img_store);

	if (!setupCalibration(method, calib_path)) {
		std::cout << "Calibration Setup failed \n";
		return {};
	}

	std::vector<cv::Mat> pattern;

	// Define the start Pattern 
	if (mode == Shift_mode::four_phase_shift ||
		mode == Shift_mode::user_defined)
	{
		// Generate Pattern
		pattern = m_pattern->generate_phaseShift(
			mode,
			n_periods_in_y,
			127.5,   // Mean Value
			127.5,   // Amplitude
			pattern_width,
			pattern_height,
			UniformRowsCols{},
			true);   // if Return value should be double 
	}

	if (mode == Shift_mode::GrayValues) {
		pattern = m_pattern->generateGrayCalibrationSequence(
			1,
			pattern_width,
			pattern_height
		);

		for (auto& img : pattern) {
			img.convertTo(img, CV_64F);
		}

	}
	//show_norm(pattern, "pattern");


	// ------------------  Gray Val Calibration -> here passive Camera Calibration and LUT  ----------------------

	// Dispaly Quantization
	std::vector<cv::Mat> qunatized_pattern;
	if (display_quantization) {
		for (const auto& img : pattern) {
			qunatized_pattern.emplace_back(
				m_img_processing->quantizeImage(
					img,
					255.0,
					0.0));
		}
	}
	else qunatized_pattern = pattern;




	// Gamma Distortion
	std::vector<cv::Mat> pattern_gamma;
	if (gamma) {
		for (const auto& img : qunatized_pattern) {
			pattern_gamma.emplace_back(
				m_img_processing->do_gamma_distortion(
					gamma,
					img,
					UniformRowsCols{}
				)
			);
		}
		//return pattern_gamma;
	}

	else pattern_gamma = pattern;

	// show_norm(pattern_gamma, "gamma_pattern");

	// Apply Passive or Lut Calibration !!! 

	std::vector<cv::Mat> calibrated{};



	// ****** Calibration Method ************
	// at the moment commented out. We need to restructure everything. 


	/*switch (method) {
	case(CalibrationMethod::Lut):
		for (auto& img : pattern_gamma) {
			calibrated.emplace_back(m_calibration->applyCalibration(method, img));
		}
		break;
	case(CalibrationMethod::Passive):
		for (auto& img : pattern_gamma) {
			calibrated.emplace_back(m_calibration->applyCalibration(method, img));
		}
		break;
	case(CalibrationMethod::Bias_Passive):
		for (auto& img : pattern_gamma) {
			calibrated.emplace_back(m_calibration->applyCalibration(method, img));
		}
		break;
	default:
		calibrated = pattern_gamma;
		break;
	}*/

	show_norm(calibrated, "Calibrated? ");

	// ------- Homogrpahy is used --------
	if (operation == Warping::homography) {

		// Try to calculate impact of luminance 
		std::vector<cv::Mat> pattern_luminance;
		if (luminance) {
			for (const auto& img : pattern_gamma) {
				pattern_luminance.emplace_back(
					m_img_processing->simulate_luminance(
						img,
						distance,
						dispaly_pixel_pitch,
						display_shift_x,
						display_shift_y,
						display_tilt_x,
						display_tilt_y)
				);
			}
		}
		else pattern_luminance = pattern_gamma;

		show_norm(pattern_luminance, "pattern_luminance");

		std::vector<cv::Mat> pattern_smoothed;

		if (smoothing) {
			for (const auto& img : pattern_luminance) {
				pattern_smoothed.emplace_back(apply_ApertureSmoorting(
					img,
					aperture_number,
					distance,
					dispaly_pixel_pitch,
					object_height,
					image_height
				));
			}
		}
		else pattern_smoothed = pattern_luminance;

		show_norm(pattern_smoothed, "pattern_smoothed");

		std::vector<cv::Mat> camera_quantized;
		if (camera_quantization) {
			for (const auto& img : pattern_smoothed) {
				camera_quantized.emplace_back(m_img_processing->quantizeImage(
					img,
					255.0,
					0.0
				));
			}
		}
		else camera_quantized = pattern_smoothed;

		show_norm(camera_quantized, "camera_quantized");

		// ------------------  Gray Val Calibration -> here active Camera Calibration  ----------------------

		std::vector<cv::Mat> pattern_warped;

		if (warping) {
			for (const auto& img : camera_quantized) {
				pattern_warped.emplace_back(warpImage(
					img,
					dest_width,
					dest_height,
					homography_points));
			}
		}
		else pattern_warped = pattern_smoothed;

		show_norm(pattern_warped, "pattern_warped");

		return pattern_warped;
	}

	// ------------------ Raycasting --------------------
	if (operation == Warping::raycasting) {

		CV_Assert(cameraMatrix.has_value());
		CV_Assert(cameraMatrix.value().size() == 2);
		cv::Mat cam_matrix = cameraMatrix.value()[0];
		cv::Mat dist_coeffs = cameraMatrix.value()[1];

		cv::Mat_<cv::Vec2d> coordinateImage =
			m_pattern->generatecoordianteImg(
				dest_width,
				dest_height);

		cv::Mat_<cv::Vec3d> rayImage =
			m_img_processing->calulateRays(
				cam_matrix,
				dist_coeffs,
				coordinateImage
			);

		cv::Mat displayPixelInCameraCoordinates =
			calcDisplayPointsinCameraCoordiantes(
				cv::Size(pattern_width, pattern_height),
				distance,
				display_shift_x,
				display_shift_y,
				display_tilt_x,
				display_tilt_y,
				dispaly_pixel_pitch
			);

		std::vector<cv::Mat> images = createImageFromRays(
			rayImage,
			displayPixelInCameraCoordinates,
			pattern_gamma
		);

		show_norm(images, "pattern_smoothed");

		m_img_store->saveRole(FrameRole::Debug, "C:/Users/grein/Desktop");
		return images;

	}
}