#include "CameraSimulation.hpp"
#include <vector>
#include <opencv2/core.hpp>
#include "imgProcessing.hpp"
#include <cmath>


struct CameraSimulation::Impl {
	std::vector<cv::Mat> images{};
};

CameraSimulation::~CameraSimulation() = default;

CameraSimulation::CameraSimulation(CameraSimulation&&) noexcept = default;

CameraSimulation::CameraSimulation(ImageProcessing& img_processing) 
	:m_img_processing{ img_processing }
{
	m_impl = std::make_unique<Impl>();
}

std::vector<cv::Mat> CameraSimulation::simulate(
	CameraSimulationConfig& config)
{
	extractImages(config);

	// --- Gamma distortion ---
	if (config.disp.gamma != 1.0) {
		CV_Assert(config.disp.gamma >= 1.0);
		for (auto& img : m_impl->images) {
			// Be carefull: This function is defaulted to UniformRowsCols{} as third paraemter.
			// For Phase Shift and GrayCode Analysis this is fine (Horizontal and Veritcal Patterns)
			// If this is not the case, set the third parameter of do_gamma_distortion(.., .. , PerElement{});
			img = m_img_processing.do_gamma_distortion(
				config.disp.gamma,
				img,
				PerElement{});
		}
	}

	// --- Apertur Smoothing ---
	if (config.camera.apertureSmoothing == true) {
		applyApertureSmoothing(config);
	}

	// --- Quantization ---
	if (config.disp.displayQuantization == true) {
		for (auto& img : m_impl->images) {
			img = m_img_processing.quantizeImage(img);
		}
	}

	CV_Assert(m_impl->images.begin() != m_impl->images.end());

	
	const cv::Size disp_size{ m_impl->images[0].size()};
	const cv::Size cam_size{ config.camera.pixel_x, config.camera.pixel_y };
	cv::Mat_<cv::Vec2d> coordinatedImage =
		generateCoordinateImage(cam_size);

	cv::Mat_<cv::Vec3d> rays =
		m_img_processing.calulateRays(
			config.camera.camera_mat,
			config.camera.dist_coeffs,
			coordinatedImage
		);

	cv::Mat displayPixelInCameraCoordinates =
		calcDisplayPointinCameracoordinates(
			disp_size,
			config.scene.disp_shift_z,
			config.scene.disp_shift_x,
			config.scene.disp_shift_y,
			config.scene.disp_tilt_x,
			config.scene.disp_tilt_y,
			config.disp.pixelPitch
		);

	// Here the Hitpoints are calculated for surface without bondaries 
	std::vector<cv::Mat> hitpoints_on_disp_surface =
		m_img_processing.calculateHitPoints(rays, displayPixelInCameraCoordinates);

	cv::Mat hitpoints = m_img_processing.mapHitPointsToDisplayCoords(
		hitpoints_on_disp_surface[0],
		displayPixelInCameraCoordinates
	);
		
	auto start = m_impl->images.begin();
	auto end = m_impl->images.end();

	/*for (auto it = start; it != end; ++it) {
		cv::Mat img = m_img_processing.createImageFromHitpointCoordinates(
			hitpoints,
			*it
		);
		*it = img;
	}*/

	remapFromHitpoints(start, end, hitpoints);

	if (config.scene.luminance == true) {
		CV_Assert(hitpoints_on_disp_surface[1].type() == CV_64F);	
		cv::Mat scaling_img(hitpoints_on_disp_surface[1].size(), CV_64F);
			
		cv::parallel_for_(cv::Range(0, hitpoints_on_disp_surface[1].rows),
			[&](const cv::Range& range)
			{
				for (int row = range.start; row < range.end; ++row) {
					const double* angel_ptr = hitpoints_on_disp_surface[1].ptr<double>(row);
					double* scaling_ptr = scaling_img.ptr<double>(row);
					for (int cols = 0; cols < hitpoints_on_disp_surface[1].cols; ++cols) {
						scaling_ptr[cols] = std::cos(angel_ptr[cols]);
					}
				}
			});
			
		for (auto it = start; it != end; ++it) {
			cv::Mat img;
			cv::multiply(*it, scaling_img, img);
			*it = img;
		}
	}

	// Quantize from the camera values 
	if (config.camera.quantization == true)
	{
		for (auto& img : m_impl->images) {
			img = m_img_processing.quantizeImage(img);
		}
	}

	for (auto it = start; it != end; ++it)
	{
		// Throw Exception if values are out of bounds. 
		cv::checkRange(*it, false, nullptr, 0, 256);
		it->convertTo(*it, CV_8U);
	}

	// If an output container is specified than fill it
	// Can be used to directly write in the accoridng frameStore class
	if (config.data.output != nullptr) {
		for (auto it = start; it != end; ++it) {
			config.data.output->push_back(*it);
		}
	}

	return m_impl->images;
}

std::vector<cv::Mat> CameraSimulation::remapFromHitpoints(
	const std::vector<cv::Mat>::iterator start,
	const std::vector<cv::Mat>::iterator end,
	const cv::Mat_<cv::Vec2d>& hitpoints) 
{
	CV_Assert(start != end);

	cv::Mat_<cv::Vec2f> hitpoints32f(hitpoints);  // Converto to float

	std::vector<cv::Mat> channels(2), out;

	cv::split(hitpoints32f, channels);

	for (auto it = start; it != end; ++it) {
		cv::Mat remapped;
		cv::remap(*it, remapped, channels[0], channels[1], cv::INTER_LINEAR);
		*it = remapped;
		out.push_back(remapped);
	}
	return out;
}


cv::Mat CameraSimulation::calcDisplayPointinCameracoordinates(
	const cv::Size& sz,
	const double shift_z,
	const double shift_x,
	const double shift_y,
	const double tilt_x,
	const double tilt_y,
	const double pixel_pitch)
{
	CV_Assert(sz.area() > 0);
	CV_Assert(shift_z > 0);
	CV_Assert(std::abs(tilt_x) <= CV_PI /4.0);
	CV_Assert(std::abs(tilt_y) <= CV_PI / 4.0);

	cv::Mat displayPixel_inCameraCoordinates =
		generateCoordinateImage(sz, 3);

	//// Shift the image coordiante System in the middle of the Display
	//const double shift_x =
	//	static_cast<double>(displayPixel_inCameraCoordinates.cols - 1) / 2.0;
	//const double shift_y =
	//	static_cast<double>(displayPixel_inCameraCoordinates.rows - 1) / 2.0;

	//displayPixel_inCameraCoordinates -= cv::Scalar(shift_x, shift_y, 0.0);

	displayPixel_inCameraCoordinates *= pixel_pitch;

	cv::Mat rot_display_coordiantes =
		m_img_processing.rotateCoordinatedGrid(
			displayPixel_inCameraCoordinates,
			cv::Vec3d(tilt_x, tilt_y, 0.0),
			Rotation::rodrigeuz
		);

	cv::Mat rot_shifted_coordinates =
		m_img_processing.shiftCoordinateGrid(
			rot_display_coordiantes,
			cv::Vec3d(shift_x, shift_y, shift_z)
		);

	return rot_shifted_coordinates;
}



cv::Mat CameraSimulation::generateCoordinateImage(
	const cv::Size& sz,
	const int dimension)
{
	CV_Assert(sz.area() > 0);
	CV_Assert(dimension == 2 || dimension == 3);

	cv::Mat coordinate_img(sz, CV_64FC2);

	cv::parallel_for_(cv::Range(0, sz.height),
		[&](const cv::Range& range) {
			for (int row = range.start; row < range.end; ++row) {
				cv::Vec2d* row_ptr = coordinate_img.ptr<cv::Vec2d>(row);
				for (int cols = 0; cols < sz.width; ++cols) {
					row_ptr[cols] = cv::Vec2d(cols, row);
				}

			}
		});

	if (dimension == 3) {
		cv::Mat dimension3 = cv::Mat::zeros(sz, CV_64F);
		cv::Mat in[2] = { coordinate_img, dimension3 };
		int fromTo[] = { 0,0 , 1,1 , 2,2 };
		cv::Mat out(sz, CV_64FC3);
		
		cv::mixChannels(in, 2, &out, 1, fromTo, 3);
		return out;
	}
	return coordinate_img;
}




cv::Mat CameraSimulation::createCircularBinaryMask(
	const int diameter)
{
	CV_Assert(diameter >= 1);
	CV_Assert(diameter % 2 == 1); // Diameter must be odd.

	cv::Mat se(diameter, diameter, CV_64F, cv::Scalar(0));

	for (int y = 0; y < diameter; ++y) {
		for (int x = 0; x < diameter; ++x) {
			double dx = x - (diameter / 2); // Shift origin in the middle 
			double dy = y - (diameter / 2);
			if (dx * dx + dy * dy <= (diameter / 2) * (diameter/2))
				se.at<double>(y, x) = 1; // oder 255
		}
	}
	return se;
}


void CameraSimulation::extractImages(
	const CameraSimulationConfig& config)
{
	CV_Assert(config.data.begin != config.data.end);
	
	for (auto it = config.data.begin; it != config.data.end; ++it) {
		m_impl->images.push_back(it->clone());
	}
	return;
}

std::vector<cv::Mat>& CameraSimulation::applyApertureSmoothing(
	const CameraSimulationConfig& config)
{
	CV_Assert(config.camera.f_number > 0.0);
	CV_Assert(config.camera.sensor_size > 0.0);
	CV_Assert(config.scene.object_size > 0.0);

	int confusion_circle_in_disp_pix{};

	if (config.camera.circle_of_confusion_n_disp == 0) {
		// Used to calculate the size of the confusion circle
		const double scale = config.camera.sensor_size / config.scene.object_size;
		const double distance = cv::norm(
			cv::Vec3d(config.scene.disp_shift_z, config.scene.disp_shift_y / 2, config.scene.disp_shift_x / 2));
		const double focal_length = distance * scale / (scale + 1); // g = (m+1)/m * f
		const double circ_entrance_pupil =   // = double confusion_circle_diamter
			focal_length / config.camera.f_number; // f# = f/D -> D = entrance pupil 
		confusion_circle_in_disp_pix =
			static_cast<int>(std::ceil(circ_entrance_pupil / config.disp.pixelPitch));
	}
	else
		confusion_circle_in_disp_pix = config.camera.circle_of_confusion_n_disp;

	if (!(confusion_circle_in_disp_pix % 2)) ++confusion_circle_in_disp_pix;

	// Get circulat BinaryMask
	cv::Mat circular_binary = 
		createCircularBinaryMask(confusion_circle_in_disp_pix);
	const double sum = cv::sum(circular_binary)[0];

	// Normalize the Mask 
	circular_binary /= sum;

	for (auto it = m_impl->images.begin(); it != m_impl->images.end(); ++it)
	{
		cv::Mat smoothed;
		cv::filter2D(*it, smoothed, CV_64F, circular_binary);
		*it = smoothed;
	}

	return m_impl->images;
}


