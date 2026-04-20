#include "CameraSimulation.hpp"
#include <vector>
#include <opencv2/core.hpp>
#include "imgProcessing.hpp"
#include <cmath>
#include <cstddef>

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

	CV_Assert(m_impl->images.begin() != m_impl->images.end());

	// --- Gamma distortion ---
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

	// Scaling and Bias
	applyScalingAndBias(config);

	// --- Quantization ---
	if (config.disp.quantization == true) {
		for (auto& img : m_impl->images) {
			img = m_img_processing.quantizeImage(img);
		}
	}

	const cv::Size disp_size{ m_impl->images[0].size()};
	const cv::Size cam_size{ config.camera.pixel_x, config.camera.pixel_y };
	cv::Mat_<cv::Vec2d> coordinatedImage =
		generateCoordinateImage(cam_size);


	// Ray Calculation
	cv::Mat_<cv::Vec3d> rays =
		m_img_processing.calulateRays(
			config.camera.camera_mat,
			config.camera.dist_coeffs,
			coordinatedImage
		);
	
	// Create a Matrix containing the starting Points for the rays.
	cv::Mat ray_origin = cv::Mat(rays.size(), CV_64FC3, cv::Scalar(0, 0, 0));

	if (config.mirror.contains_mirror == true) {

		// Calculate Discrete MirrorPoints in Cameracoordinates
		cv::Mat mirrorPointsinCameraCoordiantes =
			calcDisplayPointinCameracoordinates(
				config.mirror.mirror_sz,
				config.scene.mirror_shift_z,
				config.scene.mirror_shift_x,
				config.scene.mirror_shift_y,
				config.scene.mirror_tilt_x,
				config.scene.mirror_tilt_y,
				1.0 
			);

		std::vector<cv::Mat> hitpoints_on_mirror_surface =
			m_img_processing.calculateHitPoints(
				rays, 
				mirrorPointsinCameraCoordiantes,
				ray_origin);

		cv::Mat hitpoints_mir = m_img_processing.mapHitPointsToDisplayCoords(
			hitpoints_on_mirror_surface[0],
			mirrorPointsinCameraCoordiantes
		);

		cv::Mat hitpoints_on_mirror = hitpoints_on_mirror_surface[0].clone();
		std::vector<cv::Mat> channels;
		cv::split(hitpoints_mir, channels);

		cv::Mat mask = (channels[0] == -1.0) & (channels[1] == -1.0);

		hitpoints_on_mirror.setTo(cv::Vec3d(0, 0, 0), mask);

		ray_origin = hitpoints_on_mirror;

		rays.setTo(cv::Vec3d(-1.0, -1.0, -1.0), mask);

		cv::Vec3d rvec(config.scene.mirror_tilt_x, config.scene.mirror_tilt_y, 0);
		cv::Mat rot;

		cv::Rodrigues(rvec, rot);
		cv::Vec3d surface_normal = rot.col(2);

		mirrorRays(rays, surface_normal);
	}

	cv::Mat displayPixelInCameraCoordinates =
		calcDisplayPointinCameracoordinates(
			disp_size,
			config.scene.disp_shift_z,
			config.scene.disp_shift_x,
			config.scene.disp_shift_y,
			config.scene.disp_tilt_x,
			config.scene.disp_tilt_y,
			config.disp.pixelPitch,
			true 
		);

	// Here the Hitpoints are calculated for surface without bondaries 
	std::vector<cv::Mat> hitpoints_on_disp_surface =
		m_img_processing.calculateHitPoints(
			rays,
			displayPixelInCameraCoordinates,
			ray_origin);

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

	// --- Apertur Smoothing ---
	if (config.camera.apertureSmoothing == true) {
		applyApertureSmoothing(config);
	}

	// Quantize from the camera values 
	if (config.camera.quantization == true)
	{
		for (auto& img : m_impl->images) {
			img = m_img_processing.quantizeImage(img);
		}
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

std::vector<cv::Mat>& CameraSimulation::applyScalingAndBias(
	const CameraSimulationConfig& config)
{
	CV_Assert((config.disp.scaling <= 1.0) && (config.disp.scaling > 0.0));
	CV_Assert((config.disp.bias >= 0) && (config.disp.bias < 200.0));

	for (auto& img : m_impl->images) {
		cv::Mat img64;
		if (img.type() != CV_64F) {
			img.convertTo(img64, CV_64F);
		}
		else img64 = img;

		img64 *= config.disp.scaling;

		img64 += config.disp.bias;

		cv::checkRange(img64, false, nullptr, 0 - 1E-6, 255 + 1E6);

		img = img64;
	}
	return m_impl->images;
}

void CameraSimulation::mirrorRays(
	cv::Mat_<cv::Vec3d>& rays,
	const cv::Vec3d surface_normal)
{
	cv::Vec3d n = surface_normal / cv::norm(surface_normal);
	cv::Mat reflectionMat = (cv::Mat::eye(cv::Size(3, 3), CV_64F) - 2 * n * n.t());
	cv::parallel_for_(cv::Range(0, rays.rows),
		[&](const cv::Range& range)
		{
			for (int row = range.start; row < range.end; ++row) {
				cv::Vec3d* ray_ptr = rays.ptr<cv::Vec3d>(row);
				for (int col = 0; col < rays.cols; ++col) {
					if (ray_ptr[col].val[0] == -1.0 &&
						ray_ptr[col].val[1] == -1.0 &&
						ray_ptr[col].val[2] == -1.0) continue;
					cv::Mat reflected = reflectionMat * ray_ptr[col];
					ray_ptr[col] = cv::Vec3d(reflected);
				}
			}
		});
}

std::vector<cv::Mat> CameraSimulation::remapFromHitpoints(
	const std::vector<cv::Mat>::iterator start,
	const std::vector<cv::Mat>::iterator end,
	const cv::Mat_<cv::Vec2d>& hitpoints) 
{
	CV_Assert(start != end);

	cv::Mat_<cv::Vec2f> hitpoints32f(hitpoints);  // Converto to float

	int n_pics = static_cast<int>(std::distance(start, end));

	std::vector<cv::Mat> channels(2), out(n_pics);

	cv::split(hitpoints32f, channels);

	cv::parallel_for_(cv::Range(0, n_pics),
		[&](const cv::Range& range)
		{
			for (int i = range.start; i < range.end; ++i) {
				cv::Mat remapped;
				std::vector<cv::Mat>::iterator it = std::next(start, i);
				cv::remap(*it, remapped, channels[0], channels[1], cv::INTER_LINEAR);
				*it = remapped;
				out[i] = remapped;
			}
		}
		);

	//for (auto it = start; it != end; ++it) {
	//	const cv::Size sz = it->size();
	//	cv::Mat img(hitpoints.size(), CV_64F);
	//	cv::parallel_for_(cv::Range(0, hitpoints.rows),
	//		[&](const cv::Range& range) {
	//			for (int row = range.start; row < range.end; ++row) {
	//				//double* img_ptr = it->ptr<double>(row);
	//				const cv::Vec2d* map_ptr = hitpoints.ptr<cv::Vec2d>(row);
	//				double* img_ptr_out = img.ptr<double>(row);
	//				for (int col = 0; col < hitpoints.cols; ++col) {
	//					img_ptr_out[col] = m_img_processing.bilinearInterpolation(
	//						*it, cv::Vec2d(map_ptr[col].val[1], map_ptr[col].val[0]));
	//				}
	//			}
	//		}
	//	);
	//	*it = img;
	//}

	return out;
}


cv::Mat CameraSimulation::calcDisplayPointinCameracoordinates(
	const cv::Size& sz,
	const double shift_z,
	const double shift_x,
	const double shift_y,
	const double tilt_x,
	const double tilt_y,
	const double pixel_pitch,
	bool flip_vertical)
{
	CV_Assert(sz.area() > 0);
	CV_Assert(shift_z >= 0);
	CV_Assert(std::abs(tilt_x) <= CV_PI);
	CV_Assert(std::abs(tilt_y) <= CV_PI);

	cv::Mat displayPixel_inCameraCoordinates =
		generateCoordinateImage(sz, 3);

	if (flip_vertical) {
		cv::Mat flipped;
		cv::flip(displayPixel_inCameraCoordinates, flipped, 1);
		displayPixel_inCameraCoordinates = flipped;
	}
	

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
	m_impl->images.clear();
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

	int dist = std::distance(m_impl->images.begin(), m_impl->images.end());

	cv::parallel_for_(cv::Range(0, dist),
		[&](const cv::Range& range) {
			for (int index = range.start; index < range.end; ++index) {

				cv::Mat& img = *std::next(m_impl->images.begin(), index);

				cv::Mat temp(img == 0);

				cv::Mat filtered;
				cv::filter2D(img, filtered, CV_64F, circular_binary, { -1,-1 }, 0.0, cv::BORDER_REFLECT);

				filtered.setTo(0, temp);

				// 5. zurückschreiben
				img = filtered;
			}
		});

	
	return m_impl->images;
}


