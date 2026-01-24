#ifndef	DEFLECTOMETRY_H
#define DEFLECTOMETRY_H
#include <memory>
#include <vector>
#include <exception>
#include <iostream>
#include <cassert>
#include <thread>
#include <opencv2/opencv.hpp>
#include "GrayCalibVector.hpp"
#include <numeric>
#include "utils.hpp"
#include <array>
#include "enums.hpp"



//Forward Decleration Enums + Class

enum class FrameRole;
enum class UnwrapMode;
enum class ReferenceMode;

class Pattern;
class AcquisitionWorker;
class ImageProcessing;
class ImageStore;
class ScreenDisplay;
class GrayCalibration;

namespace defl {
	class AcquisitionController;
	struct PhaseShiftConfig;
	struct CameraConfig;
}


class Deflectometry {
public:
	 Deflectometry();
	 ~Deflectometry();

	 cv::Mat getMask(
		 const std::vector<cv::Mat>&,
		 const double thresh,
		 const bool dilate
	 );

	 // Tries to setup the Calibration class for later use

	 bool setupCalibration(
		 const CalibrationMethod method,
		 const std::string& path
	 );

	 // Method tries to acquiere config files if not available
	 // and saves at the selected Path in a XML file for each config. 
	 bool logging(
		 const std::string& paht);

	 // Methods tries to find Parallelogram corners of a Binary mask
	 // The corners are sorted in TL TR DL DR order! 
	 // The method can fail if more than 4 corners are found. 
	 // If happens: go into the cv::findcontours and change Epsilon. 
	 std::vector<std::pair<double,double>> findParallelogramCorners(const cv::Mat& bin);


	 std::vector<cv::Vec2d> getReferencePoint(
		 const std::vector<cv::Mat>& img,
		 const ReferenceMode mode,
		 const cv::Mat& mask
	 );

	 cv::Mat undistortImageManuell(
		 const cv::Mat& img,
		 const cv::Mat& cam,
		 const cv::Mat& dist_coeffs
	 );

	 GrayCalibVector borderPoints_gray_val(
		 const int pixel_x,
		 const int pixel_y,
		 const int sections_x = 0,
		 const int sections_y = 0
	 );

	 cv::Vec2d projectPoint_homography(
		 const cv::Vec2d& img,
		 const cv::Mat& homography
	 );

	 cv::Mat get_Homogrpahy_mat(
		 const std::vector < std::pair<int, int>>& src,
		 const std::vector < std::pair<double, double>>& dst,
		 const int method = 1
	 );

	 cv::Mat get_Homogrpahy_mat(
		 const std::vector<cv::Vec2d>& srcPoints,
		 const std::vector<cv::Vec2d>& dstPoints,
		 const int method = 1
	 );

	 // Method for connecting the camera, on user Input camera saves frame and stores it in the given FrameRole.
	 std::vector<cv::Mat> getFrames(FrameRole);

	 std::vector<cv::Mat> getFrames(FrameRole, cv::Mat& img);

	 //1280 x 1024 IDS
	 std::array<double, (std::size_t)3> doWhiteBalance(
	 const std::vector<cv::Mat>&,
	 int roi_x = 250,
	 int roi_y = 250,
	 int roi_width = 200,
	 int roi_height = 200);

	 std::vector<cv::Vec2d> distortionPipelineTest(
	 const cv::Mat& mat,
	 const cv::Mat& dist,
	 const std::vector<cv::Vec2d>);
	 
	 // Method to Acquire the phase Shifted pictures. 
	 // Shift_mode: classifies wich shift mode is used 4 Shift mode or user defined
	 // n_pics: how many pictures per Phase picture
	 // save: Boolean value if the pictures should be saved. 
	 // number_of_Periods: Classfies the wavelength of the pattern by specifing the number
	 // of Waves on the y-axis
	 std::vector<cv::Mat> do_phase_measurement(Shift_mode,
		 int n_pics,
		 bool save,
		 const std::string& save_path,
		 int number_of_perdios = 7);

	 std::vector<cv::Mat> do_wrapped_phase(
		 const std::vector<cv::Mat>& vec,
		 int n_pics_perPhase,
		 int n_shifts,
		 bool save,
		 const std::string& path_save);

	 std::vector<cv::Mat> do_unwrapped_phase(
		 const std::vector<cv::Mat>& wrapped,
		 const cv::Mat& mask,
		 UnwrapMode mode,
		 bool save,
		 const std::string& save_path
	 );

	 std::vector<cv::Mat> do_reprojection(
		 const std::vector<cv::Mat>& unwrapped,
		 const cv::Mat& mask, 
		 const cv::Vec2d refPoint,
		 const cv::Mat& cam_Matrix,
		 const cv::Mat& dist_Coeffs,
		 const double wavelength,
		 const int grid_points_x,
		 const int grid_points_y,
		 const double pixel_pitch,
		 bool save, 
		 const std::string& save_path,
		 const double screen_width,
		 const double screen_height
	 );

	 cv::Mat generate_reference_Pattern(
		 const ReferenceMode mode,
		 const int n_pics,
		 bool save,
		 const std::string& path
	 );

	 // Function to create GrayValue calibration. 
	 // n_pics_per_value: the ammount of pictures taken per gray value
	 // save_path: The path where the LUT is stored as a .csv file
	 bool do_grayvalue_calibration(
		 const int n_pics_per_value,
		 const int n_steps, 
		 bool save,
		 const std::string& save_path);

	 

	 GrayCalibVector calc_response_curve_sections(
		 const int n_pics_per_value,
		 const int n_steps,
		 bool save,
		 const std::string& save_path,
		 const int sections_x,
		 const int sections_y
	 );

	 std::vector<cv::Mat> acquire_img(
		 const std::vector<cv::Mat>&,
		 const FrameRole dst,
		 const int n_pics_per_val
	 );


	 std::vector<cv::Mat> acquire_img(
		 const FrameRole src,
		 const FrameRole dst,
		 const int n_pics_per_value
	 );

	 bool do_camera_calibration();

	 void load(FrameRole, const std::string&);

	 std::vector<cv::Mat> get(FrameRole role);

	 cv::Mat calcDistortionError(const cv::Mat& img);

	 cv::Mat generateCoordinateImg(
		 bool save,
		 const std::string& path);

	 // Does Setup the Pattern class. 
	 // Pattern class creates the Pattern like phase shift pattern and grayVal 
	 // Pattern class needs a shared Pointer to a img_store class
	 void setupPattern(
		 ImageStore& img_store,
		 int pixelX = 1920,
		 int pixelY = 1080);


	 void saveVecImage(
		 const FrameRole,
		 const std::vector<cv::Mat>&,
		 bool save,
		 const std::string& path
	 );

	 cv::Mat subtractSurface(
		 const cv::Mat& img,
		 const cv::Mat& mask,
		 bool show_subtracted = false
	 );

	 void saveSingleImage(
		 const FrameRole,
		 const cv::Mat&,
		 bool save,
		 const std::string& path);

	 std::vector<cv::Mat> generatePattern(
		 const Shift_mode mode,
		 const CalibrationMethod method,
		 const std::string& calib_path,
		 const FrameRole dst,
		 bool save, 
		 const std::string& path);

	 std::vector<cv::Mat> generateCartesian(
		 bool save,
		 const std::string& path,
		 int gridX,
		 int gridY);

	 // calculates the element wise difference of a two images and return the result. 
	 // when save value is set to true the value is saved in Debug FrameRole of the ImageStore
	 cv::Mat get_difference_debug(const cv::Mat& mat1, const cv::Mat& mat2, bool save = false); 

	 cv::Mat undistortImage(const cv::Mat&, const cv::Mat& cam_Matrix, const cv::Mat& dist_coeffs);

	 cv::Mat distortImage_manual(const cv::Mat& img, const cv::Mat& cam_Matrix, const cv::Mat& dist_coeffs);

	 // This should in Future be use instead of the many forwarding functions to image Processing
	 // Sadly this has to be done at another time 
	 ImageProcessing& processing() const {
		 return *m_img_processing;
	 };

	 // Same here for the pattern class but this should be also done at another time. 
	 Pattern& img_generation() const {
		 return *m_pattern;
	 }


	 // Creates syntehical images of the scene by doing a homography and smoothing acoording to the
	 // circle of confusion. 
	 // If no smoothing should be done set Image height to zero!
	 // If no homogrpahy should be done. set destWidth or destheight to zero!
	 std::vector<cv::Mat> createSyntheticalImages(
		 const double image_height = 400,  // Mirror circumference  
		 const int dest_width = 2464,      // Mako G-507-B width
		 const int dest_height = 2056,     // Mako G-507-B height
		 const double display_pixel_pitch = 0.2745,                //PixelPitch  FH 0.277    BMZ: 
		 const Shift_mode mode = Shift_mode::four_phase_shift,
		 const CalibrationMethod method = CalibrationMethod::None,
		 const int display_pixel_x = 1920,
		 const int display_pixel_y = 1080,
		 const int n_periods_in_y = 10,
		 const double aperture_number = 2.4,
		 const double distance = 3200,     // f = 1600 distance 2*f
		 const double object_height = 6.6, // 2/3" Sensor 8,8 * 6,6 
		 const RoiBorders<double> destination = {
		 {200, 200},   // left up corner
		 {1800, 180},  // left down corner,
		 {210, 2200},  // right up corner,
		 {1900, 1900}, // right down corner
		 }
		 );
	 

private:
	std::unique_ptr<ImageStore> m_img_store{ nullptr }; // Must be shared_ptr
	std::unique_ptr<Pattern> m_pattern{nullptr}; // could be Unique
	std::unique_ptr<AcquisitionWorker> m_acquisition_worker{ nullptr };  // Must be shared
	std::unique_ptr<ImageProcessing> m_img_processing{ nullptr };  // could be unqiue
	std::unique_ptr<ScreenDisplay> m_screenDisplay{ nullptr };  // Better shared 
	std::unique_ptr<defl::AcquisitionController> m_acquisition_controller{ nullptr };  // better shared
	std::unique_ptr<GrayCalibration> m_calibration{ nullptr };

	
	// If Curly brackets for default initialization are used forward decleration breaks
	std::vector<std::unique_ptr<defl::PhaseShiftConfig>> m_pattern_config;
	std::vector<std::unique_ptr<defl::CameraConfig>> m_camera_config;

	std::pair<double, double> fitLine1D(const std::vector<double>& y);

	// Methods to connect and Setup Hardware -> Camera and Acquisitionworker class get Setup
	bool init();



	cv::Mat createRealisticFromPattern(
		const cv::Mat& pattern,
		const double aperture_number,
		const double display_pixel_pitch = 0.2745, //PixelPitch  FH 0.277    BMZ: 
		const double distance = 3200,     // f = 1600 distance 2*f
		const double object_height = 6.6, // 2/3" Sensor 8,8 * 6,6 
		const double image_height = 400,  // Mirror circumference  
		const int dest_width = 2464,      // Mako G-507-B width
		const int dest_height = 2056,     // Mako G-507-B height
		const RoiBorders<double> destination = {
			{200, 200},   // left up corner
			{1800, 180},  // left down corner,
			{210, 2200},  // right up corner,
			{1900, 1900}, // right down corner
		}
		);

	// Class to disconnect from Hardware -> Camera and Acquisitionworker get set to nullptr
	bool disconnect();

	bool check_synthaticall_points(const std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>>&);
	// --- End -----


	void saveSliceToCSV(const std::string& filename,
		const std::vector<double>& unwrap,
		const std::pair<double, double>& unwrapFit,// regressions a,b unwrap
		const std::vector<double>& repro,
		const std::pair<double, double>& reproFit);
	

};

#endif 