#ifndef	DEFLECTOMETRY_H
#define DEFLECTOMETRY_H
#include <memory>
#include <vector>
#include <exception>
#include <iostream>
#include <cassert>
#include <thread>
#include <opencv2/opencv.hpp>

//Forward Decleration Enums + Class

enum class Shift_mode;
enum class FrameRole;
enum class UnwrapMode;
enum class ReferenceMode;

class Pattern;
class AcquisitionWorker;
class ImageProcessing;
class ImageStore;
class ScreenDisplay;
namespace defl {
	class AcquisitionController;
	class PhaseShiftConfig;
	class CameraConfig;
}

class Deflectometry {
public:
	 Deflectometry();

	 //void phase_unwrap(bool save, const std::string& path);

	 //void TestOptimal();
	 //
	 //void camera_calibration(int camera, std::string image_path);
	

	 //// Stores 8 bit images in .png / .jpg format. Images have to be given in in a uchar 8 bit with 1 or 3 channels, in 
	 //// a vector<cv::Mat> format and the address where the files need to be stored. 
	 //// If floating point images are given to this function, cv::imwrite will try to save them which leads to data loss. 
	 //void save_frames(std::vector<cv::Mat>& frames, const std::string& path);

	 //void calc_reproject_error(bool visualizing = true, bool saving = false, const std::string& path = "");

	 //void load_frames(const std::string& path);
	 //void load_calib(std::string path = "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-12_camera_calib.xml");

	 //void load_gray_value_calib(const std::string& path);

	 //void gray_value_apply();

	 //void extract_Column(std::string path);
	 
	 //void extract_Line(std::string path);

	 //std::pair<double, double> fitLine1D(const std::vector<double>& y);
	 //

	 cv::Mat undistortImageManuell(
		 const cv::Mat& img,
		 const cv::Mat& cam,
		 const cv::Mat& dist_coeffs
	 );

	 std::vector<cv::Mat> getFrames(FrameRole);

	 //1280 x 1024
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
		 const std::vector<cv::Mat>& contrast,
		 UnwrapMode mode,
		 bool save,
		 const std::string& save_path
	 );



	 std::vector<cv::Mat> do_reprojection(
		 const std::vector<cv::Mat>& unwrapped,
		 const std::vector<cv::Mat>& contrast, 
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

	 cv::Mat do_reference_Pattern(
		 const ReferenceMode mode,
		 const int n_pics,
		 bool save,
		 const std::string& path
	 );

	 // Function to create GrayValue calibration. 
	 // n_pics_per_value: the ammount of pictures taken per gray value
	 // save_path: The path where the LUT is stored as a .csv file
	 bool do_grayvalue_calibration(
		 int n_pics_per_value,
		 std::string save_path);

	 bool do_camera_calibration();

	 void load(FrameRole, const std::string&);

	 std::vector<cv::Mat> Deflectometry::get(FrameRole role);

	 cv::Mat calcDistortionError(const cv::Mat& img);

	 cv::Mat generateCoordinateImg(
		 bool save,
		 const std::string& path);

	 void setupPattern(
		 std::string path = "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-22",
		 bool useLUT = true,
		 int pixelX = 1920,
		 int pixelY = 1080);

	 /*void loadPhaseConfig(const std::string& path);*/

	 void saveVecImage(
		 const FrameRole,
		 const std::vector<cv::Mat>&,
		 bool save,
		 const std::string& path
	 );

	 void saveSingleImage(
		 const FrameRole,
		 const cv::Mat&,
		 bool save,
		 const std::string& path);

	 std::vector<cv::Mat> generatePattern(
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

private:
	std::shared_ptr<ImageStore> m_img_store{ nullptr };
	std::shared_ptr<Pattern> m_pattern{nullptr};
	std::shared_ptr<AcquisitionWorker> m_acquisition_worker{ nullptr };
	std::shared_ptr<ImageProcessing> m_img_processing{ nullptr };
	std::shared_ptr<ScreenDisplay> m_screenDisplay{ nullptr };
	std::shared_ptr<defl::AcquisitionController> m_acquisition_controller;
	
	std::vector<std::shared_ptr<defl::PhaseShiftConfig>> m_pattern_config{};
	std::vector<std::shared_ptr<defl::CameraConfig>> m_camera_config{};

	// Methods to connect and Setup Hardware -> Camera and Acquisitionworker class get Setup
	bool init();

	// Class to disconnect from Hardware -> Camera and Acquisitionworker get set to nullptr
	bool disconnect();

	//bool capture_single_frame(FrameRole role);

	bool check_synthaticall_points(const std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>>&);
	// --- End -----


	//std::vector<cv::Mat> m_optimalFrames;
	//std::vector<cv::Mat> m_optimalPhase;

	//void saveSliceToCSV(const std::string& filename,
	//	const std::vector<double>& unwrap,
	//	const std::pair<double, double>& unwrapFit,// regressions a,b unwrap
	//	const std::vector<double>& repro,
	//	const std::pair<double, double>& reproFit);
	
};

#endif 