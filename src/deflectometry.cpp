#pragma once
#include "deflectometry.hpp"
#include "screen.hpp"
#include "acquisitionworker.hpp"
#include "cassert"
#include "enums.hpp"
#include "imageHandler.hpp"
#include "flagHandler.hpp"
#include "imgProcessing.hpp"
#include "camera_calib.hpp"

#include <filesystem>
#include <utility>

// ****Just for Fun ****
//Marcro to check type of expression at compile time 
// (type*)0 creates a null pointer of type 'type*', &expr gets the address of expr
// If expr is not of type 'type', the comparison will fail, causing a compile-time error.
// C++ forbits comparing two pointer of incompatible types. Therefor this will create a compile time error if types do not match.
// The literal 0 gets cast to a pointer of type 'type*', this allows the comparison. 
#define CHECK_TYPE(expr, type) ((void)((type*)0 == &(expr))))

/*
* 
* Very cool demonstratoin of perfect forwarding. 
* Differences between compile time values and runtime values must be understood. 
* 
#include <iostream>
#include <string>
#include <utility>   // for std::forward

// our wrapper template
template<typename F, typename... Args>
auto log_call(F&& f, Args&&... args)
{
	std::cout << "Calling function...\n";
	// perfectly forward f and all args
	return std::forward<F>(f)(std::forward<Args>(args)...);
}

// --- a few functions to test with ---

void greet(const std::string& name)
{
	std::cout << "Hello, " << name << "!\n";
}

void modify(std::string& s)
{
	s += " (modified)";
}

void print_ref(std::string&& s)
{
	std::cout << "Rvalue received: " << s << '\n';
}

int sum(int a, int b)
{
	return a + b;
}

//Access datatype of N-th argument
* By putting the compile time variables through std::forward_as_tuple, we create a tuple of references that preserve the value category (lvalue/rvalue) of each argument.
* This can be accessed. The compiler itself just uses datatypes. Through the usage of a runtime variable, we can access it. 
* Also very intersting: fold expression ((std::cout << args << ' '), ...); // fold expression. This is opened by the compiler to print all arguments.
* ((expression), ...) -> (expr op ...) expands to -> (((expr1 op expr2) op expr3) op ...)
* 
template<std::size_t N, typename... Ts>
decltype(auto) get_arg(Ts&&... ts)
{
	return std::get<N>(std::forward_as_tuple(std::forward<Ts>(ts)...));
}

int main()
{
	std::string name = "Guido";

	//Call with a normal lvalue reference
	log_call(greet, name);

	//Call with an lvalue reference that gets modified
	log_call(modify, name);
	std::cout << "After modify: " << name << '\n';

	//Call with an rvalue
	log_call(print_ref, std::string("temporary"));

	//Call with a function returning a value
	int result = log_call(sum, 10, 20);
	std::cout << "Sum result: " << result << '\n';

	//Call with a lambda
	log_call([](auto&& msg) { std::cout << "Lambda says: " << msg << '\n'; }, "hi");
}


// std::apply is more or less the inverse of std::forward_as_tuple. A small simplified example here:

template<typename F, typename Tuple, std::size_t... I>
decltype(auto) apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>)
{
	// unpack tuple elements into the function call
	return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
}

template<typename F, typename Tuple>
decltype(auto) apply(F&& f, Tuple&& t)
{
	constexpr std::size_t N = std::tuple_size_v<std::decay_t<Tuple>>;
	return apply_impl(
		std::forward<F>(f),
		std::forward<Tuple>(t),
		std::make_index_sequence<N>{} // generates 0,1,2,...,N-1
	);
}

// Also look into the c++ explenation file for the std::reference_wrapper -- insane
// There is a ChatGPT chat for that. 


*/

cv::Mat rotImage180(const cv::Mat&);

//if image should be saved, assign here a handler function that can save the image. 
void showRawImage(const cv::Mat& mat) {
	imgHandler.imshow_Camera(mat);
}


// Constructor Deflectometry() takes no argument. Automatically creates Camera class with ids::peak library. Acuqistionworker inherits from that. 
// If multiple cameras are used these can be choosen by the input Argument of Acquisitionworker. 
// 
// For each camera, a new Acuistionworker instance must be created with the according index. 
// Screen class is created for Fringe projection
Deflectometry::Deflectometry() {
	m_screen = std::make_shared<Screen>();
	m_acquisition_worker = std::make_shared<AcquisitionWorker>(0); 
	m_img_processing = std::make_shared<ImageProcessing>();
}

void Deflectometry::phase_unwrap() {
	m_img_processing->assginFrames(std::move(m_acquisition_worker->m_frames));
	// Calculates the wrapped phase. Pictures are stored in m_acquisition_worker.
	// Wrapped phase, Base Intensity, and Contrast are stored in m_img_processing as members. 
	m_img_processing->wrapped_phase();
	//m_img_processing->goldsteinUnwrap();
	m_img_processing->unwrapped_phase();

}

void Deflectometry::show_acquistion() {
	std::cout << m_acquisition_worker->m_frames.size() << std::endl;
	for (const auto& frame : m_acquisition_worker->m_frames) {
		cv::namedWindow("Raw Phase", cv::WINDOW_NORMAL);
		cv::setWindowProperty("Raw Phase", cv::WINDOW_NORMAL, cv::WINDOW_FREERATIO);
		cv::imshow("Raw Phase", frame);
		cv::waitKey(0);
	}
}

//Returns true if path exists, false if created new directory
auto check_create_dir = [&](auto&& p) -> bool {
	return std::filesystem::exists(p)
		? (std::cout << "Path exists!\n", true)
		: (std::filesystem::create_directory(p), false);
	};


void Deflectometry::save_frames(std::string& path) {
	m_img_processing->saveImages(path);
}

void Deflectometry::load_frames(const std::string& path) {
	m_img_processing->load_frames(path);
}

void Deflectometry::save_frames(std::vector<cv::Mat>& frames, const std::string& path) {
	int counter = 0;
	std::filesystem::path p(path);
	//Use ternary operator with side effect -> comma operator, to achieve same return type
	//std::filesystem::exists(p) ? (std::cout << "Path exists! \n", true) : (std::filesystem::create_directory(p), true); //ternary operator must in each path return same kind of value

	//Or use lambda expressions because have same datatype
	//std::filesystem::exists(p) ? []() {std::cout << "path exists! \n"; } : 
	//	[&]() {std::filesystem::create_directory(p); std::cout << "Path created! \n"; }();

	// Create a callable lambda that checks and creates directory
	check_create_dir(std::move(p));

	for (const auto& frame : frames) {
		std::string file_path = path + "frame_" + std::to_string(counter) + ".png";
		cv::imwrite(file_path, frame);
		++counter;
	}
}

void Deflectometry::camera_calibration(int camera) {
	assert(m_screen && m_acquisition_worker && runtime_flags.get_camera_running_flag());
	//After setUpAcuqisition Datastream is available
	m_acquisition_worker->setUpAcquisition(camera);
	m_acquisition_worker->getDatastream(camera);

	// All image dispalying is running via the image handler class
	m_acquisition_worker->assignImageHandler(showRawImage);
	runtime_flags.set_next_fringe_pattern_flag_false(); //First set "next image flag" to false
	img_handler_thread = std::thread(&ImageHandler::run, &imgHandler, 2);

	// Lauch	
	std::thread controller(&Deflectometry::controller_userInput, this);
	std::thread camera_thread(&AcquisitionWorker::start, m_acquisition_worker.get());

	if (controller.joinable()) controller.join();
	if (camera_thread.joinable()) camera_thread.join();
	if (img_handler_thread.joinable()) img_handler_thread.join();	
	
	std::vector<cv::Mat> calibration_frames(std::move(m_acquisition_worker->m_frames)); //Moves the frames from acquisitionworker to local variable calibratoin_frames
	std::cout << "Number of calibration frames taken: " << calibration_frames.size() << '\n' <<
		" m_acuqisitionworker m_frames hopefully empty " << m_acquisition_worker -> m_frames.size() << std::endl;

	std::string settings_path("C:\\Users\\grein\\Desktop\\Master\\Project\\deflectometrie\\data\\in_VID5.xml");
	std::filesystem::path p(settings_path);
	if (std::filesystem::exists(p)) {
		runCameraCalibration(calibration_frames, true, p.string());
	}
}

void Deflectometry::controller_automatic() {
	//Check for user if Setup is correct
	std::cout << "Press any key and ENTER to start measurement if camera sees full fringe pattern\n";
	char u{ '\0' };
	std::cin.ignore(1000, '\n');
	while (!u) {
		std::cin.get(u);
		if (!std::cin) {
			std::cin.clear();
			std::cin.ignore(1000, '\n');
		}
	}
	runtime_flags.set_number_of_pictures_per_pattern(5);
	
	std::this_thread::sleep_for(std::chrono::seconds(2));
	std::cout << "Starting automatic meassurement now! \n";
	for (int i = 0; i < runtime_flags.get_number_of_shifts()*2; ++i) { //times two for vertikal and horizontal
		for (int j = 0; j < runtime_flags.get_number_of_pictures_per_pattern(); ++j) {
			//Savety first: save_processed finished to false;
			runtime_flags.set_false_save_process_finished();
			while (!runtime_flags.get_next_fringe_process_finished_flag()) {
				std::cout << "Wait for next fringe pattern \n";
				std::this_thread::sleep_for(std::chrono::milliseconds(200));
			}
			runtime_flags.set_true_imSave_flag();
			while (!runtime_flags.get_save_processed_finished_flag()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(200));
				std::cout << "Wait for save Process to finish \n";
			}
		}
		runtime_flags.next_fringe_process_finished_false();
		runtime_flags.set_next_fringe_pattern_flag_true();
	}
	std::cout << "Finished :D \n";
	if (runtime_flags.get_finished_fringe_Iteration()) {
		std::cout << "finished? \n";
		runtime_flags.set_false_acquisition_flag();
		runtime_flags.set_stop_fringe_projection_flag_true();
		imgHandler.stop();
	}
}


void Deflectometry::controller_userInput() {
	std::cout << "Press S to save image. \n" <<
		"Press P for next fringe pattern \n" <<
		"Press E to abort \n";
	int img_counter{ 1 };
	for (;;) {
		char c = std::cin.get();          // use int to detect EOF
		if (c == EOF) {
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			//m_controll_variable = command::stop;
			continue;
		}
		// skip whitespace (space, tab, enter, etc.)
		//if (std::isspace(static_cast<unsigned char>(c)))
		//	continue;

		switch (std::toupper(static_cast<unsigned char>(c))) {
		case 'P':
			runtime_flags.set_next_fringe_pattern_flag_true();
			std::cout << "Next fringe pattern gets projected \n";
			std::cin.ignore(1000, '\n');
			break;

		case 'S':
			runtime_flags.set_true_imSave_flag();
			std::cout << img_counter++ << "Image Saved!\n";
			std::cin.ignore(1000, '\n');
			break;
		case 'E':
			runtime_flags.set_false_acquisition_flag();
			runtime_flags.set_stop_fringe_projection_flag_true();
			imgHandler.stop();
			return;
		default:
			std::cout << "Not a valid input! (Use S, P or E )\n";
			break;
		}
	}
}


void Deflectometry::start_meassurement(Shift_mode shift_mode, DisplayMode disp_mode, int camera) {
	
	// m_screen & m_acquisition_worker must not be nullptr. Also camera running must be set. 
	assert(m_screen && m_acquisition_worker && runtime_flags.get_camera_running_flag());
	//After setUpAcuqisition Datastream is available
	m_acquisition_worker->setUpAcquisition(camera);
	m_acquisition_worker->getDatastream(camera);

	// All image dispalying is running via the image handler class
	m_acquisition_worker->assignImageHandler(showRawImage);
	//Shift mode is neede to generate the Pattern
	m_screen->generate_phaseShift(shift_mode);
	
	runtime_flags.set_next_fringe_pattern_flag_false(); //First set "next image flag" to false
	runtime_flags.set_stop_fringe_projection_flag_false(); //Set stop fringe projection flag to false
	
	// Run the Imagehandler class
	img_handler_thread = std::thread(&ImageHandler::run, &imgHandler, 2);
	
	// Lauch 
	std::thread controller;
	if (disp_mode == DisplayMode::UserInput) {
		controller = std::thread(&Deflectometry::controller_userInput, this);
	}
	if (disp_mode == DisplayMode::Automatic) {
		controller = std::thread(&Deflectometry::controller_automatic, this);// some function for automatic handling !!! 
	}

	std::thread fringe_pattern_thread(&Screen::displayPatterns_multi_thread, m_screen.get()); 
	
	std::thread camera_thread(&AcquisitionWorker::start, m_acquisition_worker.get());
	
	// First thread to finish, should be Camera_thread. 
	// runtime_flags.acquisition_flag -> first set to false
	if (camera_thread.joinable()) camera_thread.join();

	// Second thread to finish, should be fringe_pattern thread
	// runtime_flags().stop_fringe_projection set to true;
	if (fringe_pattern_thread.joinable()) fringe_pattern_thread.join();

	// Third is ImageHandler img_handler->stop();
	if (img_handler_thread.joinable()) img_handler_thread.join();

	//Controller thread .join() call should block 
	if (controller.joinable()) controller.join();
	
	std::cout << "Function *Start Meassurement* Exit \n";

	return;
	
}