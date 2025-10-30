#ifndef FLAGHANDLER_H
#define FLAGHANDLER_H

#include <atomic>
#include <stdexcept>
#include <iostream>
//If not defined, getting problms witch std::max/ std::min
#define NOMINMAX
#include <windows.h>
#include <mutex>

struct GrayValueCalib {
public:
	int stepwidth{};
	int pictures_per_value{};

	static GrayValueCalib& instance() {
		static GrayValueCalib gray_val_data;
		return gray_val_data;
	}
	
	// Small explenation why return by reference: If we call GrayValueCalib a= GrayValueCalib b-> a already exists. Both objects 
	// are passed to the copy constructor. Therefor return by reference to avoid creating a copy of the object. Since Object a, already exists. 
	GrayValueCalib& operator=(const GrayValueCalib&) = delete; 
	GrayValueCalib& operator=(GrayValueCalib&&) = delete;
	GrayValueCalib(const GrayValueCalib&) = delete;
	GrayValueCalib(GrayValueCalib&&) = delete;
private:
	GrayValueCalib() = default;
};


struct DisplayInformation {
public:
	int posx{}, posy{};
	int width{}, height{};

	float width_mm{ 527.04f }, height_mm{ 296.46f }, diagonal_mm{ 605.0f };
	float pixelptich_mm{ 0.2745f };

	float wavelength{}; //is given in number per 2pi


	// Delete copy and move to prevent duplicates
	DisplayInformation(const DisplayInformation&) = delete;
	DisplayInformation& operator=(const DisplayInformation&) = delete;
	DisplayInformation(DisplayInformation&&) = delete;
	DisplayInformation& operator=(DisplayInformation&&) = delete;

	// Static Member functions that holds static instance of the struct object. 
	// The instance is returned by reference which leads to only one instance created in the file.
	// The default constructor is private therefore can not be called from outside the class, only be the static member function. 
	static DisplayInformation& instance() {
		static DisplayInformation single_instance; // created once, on first call
		single_instance.getDispalyInformation();
		return single_instance;
	}



private:
	DisplayInformation() = default;
	void getDispalyInformation() {
		DISPLAY_DEVICE dd;
		dd.cb = sizeof(dd);

		if (!EnumDisplayDevices(nullptr, 1, &dd, 0)) {
			std::cerr << "No second display found.\n";
			return;
		}

		DEVMODE dm;
		dm.dmSize = sizeof(dm);
		if (!EnumDisplaySettings(dd.DeviceName, ENUM_CURRENT_SETTINGS, &dm)) {
			std::cerr << "Could not get display settings.\n";
			return;
		}
		// Be carefull second window position is just hardcoded!!
		posx = 1920; //dm.dmPosition.x;
		posy = dm.dmPosition.y; //normally 0
		width = dm.dmPelsWidth;
		height = dm.dmPelsHeight;
	}
};


struct Flags {
public:
	Flags() {
		if (instance_counter) { throw std::runtime_error("Only one Instance of this class is allowed. \n"); }
		++instance_counter;
	}
	
	// ******* ImageSaveFlags *************
	
	// Sets the imSave flag to true.
	void set_true_imSave_flag() {
		image_save.store(true);
	}

	void set_false_imSave_flag() {
		image_save.store(false);
	}
	// Gets the imSave flag.
	bool get_imSave_flag() {
		return image_save.load();
	}
	//Is true if saving is finished 
	void set_true_save_process_finished() {
		save_process_finished.store(true);
	}

	void set_false_save_process_finished() {
		save_process_finished.store(false);
	}

	bool get_save_processed_finished_flag() {
		return save_process_finished.load();
	}

	// **************  Acquisition Flags ***********
	// Sets the Acuqisition flag to true
	void set_true_acquisition_flag() {
		acquisition_flag.store(true);
	}

	void set_false_acquisition_flag() {
		acquisition_flag.store(false);
	}

	bool get_acquisition_flag() {
		return acquisition_flag.load();
	}

	// *************** CameraRunningFlags ************
	
	// Sets camera running flat to false
	// Datastream of camera gets closed. 
	void set_stop_camera_running_flag() {
		camera_running_flag.store(false);
	}
	// Sets camera running flag true.
	void set_start_camera_running_flag() {
		camera_running_flag.store(true);
	}
	//Gets current camera running flag. 
	bool get_camera_running_flag() {
		return camera_running_flag.load();
	}

	// *********Next Fringe Pattern Flags. ***************
	// Sets next fringe pattern flag true.
	void set_next_fringe_pattern_flag_true() {
		next_fringe_pattern.store(true);
	}

	// Thats next_fringe_pattern flag to false
	void set_next_fringe_pattern_flag_false() {
		next_fringe_pattern.store(false);
	}
	//Gets the fringe pattern flag. 
	bool get_next_fringe_pattern_flag() {
		return next_fringe_pattern.load();
	}
	//Processed next fringe finished flags.
	void next_fringe_process_finished_true() {
		next_pattern_finished.store(true);
	}

	void next_fringe_process_finished_false() {
		next_pattern_finished.store(false);
	}

	bool get_next_fringe_process_finished_flag() {
		return next_pattern_finished.load();
	}
	
	
	// Stop Fringe Projection 
	// Sets the Stop fringe Projection flag true. 
	// Fringe projection stops. 
	void set_stop_fringe_projection_flag_true() {
		stop_fringe_projection.store(true);
	}

	void set_stop_fringe_projection_flag_false() {
		stop_fringe_projection.store(false);
	}

	bool get_stop_fringe_projection_flag() {
		return stop_fringe_projection.load();
	}

	int get_number_of_shifts() {
		return n_shifts;
	}

	void set_number_of_shifts(int n) {
		n_shifts = n;
	}

	void set_finished_fringe_Iteration_true() {
		finished_fringe_Iteration.store(true);
	}

	void set_finished_fringe_Iteration_false() {
		finished_fringe_Iteration.store(false);
	}

	bool get_finished_fringe_Iteration() {
		return finished_fringe_Iteration.load();
	}

	void set_number_of_pictures_per_pattern(int n){
		n_pictures_per_pattern = n;
	}

	int get_number_of_pictures_per_pattern() {
		return n_pictures_per_pattern;
	}

	~Flags() {
		--instance_counter; std::cout << "Flag Handler Object destroyed. \n";
	}
	// CameraPixel
	int pixel_x;
	int pixel_y;
	// DispalyInformation 
	DisplayInformation& disp = DisplayInformation::instance();
	GrayValueCalib& calib = GrayValueCalib::instance();

	std::mutex save_mutex;
	std::mutex pattern_mutex;
	std::condition_variable cv;


private:
	// ImageSaveFlags
	std::atomic<bool> image_save{ false };
	std::atomic<bool> save_process_finished{ false };

	//CameraRunningFlags
	std::atomic<bool> camera_running_flag{ false };
	// Next Fringe Pattern Flags.
	std::atomic<bool> next_fringe_pattern{ false };
	std::atomic<bool> next_pattern_finished{ false };

	// Stop Fringe Projection
	std::atomic<bool> stop_fringe_projection{ false };
	// Acuqisition flag
	std::atomic<bool> acquisition_flag{ false };
	// Fringe Pattern Iterator reached end
	std::atomic<bool> finished_fringe_Iteration{ false };

	int n_shifts{};

	int n_pictures_per_pattern{};

	static inline int instance_counter{ 0 };
};

inline Flags runtime_flags{};

#endif