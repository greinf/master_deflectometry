#ifndef FLAGHANDLER_H
#define FLAGHANDLER_H

#include <atomic>
#include <stdexcept>

struct Flags {
public:
	Flags() {
		if (instance_counter) { throw std::runtime_error("Only one Instance of this class is allowed. \n"); }
		++instance_counter;
	}
	
	// ImageSaveFlags
	
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
	
	// Acquisition Flags
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

	// CameraRunningFlags
	
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

	// Next Fringe Pattern Flags.
	// Sets next fringe pattern flag true.
	void set_next_fringe_pattern_flag_true() {
		next_fringe_pattern.store(true);
	}

	// Thats next_fringe_pattern flag to false
	void set_next_fringe_pattern_flag_false() {
		next_fringe_pattern.store(false);
	}

	//Gets the fringe pattern flag. 
	//
	bool get_next_fringe_pattern_flag() {
		return next_fringe_pattern.load();
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

private:
	// ImageSaveFlags
	std::atomic<bool> image_save{ false };
	//CameraRunningFlags
	std::atomic<bool> camera_running_flag{ false };
	// Next Fringe Pattern Flags.
	std::atomic<bool> next_fringe_pattern{ false };
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