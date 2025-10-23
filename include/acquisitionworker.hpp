#ifndef ACQUISITIONWORKER_H
#define ACQUISITIONWORKER_H
#include "camera.hpp"
#include "flagHandler.hpp"
#include "imageHandler.hpp"

#include <mutex>
inline cv::Mat rotImage180(const cv::Mat& mat);


struct calibrationData {
	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
};

inline calibrationData getfromFile(std::string& path) {
	cv::FileStorage fs(path, cv::FileStorage::READ);
	calibrationData data;
	fs["distortion_coefficients"] >> data.distCoeffs;
	fs["camera_matrix"] >> data.cameraMatrix;
	return data;
}




class AcquisitionWorker: public Camera{
public:
	explicit AcquisitionWorker(int i) : //try to acess the m_datastream and m_nodemapRemoteDevicec from m_camera
		// integer let choose from different connected cameras. 
		Camera(), //setup Camera Base Class
		camera_n{ i }
	{
		if(runtime_flags.get_camera_running_flag())
			m_nodemapRemoteDevice_A = m_nodemapRemoteDevice.at(camera_n);
		
		//set acquisitionflag to falls
		runtime_flags.set_false_acquisition_flag();
		if (!m_nodemapRemoteDevice_A) {
			throw (std::runtime_error("RemoteDevice Object is Nullptr \n"));
		}
	}

	~AcquisitionWorker() {
		std::cout << "Acuisitionworker Obj Destroyed \n"; 
		close();
	}

	enum class command {
		stop,
		save,
		acquire,
		max_value
	};

	void start() {
		//one instaces of the AcquisitionWorker class is allowed to have only one start() function running
		std::lock_guard<std::mutex> acquisition_block(m_acquisition_block);
		
		if (!m_image_handler) { //&& m_acquire_command_handler
			std::runtime_error e("Image_Handler is Nullpointer. \n"); //or Command_Handler are
		}
		try {
			runtime_flags.set_true_acquisition_flag();
			m_readout_thread = std::thread([this] { run_readout(); });
			if (m_readout_thread.joinable()) m_readout_thread.join();
		}
		catch (std::exception& e) { std::cout << "EXCEPTION " << e.what(); }
	}

	
	void assignImageHandler(void(*funct_ptr)(const cv::Mat&)) {
		m_image_handler = funct_ptr;
	}

	// Readout Frames are stored here. 
	std::vector<cv::Mat> m_frames;

	
	void image_save() {
		runtime_flags.set_true_imSave_flag();
	}

	void getDatastream(int i) {
		m_datastream_A = m_dataStream.at(i);
	}

	
private:
	// Try to work with a function pointer this time for callback.
	// While the m_funct_ptr is a member, the adresse it not owned by the class!
	std::mutex m_acquisition_block;
	void(*m_image_handler)(const cv::Mat&) = nullptr;
	int camera_n{};

	std::shared_ptr<peak::core::DataStream> m_datastream_A;
	std::shared_ptr<peak::core::NodeMap> m_nodemapRemoteDevice_A;
	//std::atomic<AcquisitionWorker::command> m_controll_variable{ command::stop };
	std::thread m_readout_thread;
	std::thread m_controller_thread;


	// Used by ~AcquisitioWorker() to close according Datastream()
	void close() {
		runtime_flags.set_false_acquisition_flag();
		try { m_datastream_A->StopAcquisition(); }
		catch (...) {}
		if (m_readout_thread.joinable()) m_readout_thread.join();

		// avoid self-join deadlock: only join process thread from a different thread
		if (m_controller_thread.joinable()
			&& std::this_thread::get_id() != m_controller_thread.get_id()) {
			m_controller_thread.join();
		}
		//resetHandlers();
	}


	void run_readout() {
		while (runtime_flags.get_acquisition_flag()) {
			try {
				//std::cout << "Visualize ";
				const auto buffer = m_datastream_A->WaitForFinishedBuffer(5000); //could use peak::core::Timeout::INFINITE_TIMEOUT
				
				if (buffer) {
					// Be carefull!!! view does NOT own the data. It is read directly from the buffer through 
					// the void pointer, pointing to the first element of the Buffer. 
					// Even more dangerous -> Void pointer used. At this point openCV does not know 
					// if the image is Mono8 Mono10 or Mono12. 
					cv::Mat view(buffer->Height(), buffer->Width(), CV_8UC1, (void*)buffer->BasePtr(), buffer->Width());
					//Callback called
					cv::cvtColor(view, view, cv::COLOR_BayerRG2GRAY); //Camera is BayerBG
					cv::Mat rotated = rotImage180(view);
					m_image_handler(rotated);
					if (runtime_flags.get_imSave_flag()) {
						std::this_thread::sleep_for(std::chrono::milliseconds(200));
						m_frames.push_back(rotated.clone());
						runtime_flags.set_false_imSave_flag();
						runtime_flags.set_true_save_process_finished();
					}
					//if (runtime_flags.get_acquisition_flag()) return;
					m_datastream_A->QueueBuffer(buffer);
					
				}
			}
			catch (std::exception& e) { std::cout << "EXCEPTION " << e.what() << std::endl; }
		}
	}

	void resetHandlers() {
		std::cout << "Image Handler (AcquisitionWorker) is set to null \n";
		m_image_handler = nullptr;
	}
};


cv::Mat rotImage180(const cv::Mat& mat) {
	int width = mat.cols; //no function just public member variable
	int height = mat.rows;
	cv::Vec<float, 2> center(float(width / 2.), float(height / 2.));
	cv::Mat rot_Matrix = cv::getRotationMatrix2D(center, 180., 1.);
	cv::Mat destination(height, width, CV_8UC1);
	cv::warpAffine(mat, destination, rot_Matrix, destination.size());
	return destination;
}



#endif //  ACUISITIONWORKER_H
