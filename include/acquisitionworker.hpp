#ifndef  ACQUISITIONWORKER_H
#define ACQUISITIONWORKER_H
#include "camera.hpp"

class AcquisitionWorker: public Camera{
public:
	explicit AcquisitionWorker(int i) : //try to acess the m_datastream and m_nodemapRemoteDevicec from m_camera
		// integer let choose from different connected cameras. 
		Camera(), //setup Camera Base Class
		camera_n{ i },
		n_pictures {5}
	{
		m_nodemapRemoteDevice_A = m_nodemapRemoteDevice.at(camera_n);
		m_controll_variable.store(command::stop);
		if (!m_nodemapRemoteDevice_A) {
			throw (std::runtime_error("RemoteDevice Object is Nullptr \n"));
		}
		
	}

	~AcquisitionWorker() { stop(); }

	enum class command {
		stop,
		save,
		acquire,
		max_value
	};

	void start(AcquisitionMode mode) {
		if (mode == AcquisitionMode::max_value) {
			std::cout << "No Acuqisition is set \n";
			return;
		}
		if (m_controll_variable.load() != command::stop) { //m_running
			std::cout << "Controll Varialbe m_running is already set to true. Acuisition is already running!" << std::endl;
			return; //if already true no other running is allowed
		}
		if (!m_image_handler) { //&& m_acquire_command_handler
			std::runtime_error e("Image_Handler is Nullpointer. "); //or Command_Handler are
		}
		if (mode == AcquisitionMode::UserInput) {
			try {
				m_controll_variable.store(command::acquire);
				//if assigned start both threads
				m_readout_thread = std::thread([this] { run_readout(); });
				std::thread m_process_thread = std::thread([this] { userInput(); });
				//userInput();
				if (m_readout_thread.joinable()) m_readout_thread.join();
				if (m_process_thread.joinable()) m_process_thread.join();
			}
			catch (std::exception& e) { std::cout << "EXCEPTION " << e.what(); }
		}
		if (mode == AcquisitionMode::Automatic) {
			try {
				m_controll_variable.store(command::acquire);
				m_readout_thread = std::thread([this] { run_readout(); });
				std::this_thread::sleep_for(std::chrono::milliseconds(200));
				for (int i = 0; i < n_pictures; ++i) {
					std::this_thread::sleep_for(std::chrono::milliseconds(100));
					m_controll_variable.store(command::save);
				}
				m_controll_variable.store(command::stop);
									
				if (m_readout_thread.joinable())m_readout_thread.join();
			}
			catch (std::exception& e) { std::cout << "EXCEPTION " << e.what(); }
			
		}
	}

	void stop() {
		m_controll_variable.store(command::stop);
		try { m_datastream_A->StopAcquisition(); }
		catch (...) {}
		if (m_readout_thread.joinable()) m_readout_thread.join();

		// avoid self-join deadlock: only join process thread from a different thread
		if (m_process_thread.joinable()
			&& std::this_thread::get_id() != m_process_thread.get_id()) {
			m_process_thread.join();
		}
		resetHandlers();
	}

	void assignImageHandler(void(*funct_ptr)(const cv::Mat&)) {
		m_image_handler = funct_ptr;
	}

	std::vector<cv::Mat> m_frames;


	void image_save() {
		m_controll_variable.store(command::save);
	}

	void getDatastream(int i) {
		m_datastream_A = m_dataStream.at(i);
	}

private:
	// Try to work with a function pointer this time for callback.
	// While the m_funct_ptr is a member, the adresse it not owned by the class!
	void(*m_image_handler)(const cv::Mat&) = nullptr;
	int camera_n{};
	int n_pictures{5};

	std::shared_ptr<peak::core::DataStream> m_datastream_A;
	std::shared_ptr<peak::core::NodeMap> m_nodemapRemoteDevice_A;
	std::atomic<AcquisitionWorker::command> m_controll_variable{ command::stop };
	std::thread m_readout_thread;
	std::thread m_process_thread;


	void run_readout() {
		while (m_controll_variable.load() != command::stop) {
			try {
				//std::cout << "Visualize ";
				const auto buffer = m_datastream_A->WaitForFinishedBuffer(5000); //could use peak::core::Timeout::INFINITE_TIMEOUT
				
				if (buffer) {
					// Be carefull!!! view does NOT own the data. It is read directly from the buffer through 
					// the void pointer, pointing to the first element of the Buffer. 
					// Even more dangerous -> Void pointer used. At this point openCV does not know 
					// if the image is Mono8 Mono10 or Mono12. 

					cv::Mat view(buffer->Height(), buffer->Width(), CV_8UC1, (void*)buffer->BasePtr(), buffer->Width());
					std::cout << "How often? \n";
					//Callback called
					m_image_handler(view);
					if (m_controll_variable.load() == command::save) {
						m_frames.push_back(view.clone());
						m_controll_variable.store(command::acquire, std::memory_order_release);
						
					}
					if (m_controll_variable.load() == command::stop) return;
					m_datastream_A->QueueBuffer(buffer);
				}
			}
			catch (std::exception& e) { std::cout << "EXCEPTION " << e.what() << std::endl; }
		}
	}

	void resetHandlers() {
		m_image_handler = nullptr;
	}

	void userInput() {
		std::cout << "Press S to save image. \nPress Q to stop. \n";
		for (;;) {
			int ch = std::cin.get();          // use int to detect EOF
			if (ch == EOF) {
				std::this_thread::sleep_for(std::chrono::milliseconds(500));
				//m_controll_variable = command::stop;
				continue;
			}

			char c = static_cast<char>(ch);

			// skip whitespace (space, tab, enter, etc.)
			if (std::isspace(static_cast<unsigned char>(c)))
				continue;

			switch (std::toupper(static_cast<unsigned char>(c))) {
			case 'Q':
				std::cout << "Image Acquisition stopped!\n";
				m_controll_variable = command::stop;

				return; // exit function

			case 'S':
				std::cout << "Image Saved!\n";
				m_controll_variable = command::save;
				std::cin.ignore(1000, '\n');
				// keep listening for more commands
				break;

			default:
				std::cout << "Not a valid input! (Use S or Q)\n";
				break;
			}
		}
	}
};

#endif //  ACUISITIONWORKER_H
