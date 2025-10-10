#ifndef CAMERA_H
#define CAMERA_H

#include <cstddef>
#include <iostream>
#include <memory>
#include <peak/peak.hpp>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <string>
#include <limits>
#include <cctype>
#include <array>

class AcquisitionWorker {
public:
	explicit AcquisitionWorker(std::shared_ptr<peak::core::DataStream> ds, std::shared_ptr<peak::core::NodeMap> nm):
		m_datastream{ ds }, m_nodemapRemoteDevice{ nm }
	{
		m_controll_variable.store(command::stop);
		if (!m_datastream || !m_nodemapRemoteDevice) {
			throw (std::runtime_error("Acuqisition Worker has empty Datastream or RemoteDevice Object \n"));
		}
	}
	 
	~AcquisitionWorker() { stop(); }

	enum class command {
		stop,
		save,
		acquire,
		max_value
	};

	std::array<std::string, static_cast<size_t>(command::max_value)> command_string{ "Stop", "Save", "Continue"};


	void start() {
		if (m_controll_variable.load() != command::stop) { //m_running
			std::cout << "Controll Varialbe m_running is already set to true. Acuisition is already running!" << std::endl;
			return; //if already true no other running is allowed
		}
		if (!m_image_handler ) { //&& m_acquire_command_handler
			std::runtime_error e ("Image_Handler is Nullpointer. "); //or Command_Handler are
		}
		
		try {
			m_controll_variable.store(command::acquire);
			//if assigned start both threads
			m_readout_thread = std::thread([this] { run_readout(); });
			//m_process_thread = std::thread([this] { userInput(); });
			userInput();
		}
		catch (std::exception& e) { std::cout << "EXCEPTION " << e.what(); }
	}
	void stop(){
		m_controll_variable.store(command::stop);
		try { m_datastream->StopAcquisition(); } catch(...){}
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
	


private:
	// Try to work with a function pointer this time for callback.
	// While the m_funct_ptr is a member, the adresse it not owned by the class!
	void(*m_image_handler)(const cv::Mat&) = nullptr;

	std::shared_ptr<peak::core::DataStream> m_datastream;
	std::shared_ptr<peak::core::NodeMap> m_nodemapRemoteDevice;
	std::atomic<AcquisitionWorker::command> m_controll_variable{ command::stop };
	std::thread m_readout_thread;
	std::thread m_process_thread;
	std::vector<cv::Mat> m_frames;
	
	void run_readout() {
		while (m_controll_variable.load() != command::stop) {
			try {
				//std::cout << "Visualize ";
				const auto buffer = m_datastream->WaitForFinishedBuffer(1000); //could use peak::core::Timeout::INFINITE_TIMEOUT
				if (buffer) {
					// Be carefull!!! view does NOT own the data. It is read directly from the buffer through 
					// the void pointer, pointing to the first element of the Buffer. 
					// Even more dangerous -> Void pointer used. At this point openCV does not know 
					// if the image is Mono8 Mono10 or Mono12. 
					
					cv::Mat view(buffer->Height(), buffer->Width(), CV_8UC1, (void*)buffer->BasePtr(), buffer->Width());
					//Callback called
					m_image_handler(view);
					if (m_controll_variable.load() == command::save) {
						m_frames.push_back(view.clone());
						m_controll_variable.store(command::acquire, std::memory_order_release);
					}
					m_datastream->QueueBuffer(buffer);
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
				std::this_thread::sleep_for(std::chrono::milliseconds(200));
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
				// keep listening for more commands
				break;

			default:
				std::cout << "Not a valid input! (Use S or Q)\n";
				break;
			}
		}
	}
};



class Camera
{
public:
	Camera();
	Camera(const Camera&) = delete;
	Camera& operator=(const Camera&) = delete;
	Camera(Camera&&) = default;
	Camera& operator=(Camera&&) = default;
	~Camera();
	
	void getFrames(std::size_t i = 0);

private:
	std::vector<std::shared_ptr<peak::core::Device>> m_device; //normally only one camera is used
	std::vector<std::shared_ptr<peak::core::DataStream>> m_dataStream;
	std::vector<std::shared_ptr<peak::core::NodeMap>> m_nodemapRemoteDevice;
	bool PrepareAcuqisition(std::size_t i=0);
	bool SetRoi(std::int64_t x, std::int64_t y, std::int64_t width, std::int64_t height, std::size_t i=0);
	bool AllocAndAnnounceBuffers(std::size_t i = 0 );
	bool StartAcquisition(std::size_t i = 0);

};

#endif