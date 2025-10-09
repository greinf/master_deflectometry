#ifndef CAMERA_H
#define CAMERA_H

#include <cstddef>
#include <iostream>
#include <memory>
#include <peak/peak.hpp>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>

class AcquisitionWorker {
public:
	explicit AcquisitionWorker(std::shared_ptr<peak::core::DataStream> ds, std::shared_ptr<peak::core::NodeMap> nm):
		m_datastream{ ds }, m_nodemapRemoteDevice{ nm }, m_running {false}
	{
		if  (!m_datastream && !m_nodemapRemoteDevice){
			throw (std::exception("Acuqisition Worker has empty Datastream or RemoteDevice Object \n"));
		}
	}

	~AcquisitionWorker() { stop(); }

	void start() {
		if (m_running) {
			std::cout << "Controll Varialbe m_running is already set to true. Acuisition is already running!" << std::endl;
			return; //if already true no other running is allowed
		}
		m_running = true;
		readout = std::thread([this] { run(); });
	}
	void stop(){
		m_running = false;
		try { m_datastream->StopAcquisition(); } catch(...){}
		if (readout.joinable()) readout.join();
	}

	void assignFunct_ptr(void(*funct_ptr)(const cv::Mat&)) {
		m_funct_ptr = funct_ptr;
	}

private:
	// Try to work with a function pointer this time for callback.
	// While the m_funct_ptr is a member, the adresse it not owned by the class!
	void(*m_funct_ptr)(const cv::Mat&) = nullptr;

	std::shared_ptr<peak::core::DataStream> m_datastream;
	std::shared_ptr<peak::core::NodeMap> m_nodemapRemoteDevice;
	std::atomic<bool> m_running;
	std::thread readout;
	void run() {
		while (m_running) {
			try {
				const auto buffer = m_datastream->WaitForFinishedBuffer(1000);
				if (buffer) {
					// Be carefull!!! view does NOT own the data. It is read directly from the buffer through 
					// the void pointer, pointing to the first element of the Buffer. 
					// Even more dangerous -> Void pointer used. At this point openCV does not know 
					// if the image is Mono8 Mono10 or Mono12. 
					
					cv::Mat view(buffer->Height(), buffer->Width(), CV_8UC1, (void*)buffer->BasePtr(), buffer->Width());
					//Callback called
					m_funct_ptr(view);
					m_datastream->QueueBuffer(buffer);
				}
			}
			catch (...) {}
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