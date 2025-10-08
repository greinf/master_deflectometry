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
	explicit AcquisitionWorker(std::shared_ptr<peak::core::DataStream> ds):
	m_datastream{ds}, m_running{ false }
	{
		try {
			m_nodemapRemoteDevice = m_datastream->ParentDevice()->RemoteDevice()->NodeMaps().at(0);
		}
		catch (std::exception& e) {
			std::cout << "EXCEPTION " << e.what() << std::endl;
		}
	}

	~AcquisitionWorker() { stop(); }

	void start() {
		if (m_running) return;
		m_running = true;
		readout = std::thread([this] { run(); });
	}
	void stop(){
		m_running = false;
		try { m_datastream->StopAcquisition(); } catch(...){}
		if (readout.joinable()) readout.join();
	}
private:
	std::shared_ptr<peak::core::DataStream> m_datastream;
	std::shared_ptr<peak::core::NodeMap> m_nodemapRemoteDevice;
	std::atomic<bool> m_running;
	std::thread readout;
	void run() {
		while (m_running) {
			try {
				const auto buffer = m_datastream->WaitForFinishedBuffer(1000);
				if (buffer) {
					// Option A (no IPL, Mono8 forced):
					cv::Mat view(buffer->Height(), buffer->Width(), CV_8UC1, (void*)buffer->BasePtr(), buffer->Width());
					cv::imshow("Stream", view);
					cv::waitKey(5);
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