#ifndef VIMBABACKEND_HPP
#define VIMBABACKEND_HPP

#include <memory>
#include <iostream>
#include <cstddef>
#include <vector>
#include "config/CameraConfig.hpp"
#include "ICameraBackend.hpp"
#include <opencv2/opencv.hpp>
#include <VmbCPP/VmbCPP.h>
#include <string>
#include <thread>
#include <mutex>
#include "RingBuffer.hpp"
#include <atomic>
#include <deque>


class VimbaBackend : public ICameraBackend
{
public:
	VimbaBackend();
	~VimbaBackend() override;

	// Open() Arg1 gives the ammound of camera to be connected !
	bool open(std::size_t i = 0) override;

	// This just closes all available Cameras
	void close(std::size_t i = 0) override;
	
	bool isRunning(std::size_t i) override;

	std::vector<defl::CameraConfig> getCameraConfig() override;

	// Return from all the active newest frame 
	std::vector<cv::Mat> grab(int timeout_ms) override;

private:
	std::mutex m_mut;
	
	std::vector<defl::CameraConfig> m_camera_data{};

	void logging(const std::size_t camera_index);

	void logging();

	std::deque<std::atomic<bool>> m_stopping;

	std::vector<bool> m_running{};

	std::string m_active_cam{};

	std::vector<std::string> m_availableIds{};

	// m_frame_ptr[
	// [frame[0] camera 1, frame[1] camera 1 ....],
	// [frame[0] camera 2, frame[1] camera 2 ....],
	// ]
	// VmbCPP::FramePtr -> 
	std::vector<std::vector<VmbCPP::FramePtr>> m_frame_ptr{};

	std::vector<std::shared_ptr<RingBuffer>> m_ringBufferPtr{};

	// Opens the camera withing m_cameras with index i
	bool openCamera(const std::size_t i);
	void closeCamera(const std::size_t i);
	bool adjustSettings();
	
	// Can be used if index in m_availableIDs is known
	VmbCPP::CameraPtr findCameraByID(
		const std::size_t i);

	// Should be Used to call get the cameraPtr for the according cam by extendedID
	VmbCPP::CameraPtr findCameraByID(
		const std::string& extendedID);

	static bool getFeature(const VmbCPP::CameraPtr& cam,
		const std::vector<std::string>& names,
		VmbCPP::FeaturePtr& out);
	
	static bool setEnum(const VmbCPP::CameraPtr& cam,
		const std::vector<std::string>& names,
		const char* value);

	static bool setFloatClamped(const VmbCPP::CameraPtr& cam,
		const std::vector<std::string>& names,
		double value);

	static bool setIntClamped(const VmbCPP::CameraPtr& cam,
		const std::vector<std::string>& names,
		VmbInt64_t value);

	static bool setBool(const VmbCPP::CameraPtr& cam,
		const std::vector<std::string>& names,
		bool value);

	// Arg1 shardPtr to a Camera Arg2 vector of string
	// searches all availables features if the contain the key and puts them out
	void findFeatures(
		VmbCPP::CameraPtr& cam,
		std::vector<std::string> keys
	) const;
	//bool setupBuffers(const std::size_t i);

	// Sets for every entry of Triggerselector the TrigerMode to Off
	// Sets for all ExosureModes the Autoexpsoure to OFF
	// At the end: 
	// TriggerSource is set to "Freerun" 
	bool forceFreerunTimedExposure(
		const VmbCPP::CameraPtr& cam
	);

	bool setGamma(
		const VmbCPP::CameraPtr& cam,
		double gamma
	);

	bool checkPackagesize(
		const VmbCPP::CameraPtr& cam,
		const VmbInt64_t package_sz
	);

	// If frameRate set to zero the minimal allowed Frame Rate is set 
	bool setFrameRate(
		const VmbCPP::CameraPtr& cam,
		const double frameRate
	);

	bool setRoi(
		const VmbCPP::CameraPtr& cam,
		int x_0 = 0,
		int y_0 = 0,
		int height = 2056,
		int width = 2464
	);

	bool setFormat(
		const VmbCPP::CameraPtr& cam
	);

	// Allocates the internal Buffer for Vimba. VmbCPP::FramePtr are connected with the FrameObserver.
	// If Frame Received FrameObserver (that has internal Buffer) moves the data into the RingBuffer.
	// RingBuffer is stores alway the 10 newest Picutres by the FrameObserver
	bool allocateBuffer(
		int n_buf
	);

	bool setGain(
		const VmbCPP::CameraPtr& cam,
		double Gain_dB
	);

	bool setExposureAbsRobust(
		const VmbCPP::CameraPtr& cam,
		double targetUs
	);

	bool runAcquisition();

	bool clearFailedExtraction()
	{
		if (!std::cin)
		{
			if (std::cin.eof())
			{
				std::exit(0);
			}

			std::cin.clear();
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

			return true;
		}

		return false;
	}
};

// FrameObserver Class 
class FrameObserver : public VmbCPP::IFrameObserver
{
public:

	FrameObserver(VmbCPP::CameraPtr pCamera, 
		std::shared_ptr<RingBuffer> buffer, 
		std::atomic<bool>* stopping
	)
		: VmbCPP::IFrameObserver(pCamera)
		, m_buffer{std::move(buffer)}
		, m_stopping{stopping}
	{
		assert(m_stopping != nullptr);
		assert(m_buffer != nullptr);
	}


	void FrameReceived(const VmbCPP::FramePtr pFrame) override {
		// debugging 
		if (m_stopping->load(std::memory_order_relaxed)) {
			std::cout << "Stopped and reque Frame \n";
			//m_pCamera->QueueFrame(pFrame);
			return;
		}

		VmbFrameStatusType status;

		auto err = pFrame->GetReceiveStatus(status);
		if (err != VmbErrorSuccess || status != VmbFrameStatusComplete) {
			static std::atomic<int> cnt{ 0 };
			if ((cnt.fetch_add(1) % 5) == 0) { // rate limit
				std::string cam_name;
				m_pCamera->GetID(cam_name);
				std::cout << "[Cam " << cam_name << "] non-complete status=" << status << " err=" << err << "\n";
			}
			m_pCamera->QueueFrame(pFrame);
			return;
		}

		const std::size_t index =
			m_buffer->m_counter.fetch_add(1, std::memory_order_relaxed) % m_buffer->m_size;
		
		if (!m_buffer->extractData(pFrame, index)) {
			std::cout << "crashed ? \n";
			m_pCamera->QueueFrame(pFrame);
			return;
		}

		m_buffer->m_counter.store(m_buffer->m_counter.load(std::memory_order_acquire)
			% m_buffer->m_size, std::memory_order_release);

		m_pCamera->QueueFrame(pFrame);

		m_buffer->start();
	}

	~FrameObserver() override = default;
	
private:
	std::mutex m_tx;
	// pointer to static RingBufferObject
	std::shared_ptr<RingBuffer> m_buffer;
	std::atomic<bool>* m_stopping = nullptr;
};


#endif