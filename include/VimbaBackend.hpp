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


class VimbaBackend : public ICameraBackend
{
public:
	VimbaBackend(std::size_t index = 0);
	~VimbaBackend() override;

	// Connects and setup camera directly for acuqisition
	bool open(std::size_t i = 0) override;

	// Does just close the camera. Vimba is not shutdown
	void close(std::size_t i = 0) override;
	
	bool isRunning(std::size_t i) override;
	std::vector<defl::CameraConfig> getCameraConfig() override;
	cv::Mat grab(int timeout_ms) override;

private:
	std::mutex m_mut;
	
	std::vector<defl::CameraConfig> m_camera_data{};

	void logging(const std::size_t camera_index);

	std::atomic<bool> m_stopping{ false };
	std::atomic<int>  m_callbacksInFlight{ 0 };

	// if multiple camera are used this would not work
	bool m_running{ false };

	std::string m_active_cam{};

	std::vector<std::string> m_availableIds{};
	std::size_t m_cam_index{};

	std::vector<VmbCPP::FramePtr> m_frame_ptr{};

	RingBuffer* m_ringBufferPtr{ nullptr };

	// Opens the camera withing m_cameras with index i
	bool openCamera(const std::size_t i);
	void closeCamera(const std::size_t i);
	bool adjustSettings(const std::size_t);
	
	VmbCPP::CameraPtr findCameraByID(
		const std::size_t i);

	// Deltes RingBuffer if set and sets to nullptr
	void deleteBuffer();

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
		std::vector<std::string> keys) const;
	//bool setupBuffers(const std::size_t i);

	// Sets for every entry of Triggerselector the TrigerMode to Off
	// Sets for all ExosureModes the Autoexpsoure to OFF
	// At the end: 
	// TriggerSource is set to "Freerun" 
	bool forceFreerunTimedExposure(
		const VmbCPP::CameraPtr& cam);

	bool setGamma(
		const VmbCPP::CameraPtr& cam,
		double gamma
	);

	void checkPackagesize(
	const VmbCPP::CameraPtr& cam);

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
		std::size_t i,
		int n_buf);

	bool setGain(
		const VmbCPP::CameraPtr& cam,
		double Gain_dB);

	bool setExposureAbsRobust(const VmbCPP::CameraPtr& cam, double targetUs);

	bool runAcquisition(const VmbCPP::CameraPtr& cam);

};

// FrameObserver Class 
class FrameObserver : public VmbCPP::IFrameObserver
{
public:
	FrameObserver(VmbCPP::CameraPtr pCamera, 
		RingBuffer* buffer, 
		std::atomic<bool>* stopping,
		std::atomic<int>* callbacks
	)
		: VmbCPP::IFrameObserver(pCamera)
		, m_buffer{buffer}
		, m_stopping{stopping}
		, m_callbacksInFlight{callbacks}
	{}
	
	void FrameReceived(const VmbCPP::FramePtr pFrame) override {
		if (m_buffer == nullptr) {
			std::cerr << "No buffer Assigned in Frame Observer \n";
			throw std::runtime_error("No Buffer Assigned in Frame Observer");
		}
		// mark callback in flight
		m_callbacksInFlight->fetch_add(1, std::memory_order_acq_rel);

		// if stopping, don't touch camera, don't requeue
		if (m_stopping->load(std::memory_order_relaxed)) {
			m_callbacksInFlight->fetch_sub(1, std::memory_order_acq_rel);
			return;
		}

		//std::unique_lock<std::mutex> lock(m_tx);
		m_buffer->getFrame(pFrame);
		//lock.unlock();
		m_pCamera->QueueFrame(pFrame);
		m_callbacksInFlight->fetch_sub(1, std::memory_order_acq_rel);
	}

	~FrameObserver() override = default;
	
private:
	std::mutex m_tx;
	// pointer to static RingBufferObject
	RingBuffer* m_buffer = nullptr;
	std::atomic<bool>* m_stopping = nullptr;
	std::atomic<int>* m_callbacksInFlight = nullptr;
};


#endif