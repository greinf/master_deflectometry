#ifndef IMAGEHANDLER_H
#define IMAGEHANDLER_H

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <array>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include "flagHandler.hpp"

struct ImageHandler {
public:
	//Input ammount of parallel Windows
	static inline int i{ 0 };
	int window_counter{};
	ImageHandler() {
		if (i) { throw std::runtime_error("Only one instaces of this class is allowed. "); }
		++i;
	}
	
	// Method to start continues function readout and displaying of pictures.
	// First argument determines how many windows are opended. 
	void run(int i) {
		window_counter = i;
		std::lock_guard<std::mutex> function_lock(run_lock);
		m_running.store(true);
		switch (i) {
		case(2):
			cv::namedWindow("CameraFrame", cv::WINDOW_NORMAL);
			cv::setWindowProperty("CameraFrame", cv::WINDOW_NORMAL, cv::WINDOW_FREERATIO);
			if (!runtime_flags.disp.height && !runtime_flags.disp.width) {
				cv::namedWindow("FringePattern", cv::WINDOW_NORMAL);
				cv::setWindowProperty("FringePattern", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN); //For train debugging , cv::WINDOW_FREERATIO
			}
			else {
				/*
				cv::namedWindow("FringePattern", cv::WINDOW_NORMAL);
				cv::resizeWindow("FringePattern", runtime_flags.disp.width, runtime_flags.disp.height);
				cv::setWindowProperty("FringePattern", cv::WND_PROP_TOPMOST, 1); // optional: keep on top
				cv::moveWindow("FringePattern", runtime_flags.disp.posx, runtime_flags.disp.posy);
				*/
				cv::namedWindow("FringePattern", cv::WINDOW_NORMAL);
				cv::resizeWindow("FringePattern", runtime_flags.disp.width, runtime_flags.disp.height);
				cv::setWindowProperty("FringePattern", cv::WND_PROP_TOPMOST, 1); // optional: keep on top
				cv::moveWindow("FringePattern", runtime_flags.disp.posx, runtime_flags.disp.posy);

			}
			while (m_running.load()) {
				//std::cout << "Cv::waitkey called \n";
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				std::lock_guard<std::mutex> update_lock(update_lock);
				if (!m_imgCamera.empty()) cv::imshow("CameraFrame", m_imgCamera);
				if (!m_imgPattern.empty()) cv::imshow("FringePattern", m_imgPattern);
				cv::waitKey(1);
			}
			break;
		case(3):
			cv::namedWindow("CameraFrame", cv::WINDOW_NORMAL);
			cv::setWindowProperty("CameraFrame", cv::WINDOW_NORMAL, cv::WINDOW_FREERATIO);
			cv::namedWindow("FringePattern", cv::WINDOW_NORMAL);
			cv::setWindowProperty("FringePattern", cv::WINDOW_NORMAL, cv::WINDOW_FULLSCREEN);
			cv::namedWindow("ProcesseFrame", cv::WINDOW_NORMAL);
			cv::setWindowProperty("ProcessFrame", cv::WINDOW_NORMAL, cv::WINDOW_FREERATIO);
			while (m_running.load()) {
				std::lock_guard<std::mutex> update_lock(update_lock);
				if (!m_imgCamera.empty()) cv::imshow("CameraFrame", m_imgCamera);
				if (!m_imgPattern.empty()) cv::imshow("FringePattern", m_imgPattern);
				if (!m_imgProcessed.empty()) cv::imshow("ProcessFrame", m_imgProcessed);
				cv::waitKey(20);
			}
			break;
		default: std::runtime_error runtime_error("Not able to desplay more than 3 Windows at a time \n"); break;
		}
	}

	void stop() {
		m_running.store(false);
		clear();
	}

	void imshow_Camera(const cv::Mat& img) {
		//std::cout << "Update Camera \n";
		std::lock_guard<std::mutex> update_lock(update_lock);
		m_imgCamera = img.clone();
	}

	void imshow_Pattern(const cv::Mat& img) {
		//std::cout << "Fringe Update \n";
		std::lock_guard<std::mutex> update_lock(update_lock);
		m_imgPattern = img.clone();
	}

	void imshow_Processed(const cv::Mat& img) {
		std::lock_guard<std::mutex> update_lock(update_lock);
		m_imgProcessed = img.clone();
	}

	~ImageHandler() {
		--i;
		std::cout << "Image Handler Object Destroyed! \n";
	}

	void clear() {
		m_imgCamera.release();
		m_imgPattern.release();
		m_imgProcessed.release();
	}

private:
	cv::Mat m_imgCamera, m_imgPattern, m_imgProcessed;
	std::atomic<bool> m_running{ false };
	std::mutex update_lock;
	std::mutex run_lock;
};


inline ImageHandler imgHandler;

#endif 