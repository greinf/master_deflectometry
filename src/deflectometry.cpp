#pragma once
#include "deflectometry.hpp"
#include "screen.hpp"
#include "acquisitionworker.hpp"


void showRawImage(const cv::Mat& mat) {
	cv::imshow("Frame", mat);
	cv::waitKey(20);
}

Deflectometry::Deflectometry() {
	m_screen = std::make_shared<Screen>();
	m_acquisition_worker = std::make_shared<AcquisitionWorker>(0);
}

void Deflectometry::start_meassurement(Shift_mode shift_mode, DisplayMode disp_mode, int camera) {
	//Both pointers must be set. 
	assert(m_screen && m_acquisition_worker);
	//After setUpAcuqisition Datastream is available
	m_acquisition_worker->setUpAcquisition(camera);
	m_acquisition_worker->getDatastream(camera);
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	m_acquisition_worker->assignImageHandler(showRawImage);

	m_screen->generate_phaseShift(shift_mode);
	
	if (disp_mode == DisplayMode::UserInput) {
		AcquisitionMode acqui_mode = AcquisitionMode::UserInput;
		assert(acqui_mode != AcquisitionMode::Automatic && "Manuall user Input and Automatic Acuqisation does not work \n");
		std::thread t1(&Screen::displayPatterns, m_screen.get()); 
		m_acquisition_worker->start(acqui_mode);
		//if (t1.joinable()) t1.join();
		//to dispaly all the patterns not for image acuisation!
		return;
	}

	if (disp_mode == DisplayMode::Automatic) {
		AcquisitionMode acqui_mode = AcquisitionMode::Automatic;
		for (const auto& m_pattern : m_screen->m_patterns) {
			std::thread t1 = std::thread(&Screen::showImage, m_screen.get(), m_pattern);
			std::thread t2 = std::thread(&AcquisitionWorker::start, m_acquisition_worker.get(), acqui_mode);
			//std::this_thread::sleep_for(std::chrono::seconds(1));
			
			m_screen->stopDisplaying();
			//std::thread t1(&AcquisitionWorker::start, m_acquisition_worker.get(), acqui_mode);
			//m_screen->showImage(m_pattern);
			if (t1.joinable()) t1.join();
			if (t2.joinable()) t2.join();
		}
	}
}
