#ifndef	DEFLECTOMETRY_H
#define DEFLECTOMETRY_H

#include <memory>
#include <vector>
#include <exception>
#include <iostream>
#include <cassert>
#include <thread>

enum class AcquisitionMode;

enum class Shift_mode;

enum class DisplayMode;


class Screen;
class AcquisitionWorker;

class Deflectometry {
public:
	 Deflectometry();

	 void start_meassurement(Shift_mode, DisplayMode, int);

private:
	std::shared_ptr<Screen> m_screen{nullptr};

	std::shared_ptr<AcquisitionWorker> m_acquisition_worker{ nullptr };

};

#endif