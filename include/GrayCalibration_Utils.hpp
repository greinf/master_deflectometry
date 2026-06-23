#ifndef GRAYCALIBRATION_UTILS_HPP
#define GRAYCALIBRATION_UTILS_HPP
#include <algorithm>


namespace _Base {
	struct specifier_Base{};
}

namespace GrayCalibration_specifier {

	namespace Active {
		struct LUT:_Base::specifier_Base {};
		struct Model:_Base::specifier_Base {};
		struct Model_Bias:_Base::specifier_Base {};
		struct LocLUT:_Base::specifier_Base{};
	}

	namespace Passive {
		struct LUT:_Base::specifier_Base {};
		struct Model:_Base::specifier_Base {};
		struct Model_Bias:_Base::specifier_Base {};
		struct LocLUT:_Base::specifier_Base{};
	}

	struct NoCalib:_Base::specifier_Base {};
}

struct Gray_Calib_Stats {
	int n = 0;
	int iters = 0;
	bool converged = false;

	double gamma = 1.0;
	double Imax = 1.0;
	double I_0 = 0.0;

	double sse = 0.0;
	double rmse = 0.0;
	double r2 = 0.0;
};

struct Gray_Calib_Result {
	Gray_Calib_Stats stats;
	std::array<uint8_t, 256> lut{};
};


auto LutEmpty = [](std::array<std::pair<double, double>, 256>& LUT) -> bool
	{
		// If all values are zero -> Array must have been empty
		return std::all_of(LUT.begin(), LUT.end(), [](const std::pair<double, double>& x) {
			return x.first == 0.0;
			});
	};

#endif