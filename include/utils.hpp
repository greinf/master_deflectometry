#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <cassert>
#include <algorithm>
#include <type_traits>


// Small helper that allows the GrayCalibVector struct to treat std::pair like 2d vector
inline std::pair<int, int> operator+
(const std::pair<int, int>& left, const std::pair<int, int>& right)
{
	return { left.first + right.first, left.second + right.second };
}
	
inline std::pair<double, double> operator+
(const std::pair<double, double>& left, const std::pair<double, double>& rigth) {
	return { left.first + rigth.first, left.second + rigth.second };
}

// defines a ROI on an image with [x_start, x_end), [y_start, y_end)
// This is primary Template. Compiler will look for better match and go to specialization
// The specialization is better -> no use of defaulted parameter
// Specialization works only if type is int or double. If not compiler will fall back to primary.
// Because this only decleration 
// Used in GrayCalibVector
template<typename T, typename enable = void>
struct RoiBorders;

// All datapoints are storeds as (x,y)
template<typename T>
struct RoiBorders < T, typename std::enable_if<
	std::is_same<T,int>::value || std::is_same<T,double>::value>::type>
{
	std::pair<T, T> left_up_corner{}; // inclusive
	std::pair<T, T> left_down_corner{}; // inclusive
	std::pair<T, T> right_up_corner{}; // exclusive
	std::pair<T, T> right_down_corner{}; // exclusive

	operator std::vector<std::pair<int, int>>() const {
		std::vector<std::pair<int, int>> container;
		container.push_back(left_up_corner);
		container.push_back(right_up_corner);
		container.push_back(left_down_corner);
		container.push_back(right_down_corner);
		return container;
	}

	operator std::vector<std::pair<double, double>>() const {
		std::vector<std::pair<double, double>> container;
		container.push_back(left_up_corner);
		container.push_back(right_up_corner);
		container.push_back(left_down_corner);
		container.push_back(right_down_corner);
		return container;
	}
};


// All datapoints are stored as (x,y)
struct Gray_section_data {
	RoiBorders<int> roi_img{};
	RoiBorders<double> roi_obj{};
	std::vector<double> gray_val{}; // mean val
	std::vector<double> s_deviation{}; //standard deviation over the 
};

enum class CalibrationMethod {
	None,
	Lut,
	Active,
	Passive,
	Bias_Passive
};

#endif