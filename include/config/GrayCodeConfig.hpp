#ifndef GRAYCODECONFIG_HPP
#define GRAYCODECONFIG_HPP	

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <boost/dynamic_bitset.hpp>
#include <cassert>

// pixel_x pixel_y -> size of the pattern created 
// resolution x resolution y -> expected resolution for grayCode in pixels
// Bitdepth used for both resolutions 
// bitdepth_x bitdpeth_y -> calculated when creating GrayCode Images
// inverse -> bool value if the inverse GrayCode is also available
// Startbit -> is StartBit MSB or LSB


// Holds to Struct which delceration is both private. -> No instance of them can be created outside of the struct
// Because there is public member variable of generationparamter -> this and it´s member are accessilbe form outside
// But EvalParamter is both the decleration and member variable Private -> so the Type is not known outside of the class
// If getEvalParamter is called it must be accessed via auto!!!
struct GrayCodeConfig {
public: 
	enum StartBit {
		msb,
		lsb
	};
private:
	struct Eval_parameter {
		bool inverse{};
		int bitdepth_x{};
		int bitdepth_y{};
		StartBit startbit;

		// DataPosition, Located in Frame Store but this is nicer
		std::vector<cv::Mat>::const_iterator start{};
		std::vector<cv::Mat>::const_iterator end{};
	};

	struct generationParamter {
		int pixel_x{};
		int pixel_y{};
		int resolution_x{};
		int resolution_y{};
		bool inverse{ false };
		StartBit starBit;
	};

	Eval_parameter decoding{};

	struct Results {
		std::array<cv::Mat, 2> result_img{};
		cv::Mat mask{};
	};

public:
	generationParamter creation{};

	Results results{};
	
	Eval_parameter& getEvalParameter() { return decoding; }

	operator std::vector<cv::Mat>() {
		std::vector<cv::Mat> out(2);
		out[0] = results.result_img[0];
		out[1] = results.result_img[1];
		return out;
	}
};

namespace GrayCode {
	// Input[0]: number_to_gray: the decimal that is transfered
	// Input[1]: numb of bits used for the binary
	inline boost::dynamic_bitset<> grayCode(
		unsigned long number_to_gray,
		const std::size_t num_of_bits)
	{
		// Boundary taken from W.Kahan 
		assert(num_of_bits < 27);
		assert(number_to_gray >= 0 && num_of_bits > 0);
		// dynamic bitsets default num_bin[0] as LSB
		boost::dynamic_bitset<> num_bin(num_of_bits, number_to_gray);
		boost::dynamic_bitset<> grayCode = num_bin ^ (num_bin >> 1);
		
		return grayCode;
	}
}

#endif 