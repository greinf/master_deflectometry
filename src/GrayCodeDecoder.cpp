#include "GrayCodeDecoder.hpp"
#include "GrayCodeConfig.hpp"

#include <opencv2/opencv.hpp>
#include <vector>
#include <iterator>
#include <algorithm>
#include <imgProcessing.hpp>
#include <limits>


std::vector<cv::Mat> GrayCodeDecoder::decoding(GrayCodeConfig& config)
{
	auto eval_params = config.getEvalParameter();
	CV_Assert(eval_params.start != eval_params.end);

	// Wrong Picture size !!!
	std::all_of(eval_params.start, eval_params.end,
		[&](const cv::Mat& img) -> bool {
			return eval_params.start->size() == img.size();
		});

	const cv::Size input_size = eval_params.start->size();

	const int n_pics_x = config.getBitdepth(config.creation.resolution_x);
	const int n_pics_y = config.getBitdepth(config.creation.resolution_y);

	// Extract for x and y
	std::vector<cv::Mat> grayConfig(eval_params.start, eval_params.end);

	std::vector<cv::Mat> grayBinaries = binaries(grayConfig, config);
	const std::size_t contain_size = grayBinaries.size();

	// X-Bilder extrahieren
	auto it_start_x = grayBinaries.begin();
	auto it_end_x = it_start_x + n_pics_x;
	std::vector<cv::Mat> grayBinaries_X(it_start_x, it_end_x);

	// Y-Bilder extrahieren 
	auto it_start_y = it_end_x;
	auto it_end_y = it_start_y + n_pics_y;
	std::vector<cv::Mat> grayBinaries_Y(it_start_y, it_end_y);

	// Sicherstellen, dass die Maske danach noch kommt
	CV_Assert(grayBinaries.size() >= (n_pics_x + n_pics_y + 1));

	// Create the output images
	for (auto& img : config.results.result_img) {
		img = cv::Mat(grayConfig[0].size(), 
			CV_32F, 
			cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
	}

	cv::parallel_for_(cv::Range(0, input_size.height),
		[&](const cv::Range& range) {
			for (int row = range.start; row < range.end; ++row) {
				const uchar* mask_ptr = grayBinaries[contain_size - 1].ptr<uchar>(row);
				for (int cols = 0; cols < input_size.width; ++cols) {
					if (mask_ptr[cols] == 0) continue;
					Samples sample_x{}, sample_y{};
					sample_x.pixel_x = sample_y.pixel_x = cols;
					sample_x.pixel_y = sample_y.pixel_y = row;

					sample_x.orientation = Samples::orientation::x_dir;
					extractSample(
						grayBinaries_X,
						config,
						sample_x);

					sample_y.orientation = Samples::orientation::y_dir;
					extractSample(
						grayBinaries_Y,
						config,
						sample_y
					);

					gray2dec(config, sample_x);
					gray2dec(config, sample_y);

					// Just copy the value in the output container ! 
					config.results.result_img[0].ptr<float>(sample_x.pixel_y)[sample_x.pixel_x] =
						static_cast<float>(sample_x.m_result);
					config.results.result_img[1].ptr<float>(sample_y.pixel_y)[sample_y.pixel_x] =
						static_cast<float>(sample_y.m_result);
				}
			}
		});

	/*for (const auto& img : config.results.result_img) {
		cv::Mat norm;
		cv::normalize(img, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
		cv::imshow("img", norm);
		cv::waitKey(0);
		cv::destroyWindow("img");
	}*/

	// Overloaded typecast std::vector<cv::Mat> only return the result images.
	return config;
}

void GrayCodeDecoder::gray2dec(
	GrayCodeConfig& config,
	Samples& sam)
{
	CV_Assert(sam.pixel_x != -1 && sam.pixel_y != -1);

	const std::size_t bit_depth = (sam.orientation == Samples::x_dir) ?
		(config.getBitdepth(config.creation.pixel_x)) : 
		(config.getBitdepth(config.creation.pixel_y));

	boost::dynamic_bitset<> output(bit_depth, 0);
	bool last_bit = false;
	unsigned long binary_val = 0;

	for (int i = bit_depth - 1; i >= 0; --i)
	{
		bool current_gray_bit = sam.m_data.test(i);

		bool binary_bit = current_gray_bit ^ last_bit;

		if (binary_bit) {
			output.set(i);
		}
		last_bit = binary_bit;
	}

	sam.m_result = static_cast<int>(output.to_ulong());

	return;
}


void GrayCodeDecoder::extractSample(
	const std::vector<cv::Mat>& img,
	GrayCodeConfig& config,
	Samples& sam)
{
	CV_Assert(sam.pixel_x >= 0 && sam.pixel_y >= 0);
	CV_Assert(std::all_of(img.begin(), img.end(),
		[](const cv::Mat& img) -> bool {
			return (img.type() == CV_8U) &&
				(img.channels() == 1);
		}));

	const std::size_t bin_size{ img.size() };

	const int bitdepth_res = (sam.orientation == Samples::orientation::x_dir) ?
		(config.getBitdepth(config.creation.resolution_x)) : 
		(config.getBitdepth(config.creation.resolution_y));

	CV_Assert(static_cast<std::size_t>(bitdepth_res) == bin_size);

	const int bitdepth_real = (sam.orientation == Samples::orientation::x_dir) ?
		(config.getBitdepth(config.creation.pixel_x)) :
		(config.getBitdepth(config.creation.pixel_y));

	sam.m_data = boost::dynamic_bitset(static_cast<std::size_t>(bitdepth_real), 0);
	
	for (std::size_t i = 0; i < bin_size; ++i)
	{
		std::size_t counter;
		if (config.creation.starBit == GrayCodeConfig::msb) {
			counter = static_cast<std::size_t>(bitdepth_real) - 1 - i;
		}
		else counter = i;

		const uchar* bin_ptr = img[i].ptr<uchar>(sam.pixel_y);
		bool bit = static_cast<bool>(bin_ptr[sam.pixel_x] != 0);
		sam.m_data.set(counter, bit);
	}

	return;
}


std::vector<cv::Mat> GrayCodeDecoder::binaries(
	const std::vector<cv::Mat>& img,
	GrayCodeConfig& config)
{
	CV_Assert(!img.empty());
	CV_Assert(std::all_of(img.begin(), img.end(),
		[](const cv::Mat& pic) -> bool {
			return (pic.channels() == 1);
		}));

	// The White black images is appended to the back
	// The GrayCode images are only of img.size() - 2
	const std::size_t sz = img.size(); 
	const auto& eval_param = config.getEvalParameter();
	std::vector<cv::Mat> binary;
	
	const cv::Mat mask = config.results.mask = m_img_processing.grayCalibMask(
		std::vector<cv::Mat>{*std::prev(img.end()), *std::prev(img.end(), 2)});

	const int bitdepth_x = config.getBitdepth(config.creation.resolution_x);
	const int bitdepth_y = config.getBitdepth(config.creation.resolution_y);

	if (config.creation.inverse)
	{
		// double for each bitdepth and two images for masking
		const int expected{ bitdepth_x * 2 + bitdepth_y * 2 + 2};

		CV_Assert(expected == static_cast<int>(sz));
		CV_Assert(sz % 2 != 1);

		for (std::size_t i = 0; i < ((sz-2)/2); ++i) {
			// Change type to double
			int bitdepth = bitdepth_x;
			if (i >= bitdepth_x) bitdepth = bitdepth_y;
			cv::Mat gray64, gray_inv64, gray64mask, gray_inv64mask;
			img[i].convertTo(gray64, CV_64F);
			img[i + bitdepth].convertTo(gray_inv64, CV_64F);

			// Mask the images
			gray64mask = gray64.setTo(0, ~mask);
			gray_inv64mask = gray_inv64.setTo(0, ~mask);

			cv::Mat sub = gray64mask - gray_inv64mask; // Subract inverse from normal GrayCode
			sub = cv::max(sub, 0);
			cv::Mat sub_bin, sub8u;

			cv::checkRange(sub, false, nullptr, -1e-9, 255 + 1e-9);

			sub.convertTo(sub8u, CV_8U);

			cv::threshold(sub8u, sub_bin, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
			binary.push_back(sub_bin);
		}
	}

	else {
		const int expected{ bitdepth_x + bitdepth_y + 2 };
		CV_Assert(expected == static_cast<int>(sz));
		
		for (std::size_t i = 0; i < (sz - 2); ++i) {
			// Mask the image
			cv::Mat gray_masked, gray_masked8u, gray_masked_bin;

			gray_masked = img[i].clone().setTo(0, ~mask);

			cv::checkRange(gray_masked, false, nullptr, -1e-9, 255 + 1e-9);

			gray_masked.convertTo(gray_masked8u, CV_8U);

			cv::threshold(gray_masked8u, gray_masked_bin, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

			binary.push_back(gray_masked_bin);
		}
	}

	/*for (const auto& img : binary) {
		cv::imshow("binaries", img);
		cv::waitKey(0);
	}*/

	// Append the mask at the end of the Container.
	binary.push_back(mask);

	return binary;

}



