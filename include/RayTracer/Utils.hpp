#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <array>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <stdexcept>




struct RayStructure {
	const Eigen::Vector3d* dir;
	const Eigen::Vector3d* origin;
};

struct SamplingSetting {
	std::size_t samples_x{ 1 };
	std::size_t samples_y{ 1 };

	bool operator==(const SamplingSetting& setting) const {
		if (this->samples_x != setting.samples_x) return false;
		if (this->samples_y != setting.samples_y) return false;
		return true;
	}
};

namespace Protocoll {

	class CameraData {
		bool is_Intrinsic(const Eigen::Matrix3d& matrix) const {
			constexpr double eps{ 1e-9 };

			const Eigen::Matrix<bool, 3, 3> valid{
				{true, false, true},
				{false, true, true},
				{false, false, true}
			};

			return matrix.allFinite() && (valid.array() == false).select(matrix.array(), 0.0).matrix().isZero(eps) &&
				matrix(0, 0) > 0 && matrix(1, 1) > 0 && std::abs(matrix(2, 2) - 1) < eps;
		}
	public:
		CameraData() = default;

		bool validAndSet() const {
			constexpr double eps{ 1e-9 };
			if (objective == nullptr) return false;
			if (!is_Intrinsic(intrinsicMatrix) || !distortionCoefficients.allFinite()) return false;
			if (distortionCoefficients.size() != 0 && distortionCoefficients.size() != 4 && distortionCoefficients.size() != 5) return false;
			if (!std::isfinite(used_f_num) || !std::isfinite(focus_points.NearPoint) ||
				!std::isfinite(focus_points.ObjectLength) || std::isnan(focus_points.FarPoint) ||
				focus_points.NearPoint <= 0 || focus_points.ObjectLength <= focus_points.NearPoint ||
				focus_points.FarPoint <= focus_points.ObjectLength) return false;
			if (std::abs(focus_points.FarPoint) < eps) return false;
			if (std::abs(focus_points.NearPoint) < eps) return false;
			if (std::abs(focus_points.ObjectLength) < eps) return false;
			if (used_f_num < eps) return false;

			return true;
		}

		// Class Objective is part of Camera Data 
		struct ObjectiveData {
			const double focal_length{};             // [mm]
			const double entrancePupile_diameter{};  // [mm]
			const double exitPupile_diameter{};      // [mm]
			const double min_Object_distance{};      // [m]
			const double minIris{};                  // f-number_min
			const double maxIris{};                  // f-number_max
			const std::string name{};				 // name

			bool valid_Iris(const double& f_num) const noexcept {
				return minIris <= f_num && maxIris >= f_num;
			}

			struct FocusPoints {
				double ObjectLength{};
				double NearPoint{};
				double FarPoint{};

				bool operator==(const FocusPoints& data) const {
					if (this->ObjectLength != data.ObjectLength) return false;
					if (this->NearPoint != data.NearPoint) return false;
					if (this->FarPoint != data.FarPoint) return false;
					return true;
				}
			};

			double pupileMagnification() const noexcept {
				return exitPupile_diameter / entrancePupile_diameter;
			}
			
			FocusPoints generateFocusPoints(
				const double& image_scale_goal_abs,
				const double& f_num,
				const double& circle_confusion,
				const double& object_length) const
			{
				if (image_scale_goal_abs <= 0.0) throw std::invalid_argument("Image Scale must be absolute value");

				const double pupile_magn{ pupileMagnification() };

				const double f_num_corrected{ f_num * (1 + (image_scale_goal_abs / pupile_magn)) };

				const double focal_squared{ focal_length * focal_length };

				const double denom{ f_num_corrected * circle_confusion * (object_length - focal_length) };

				const double numerator{ focal_squared * object_length };

				const double nearPoint{ numerator / (focal_squared + denom) };

				const double farPoint{ numerator / (focal_squared - denom) };

				return { object_length, nearPoint, farPoint };
			}
		};

		// Pentax Objective form the catalog 
		static inline const CameraData::ObjectiveData C2514_M{
			25.0,   // focal_length [mm]
			17.6,   // entrance pupil diameter [mm]
			22.1,   // exit pupil diameter [mm]
			0.25,   // minimum object distance [m]
			1.4,    // minimum f-number
			16.0,   // maximum f-number
			"C2514_M"
		};

		static inline const CameraData::ObjectiveData C3516_M{
			34.0,   // actual focal length from optical data [mm]
			20.8,   // entrance pupil diameter [mm]
			21.0,   // exit pupil diameter [mm]
			0.40,   // minimum object distance [m]
			1.6,    // minimum f-number
			16.0,    // maximum f-number
			"C3516_M" // name
		};

		static inline const CameraData::ObjectiveData C5028_M{
			50.0,   // focal_length [mm]
			18.2,   // entrance pupil diameter [mm]
			9.7,    // exit pupil diameter [mm]
			0.90,   // minimum object distance [m]
			2.8,    // minimum f-number
			22.0,   // maximum f-number
			"C5028_M" //name
		};

		static inline const CameraData::ObjectiveData C7528_M{
			72.8,   // actual focal length from optical data [mm]
			25.9,   // entrance pupil diameter [mm]
			12.9,   // exit pupil diameter [mm]
			0.70,   // minimum object distance [m]
			2.8,    // minimum f-number
			32.0,   // maximum f-number
			"C7528_M" // name
		};


		// Will throw if file does not exist 
		static CameraData read(const std::string& path) {
			cv::FileStorage fs{};

			if (!fs.open(path, cv::FileStorage::READ)) 
				throw std::invalid_argument("File Already opened");

			cv::FileNode node = fs[std::string(m_keys[0])];
			if (node.empty()) {
				std::cout << "Node is empty \n";
				return {};
			}
			if (!node.isMap())
				throw std::logic_error("Node Must be a map");

			CameraData data{};

			cv::Mat distortion{}, intrinsic{};
			std::string name{};

			node[std::string(m_keys[1])] >> name;

			// We Just give the pointer if the name fits 
			for (const auto& obj : m_objectiveData) {
				if (obj->name == name) {
					data.objective = obj;
					break;
				}
			}

			node[std::string(m_keys[6])] >> distortion;
			node[std::string(m_keys[7])] >> intrinsic;
			node[std::string(m_keys[8])] >> data.used_f_num;
			node[std::string(m_keys[9])] >> data.focus_points.NearPoint;
			node[std::string(m_keys[10])] >> data.focus_points.FarPoint;
			node[std::string(m_keys[11])] >> data.focus_points.ObjectLength;

			if (intrinsic.rows != 3 || intrinsic.cols != 3 || intrinsic.channels() != 1)
				throw std::invalid_argument("Intrinsic matrix must be 3x3");
			intrinsic.convertTo(intrinsic, CV_64F);
			if (!distortion.empty() && (distortion.channels() != 1 ||
				(distortion.rows != 1 && distortion.cols != 1) ||
				(distortion.total() != 4 && distortion.total() != 5)))
				throw std::invalid_argument("Only 4/5 coefficient Brown vectors are supported");
			data.distortionCoefficients = Eigen::VectorXd::Zero(5);
			if (!distortion.empty()) {
				cv::Mat flat;
				distortion.clone().reshape(1, static_cast<int>(distortion.total())).convertTo(flat, CV_64F);
				for (int i = 0; i < flat.rows; ++i) data.distortionCoefficients(i) = flat.at<double>(i, 0);
			}
			cv::cv2eigen(intrinsic, data.intrinsicMatrix);
			if (!data.validAndSet()) throw std::runtime_error("Invalid camera metadata");

			fs.release();

			return data;
		}

		void save(const std::string& path) const {
			cv::FileStorage file{};
			if (!validAndSet()) throw std::invalid_argument("Refusing invalid camera metadata");

			if (!file.open(path, cv::FileStorage::READ))
				throw std::runtime_error("Opening File: READ failed");

			cv::FileNode node = file[std::string(m_keys[0])];

			if (!node.empty())
				throw std::runtime_error("File contains already data");

			file.release();

			if (!file.open(path, cv::FileStorage::APPEND))
				throw std::runtime_error("Opening File: WRITE failed");

			// Create cv::Mat 
			// Distortion

			Eigen::Index dist_elements{ distortionCoefficients.size() };

			cv::Mat dist(dist_elements, 1, CV_64F), intrinsic(3, 3, CV_64F);

			if (dist_elements > 6) std::cout << "WARNING: More distortion Coefficients as expected \n" <<
				dist_elements << "Coefficients found \n";

			for (Eigen::Index row = 0; row < dist_elements; ++row) {
				dist.at<double>(static_cast<std::size_t>(row), 0) = distortionCoefficients(row, 0);
			}

			// intrinisc MAtrix
			for (Eigen::Index row = 0; row < intrinsicMatrix.rows(); ++row) {
				for (Eigen::Index col = 0; col < intrinsicMatrix.cols(); ++col) {
					intrinsic.at<double>(static_cast<int>(row), static_cast<int>(col)) =
						intrinsicMatrix(row, col);
				}
			}

			file << std::string(m_keys[0]) << "{";
			file << std::string(m_keys[1]) << objective->name;
			file << std::string(m_keys[2]) << objective->entrancePupile_diameter;
			file << std::string(m_keys[3]) << objective->exitPupile_diameter;
			file << std::string(m_keys[4]) << objective->minIris;
			file << std::string(m_keys[5]) << objective->maxIris;

			file << std::string(m_keys[6]) << dist;
			file << std::string(m_keys[7]) << intrinsic;
			file << std::string(m_keys[8]) << used_f_num;
			file << std::string(m_keys[9]) << focus_points.NearPoint;
			file << std::string(m_keys[10]) << focus_points.FarPoint;
			file << std::string(m_keys[11]) << focus_points.ObjectLength;
			file << "}";

			file.release();

			return;
		}

		// Members
		const ObjectiveData* objective{ nullptr };
		ObjectiveData::FocusPoints focus_points{};
		double used_f_num{};
		Eigen::Matrix3d intrinsicMatrix = Eigen::Matrix3d::Zero();
		Eigen::VectorXd distortionCoefficients{};

		bool operator==(const CameraData& data) const {
			if (!(this->objective == data.objective)) return false;
			if (!(this->focus_points == data.focus_points)) return false;
			if (this->used_f_num != data.used_f_num) return false;
			if (!this->intrinsicMatrix.isApprox(data.intrinsicMatrix, 1e-9)) return false;
			if (this->distortionCoefficients.size() != data.distortionCoefficients.size() ||
				!this->distortionCoefficients.isApprox(data.distortionCoefficients, 1e-9)) return false;
			return true;
		}

	private:
		static constexpr std::array<std::string_view, 12> m_keys{
		std::string_view{"Camera_Data"},
		std::string_view{"Objective_Name"},
		std::string_view{"EnrancePupileDiameter"},
		std::string_view{"ExitPupileDiameter"},
		std::string_view{"Min_F_Number"},
		std::string_view{"Max_F_Number"},
		std::string_view{"Distortion_Coefficient"},
		std::string_view{"Camera_Matrix"},
		std::string_view{"Used_F_Number"},
		std::string_view{"NearPoint"},
		std::string_view{"FarPoint"},
		std::string_view{"ObjectDistance"}
		};

		static constexpr std::array<const CameraData::ObjectiveData*, 4> m_objectiveData
		{ {
			&C2514_M,
			&C3516_M,
			&C5028_M,
			&C7528_M
		} };

	};

	
	class Simulation {
	public:
		SamplingSetting settings{};
		cv::Size target_image_size{2464, 2056};
		bool centered_subsamples = true;
		bool metadata_explicit = true;
		std::vector<Eigen::MatrixXd> m_simulated_images{};

		// Old Binary.bin files contain no sampling metadata. The caller MUST
		// specify it; neither the old default 1x1 nor a guessed sensor size is safe.
		void setLegacyMetadata(SamplingSetting sample, cv::Size size, bool centered = false) {
			if (metadata_explicit)
				throw std::logic_error("Metadata already exists; do not override it as legacy");
			settings = sample; target_image_size = size; centered_subsamples = centered;
			validateMetadata();
			for (const auto& img : m_simulated_images) validateImage(img);
			metadata_explicit = true;
		}

		void validateMetadata() const {
			if (settings.samples_x == 0 || settings.samples_y == 0 ||
				settings.samples_x > static_cast<size_t>(std::numeric_limits<int>::max()) ||
				settings.samples_y > static_cast<size_t>(std::numeric_limits<int>::max()) ||
				target_image_size.width <= 0 || target_image_size.height <= 0 ||
				static_cast<std::uint64_t>(target_image_size.width) * settings.samples_x > INT_MAX ||
				static_cast<std::uint64_t>(target_image_size.height) * settings.samples_y > INT_MAX)
				throw std::invalid_argument("Invalid sensor size or sampling metadata");
		}
		void validateImage(const Eigen::MatrixXd& raw) const {
			validateMetadata();
			if (raw.rows() != static_cast<Eigen::Index>(target_image_size.height) * settings.samples_y ||
				raw.cols() != static_cast<Eigen::Index>(target_image_size.width) * settings.samples_x ||
				!raw.allFinite())
				throw std::invalid_argument("Raw image size/values disagree with sampling metadata");
		}
		bool validAndSet() const {
			if (!metadata_explicit || m_simulated_images.empty()) return false;
			try {
				for (const auto& img : m_simulated_images) validateImage(img);
				return true;
			} catch (const std::exception&) { return false; }
		}
		std::size_t get_Number_of_Images() const { return m_simulated_images.size(); }
		bool operator==(const Simulation& d) const {
			if (!(settings == d.settings) || target_image_size != d.target_image_size ||
				centered_subsamples != d.centered_subsamples || metadata_explicit != d.metadata_explicit ||
				m_simulated_images.size() != d.m_simulated_images.size()) return false;
			for (size_t i = 0; i < m_simulated_images.size(); ++i) {
				const auto& a = m_simulated_images[i]; const auto& b = d.m_simulated_images[i];
				if (a.rows() != b.rows() || a.cols() != b.cols() || !a.isApprox(b, 1e-9)) return false;
			}
			return true;
		}
		[[nodiscard]] Eigen::MatrixXd prepareImageforProcessing(
			const Eigen::MatrixXd& raw, const int kernel_bias = 2) const {
			if (!metadata_explicit)
				throw std::runtime_error("Legacy data: setLegacyMetadata is required before processing");
			if (kernel_bias < -1 || kernel_bias > 100)
				throw std::invalid_argument("Kernel bias must be -1 (off) or 0..100");
			validateImage(raw);
			cv::Mat input, filtered, resized;
			cv::eigen2cv(raw, input);
			filtered = input;
			if (kernel_bias >= 0) {
				auto oddKernel = [kernel_bias](size_t samples) {
					const std::uint64_t n = static_cast<std::uint64_t>(samples) + kernel_bias;
					if (n > INT_MAX - 1) throw std::overflow_error("Kernel is too large");
					return static_cast<int>(n) | 1;
				};
				cv::GaussianBlur(input, filtered,
					{oddKernel(settings.samples_x), oddKernel(settings.samples_y)}, 0.0);
			}
			// INTER_AREA is the SIXTH argument, not the fourth (which is fx).
			cv::resize(filtered, resized, target_image_size, 0.0, 0.0, cv::INTER_AREA);
			Eigen::MatrixXd result; cv::cv2eigen(resized, result); return result;
		}
		[[nodiscard]] std::vector<Eigen::MatrixXd> prepareAllImages(int kernel_bias = 2) const {
			std::vector<Eigen::MatrixXd> result;
			result.reserve(m_simulated_images.size());
			for (const auto& raw : m_simulated_images) result.push_back(prepareImageforProcessing(raw, kernel_bias));
			return result;
		}
		void show(const size_t& i, const int& kernel_bias = 2) const {
			cv::Mat display;
			cv::eigen2cv(prepareImageforProcessing(m_simulated_images.at(i), kernel_bias), display);
			cv::normalize(display, display, 0, 255, cv::NORM_MINMAX, CV_8U);
			cv::namedWindow("Output", cv::WINDOW_NORMAL); cv::imshow("Output", display); cv::waitKey(0);
		}
		void showAll(int kernelBias = 2) const {
			for (size_t i = 0; i < m_simulated_images.size(); ++i) show(i, kernelBias);
		}
		void save(const std::string& path) const {
			checkBinaryPlatform();
			if (!metadata_explicit) throw std::runtime_error("Refusing to save ambiguous legacy sampling");
			validateMetadata();
			for (const auto& img : m_simulated_images) validateImage(img);
			std::ofstream fs(path, std::ios::binary | std::ios::trunc);
			if (!fs) throw std::runtime_error("Cannot open simulation output");
			fs << "RayTracerSimulationV2\n"
				<< settings.samples_x << ' ' << settings.samples_y << ' '
				<< target_image_size.width << ' ' << target_image_size.height << ' '
				<< static_cast<int>(centered_subsamples) << ' ' << m_simulated_images.size() << '\n'
				<< "IEEE754_LE_COLUMN_MAJOR_F64\n";
			for (const auto& img : m_simulated_images) {
				fs << "BinaryImage\n" << img.rows() << ' ' << img.cols() << " double\n";
				fs.write(reinterpret_cast<const char*>(img.data()), byteCount(img.rows(), img.cols()));
				fs << "\n\n";
			}
			fs.flush();
			if (!fs) throw std::runtime_error("Simulation write failed");
		}
		static Simulation read(const std::filesystem::path& path) {
			checkBinaryPlatform();
			std::ifstream fs(path, std::ios::binary);
			if (!fs) throw std::runtime_error("Cannot open simulation file");
			Simulation data;
			std::string token;
			size_t count = 0;
			bool v2 = false;
			if (fs >> token) v2 = token == "RayTracerSimulationV2";
			if (v2) {
				int centered = -1;
				if (!(fs >> data.settings.samples_x >> data.settings.samples_y >>
					data.target_image_size.width >> data.target_image_size.height >> centered >> count) ||
					(centered != 0 && centered != 1) || count > 100000)
					throw std::runtime_error("Invalid V2 header");
				data.centered_subsamples = centered == 1;
				data.validateMetadata();
				if (!(fs >> token) || token != "IEEE754_LE_COLUMN_MAJOR_F64")
					throw std::runtime_error("Unsupported binary layout");
			} else {
				data.metadata_explicit = false;
				data.target_image_size = {};
				fs.clear(); fs.seekg(0);
			}
			while (true) {
				if (!(fs >> token)) {
					if (fs.eof()) break;
					throw std::runtime_error("Failed reading binary marker");
				}
				if (token != "BinaryImage") throw std::runtime_error("Invalid binary image marker");
				if (v2 && data.m_simulated_images.size() >= count)
					throw std::runtime_error("Extra images after declared count");
				std::int64_t rows = 0, cols = 0;
				std::string type;
				if (!(fs >> rows >> cols >> type) || type != "double" || rows <= 0 || cols <= 0 ||
					rows > INT_MAX || cols > INT_MAX) throw std::runtime_error("Invalid binary dimensions/type");
				char newline = 0; fs.get(newline);
				if (newline == '\r') fs.get(newline);
				if (newline != '\n') throw std::runtime_error("Missing binary payload separator");
				const auto bytes = byteCount(rows, cols);
				const auto offset = fs.tellg();
				if (offset < 0 || static_cast<std::uintmax_t>(offset) > std::filesystem::file_size(path) ||
					static_cast<std::uintmax_t>(bytes) > std::filesystem::file_size(path) -
						static_cast<std::uintmax_t>(offset))
					throw std::runtime_error("Truncated binary image");
				if (v2 && (rows != static_cast<std::int64_t>(data.target_image_size.height) * data.settings.samples_y ||
					cols != static_cast<std::int64_t>(data.target_image_size.width) * data.settings.samples_x))
					throw std::runtime_error("Binary dimensions disagree with metadata");
				Eigen::MatrixXd image(rows, cols);
				if (!fs.read(reinterpret_cast<char*>(image.data()), bytes))
					throw std::runtime_error("Short binary image read");
				if (!image.allFinite()) throw std::runtime_error("Non-finite binary image");
				data.m_simulated_images.push_back(std::move(image));
			}
			if (v2 && data.m_simulated_images.size() != count) throw std::runtime_error("Missing binary images");
			return data;
		}
	private:
		static void checkBinaryPlatform() {
			const std::uint16_t test = 1;
			if (sizeof(double) != 8 || !std::numeric_limits<double>::is_iec559 ||
				*reinterpret_cast<const unsigned char*>(&test) != 1)
				throw std::runtime_error("Binary format requires little-endian IEEE754 doubles");
		}
		static std::streamsize byteCount(std::int64_t rows, std::int64_t cols) {
			if (rows <= 0 || cols <= 0 ||
				static_cast<std::uint64_t>(rows) > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) / 8 /
					static_cast<std::uint64_t>(cols)) throw std::overflow_error("Image payload size overflow");
			return static_cast<std::streamsize>(rows) * static_cast<std::streamsize>(cols) * 8;
		}
	};


	class CalibrationData {
	public:
		// Member
		bool validAndSet() const {
			constexpr double eps = 1e-9;
			if (pattern_x < 4 || pattern_y < 4) return false;
			if (!std::isfinite(pattern_width) || pattern_width < eps) return false;
			if (translationMatrix.empty()) return false;
			for (const auto& pose : translationMatrix) {
				const Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
				if (!pose.allFinite() || !pose.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-9) ||
					!(R.transpose() * R).isApprox(Eigen::Matrix3d::Identity(), 1e-7) ||
					std::abs(R.determinant() - 1) > 1e-7) return false;
			}

			return true;
		}
		static constexpr std::array<std::string_view, 5> m_keys{
			std::string_view{"CalibrationBoard"},
			std::string_view{"pattern_x"},
			std::string_view{"pattern_y"},
			std::string_view{"pattern_width"},
			std::string_view{"translation_matrices"}
		};

		std::size_t get_number_of_translation() const {
			return translationMatrix.size();
		}

		std::uint16_t pattern_x{};
		std::uint16_t pattern_y{};
		double pattern_width{};
		std::vector<Eigen::Matrix4d> translationMatrix{};

		bool operator==(const CalibrationData& data) const {
			if (this->pattern_x != data.pattern_x) return false;
			if (this->pattern_y != data.pattern_y) return false;
			if (this->pattern_width != data.pattern_width) return false;
			if (this->translationMatrix.size() != data.translationMatrix.size()) return false;

			if (std::pair<const std::vector<Eigen::Matrix4d>::const_iterator, const std::vector<Eigen::Matrix4d>::const_iterator>
			{this->translationMatrix.end(), data.translationMatrix.end()} !=
				std::mismatch(
					this->translationMatrix.begin(),
					this->translationMatrix.end(),
					data.translationMatrix.begin(),
					[](const Eigen::Matrix4d& this_mat,
						const Eigen::Matrix4d& data_mat) -> bool {
							return this_mat.isApprox(data_mat, 1e-9);
					})) return false;
			return true;
		}

		void save(const std::string& path) const {
			cv::FileStorage fs{};

			//Check
			if (!fs.open(path, cv::FileStorage::READ))
				throw std::runtime_error("Open file for Reading failed");

			cv::FileNode node = fs[std::string(m_keys[0])];

			if (!node.empty())
				throw std::runtime_error("File already contains data");

			fs.release();

			if (!fs.open(path, cv::FileStorage::APPEND))
				throw std::runtime_error("Opening the file for Writing Failed");

			fs << std::string(m_keys[0]) << "{";
			fs << std::string(m_keys[1]) << pattern_x;
			fs << std::string(m_keys[2]) << pattern_y;
			fs << std::string(m_keys[3]) << pattern_width;
			fs << std::string(m_keys[4]) << "[";

			for (const auto& matrix : translationMatrix) {
				cv::Mat translation(4, 4, CV_64F);

				// Assign elements
				for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
					for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
						translation.at<double>(row, col) = matrix(row, col);
					}
				}

				fs << translation;
			}
			fs << "]" << "}";

			fs.release();
		}

		static CalibrationData read(const std::string& path) {
			cv::FileStorage fs{};

			if (!fs.open(path, cv::FileStorage::READ)) 
				throw std::invalid_argument("Could not open. xml");
			cv::FileNode node = fs[std::string(m_keys[0])];
			if (node.empty()) {
				std::cout << "Given file does not contain needed data: \nreturn\n";
				return {};
			}
			if (!node.isMap()) throw std::logic_error("Node is supposed to be Map\n");

			if (!node[std::string(m_keys[4])].isSeq())
				throw std::logic_error("Readout at this Key should be a sequence \n");

			CalibrationData data{};

			int width = 0, height = 0;
			node[std::string(m_keys[1])] >> width;
			node[std::string(m_keys[2])] >> height;
			if (width < 4 || height < 4 || width > UINT16_MAX || height > UINT16_MAX)
				throw std::runtime_error("Invalid board dimensions");
			data.pattern_x = static_cast<std::uint16_t>(width);
			data.pattern_y = static_cast<std::uint16_t>(height);
			node[std::string(m_keys[3])] >> data.pattern_width;

			cv::FileNodeIterator start = node[std::string(m_keys[4])].begin();
			cv::FileNodeIterator end = node[std::string(m_keys[4])].end();

			std::size_t n_elements = start.remaining();

			if (n_elements == 0) {
				std::cerr << "Sequence of translation Matrix is empty \n";
			}

			data.translationMatrix.reserve(n_elements);

			for (auto it = start; it != end; ++it) {
				cv::Mat currentTranslation{};
				*it >> currentTranslation;
				if (currentTranslation.size() != cv::Size(4, 4) || currentTranslation.channels() != 1)
					throw std::runtime_error("Pose must be a 4x4 matrix");
				currentTranslation.convertTo(currentTranslation, CV_64F);

				Eigen::Matrix4d matrix = Eigen::Matrix4d::Zero();

				for (int row = 0; row < currentTranslation.rows; ++row) {
					for (int col = 0; col < currentTranslation.cols; ++col) {
						matrix(row, col) = currentTranslation.at<double>(row, col);
					}
				}

				data.translationMatrix.push_back(std::move(matrix));
			}
			if (!data.validAndSet()) throw std::runtime_error("Invalid board or pose metadata");
			return data;
		}
	};

	struct SaveProtocoll {
		std::string path{};

		bool operator==(const SaveProtocoll& data) const {
			if (this->path != data.path) return false;
			return true;
		}

		bool validAndSet() const {
			if (path.empty()) return false;
			return true;
		}

		void save(const std::filesystem::path& save_path) const {
			if (save_path.empty()) 
				throw std::invalid_argument("Filesystem class is empty");
			if (std::filesystem::exists(save_path))
				throw std::invalid_argument(".txt file at adress " + save_path.string() + " does already exist");

			std::ofstream file{};
			file.open(save_path.string(), std::ios_base::out);
			file << save_path.parent_path().string() << '\n';
			file.close();
		}

		static SaveProtocoll read(const std::string& save_path) {
			if (save_path.empty())
				throw std::invalid_argument("Filesystem instance is empty");
			if (!std::filesystem::exists(save_path))
				throw std::runtime_error("File does not exist");

			std::ifstream file{};
			file.open(save_path, std::ios_base::in);
			std::string path{};

			if (!std::getline(file, path)) throw std::runtime_error("Cannot read saved path");
			if (!path.empty() && path.back() == '\r') path.pop_back();

			SaveProtocoll save{};
			save.path = path;

			return save;
		}
	};

	
	// This struct defines mainly which variations of the synthetic groundtruth should be tested
	// 1) Noise of found image coordinates in x and y direction (calculate for a given pixeloffset)
	// 2) Maybe "undistort" images that should not have distortoin coefficient and look what happens
	// 3) Smoothing should be defined here. (Mainly the Kernel offset):
	//  Supersampling is done for each image over which we apply a smoothing whith the filter kernel is size +2 defaulted.
	// Per-axis kernel sizes are samples+kernel_offset, rounded up to an odd size.
	// The processing stage then performs true INTER_AREA downsampling.
	struct SyntheticCalibrationConfig {
		int kernel_offset{ 2 };
		// Noise that is applied on found chessboard corners 
		struct Noise {
			bool use_noise{ false };
			Eigen::Vector2d max_noise{};
			Eigen::Vector2d min_noise{};
		}
		m_noise{};

		bool validAndSet() const {
			if (kernel_offset > 100 || kernel_offset < 0) return false;
			if (!m_noise.use_noise) return true;
			if (!m_noise.max_noise.allFinite() || !m_noise.min_noise.allFinite() ||
				(m_noise.max_noise.array() < m_noise.min_noise.array()).any()) return false;

			return true;
		}
	};

	class SyntheticCalibrationConfigPacket {
		std::vector<SyntheticCalibrationConfig> m_workload{};

		bool validAndSet() const {
			return !m_workload.empty() && std::all_of(m_workload.begin(), m_workload.end(),
				[](const SyntheticCalibrationConfig& c) { return c.validAndSet(); });
		}
	};


	// Controll Class 
	class SimulationProtocoll {
	public:
		SimulationProtocoll() = default;

		bool validAndSet() const 
		{
			if (!m_camera.validAndSet()) return false;
			if (!m_calibration.validAndSet()) return false;
			const std::size_t n_trans{ m_calibration.get_number_of_translation() };
			if (!m_simulation.validAndSet()) return false;

			return n_trans == m_simulation.get_Number_of_Images();
		}

	private:
		void createEmptyProtocolXml(const std::filesystem::path& xml_path) const
		{
			cv::FileStorage storage{};
			if (!storage.open(xml_path.string(), cv::FileStorage::WRITE))
				throw std::runtime_error("Could not create protocol XML: " + xml_path.string());
			storage.release();
		}

		bool exist(const std::string& path) const noexcept {
			std::filesystem::path check(path);
			if (std::filesystem::exists(check)) return true;
			return false;
		}

		bool createDirectories(
			const std::string& save_path) const
		{
			const auto directory =
				std::filesystem::path{ save_path };


			if (!std::filesystem::create_directories(directory))
				return false;

			return true;
		}

		static constexpr const char* protocol_xml_name = "Protocoll.xml";
		static constexpr const char* simulation_binary_name = "Binary.bin";
		static constexpr const char* save_path_name = "SavePath.txt";
		static constexpr std::string_view m_fallBack{ "C:/Users/grein/Desktop/Fallback/" };

		std::string getDate() const {
			const std::chrono::system_clock::time_point time_pt =
				std::chrono::system_clock::now();
			std::time_t time{ std::chrono::system_clock::to_time_t(time_pt) };
			std::tm tm = *std::localtime(&time);

			std::stringstream ss{};

			ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");

			return ss.str();
		}

	public:
		CameraData m_camera{};
		CalibrationData m_calibration{};
		Simulation m_simulation{};
		SaveProtocoll m_save{};

		// Saves exactly ONE protocol into exactly ONE supplied directory.
		// No timestamps are added here because the batch directory itself defines the experiment.
		void saveProtocol(
			const std::filesystem::path& setup_directory)
		{
			if (std::filesystem::exists(setup_directory)) {
				if (!std::filesystem::is_directory(setup_directory))
					throw std::runtime_error("Setup path exists but is not a directory: " + setup_directory.string());

				if (!std::filesystem::is_empty(setup_directory))
					throw std::runtime_error("Setup directory is not empty: " + setup_directory.string());
			}
			else {
				if (!std::filesystem::create_directories(setup_directory))
					throw std::runtime_error("Could not create setup directory: " + setup_directory.string());
			}

			const std::filesystem::path xml_path = setup_directory / protocol_xml_name;
			const std::filesystem::path binary_path = setup_directory / simulation_binary_name;
			const std::filesystem::path save_path = setup_directory / save_path_name;

			m_save.path = save_path.string();

			createEmptyProtocolXml(xml_path);

			// Reuse the already existing Protocoll serialization methods.
			m_camera.save(xml_path.string());
			m_calibration.save(xml_path.string());
			m_simulation.save(binary_path.string());
			m_save.save(save_path.string());
		}

		static std::vector<std::filesystem::path> findSetupDirectories(const std::filesystem::path& root_directory)
		{
			if (!std::filesystem::is_directory(root_directory))
				throw std::runtime_error("Protocol root is not a directory: " + root_directory.string());

			std::vector<std::filesystem::path> result{};

			for (const auto& entry : std::filesystem::directory_iterator(root_directory)) {
				if (!entry.is_directory())
					continue;

				if (std::filesystem::exists(entry.path() / protocol_xml_name))
					result.push_back(entry.path());
			}

			// Names start with 000_, 001_, ... so lexical sort restores build order.
			std::sort(result.begin(), result.end(),
				[](const std::filesystem::path& lhs, const std::filesystem::path& rhs) {
					return lhs.filename().string() < rhs.filename().string();
				});

			return result;
		}

		bool operator==(const SimulationProtocoll& prot) const {
			if (!(this->m_calibration == prot.m_calibration)) {
				std::cout << "Calibration Data is not equal \n";
				return false;
			}
			if (!(this->m_camera == prot.m_camera)) {
				std::cout << "Camera data is not equal \n";
				return false;
			}
			if (!(this->m_simulation == prot.m_simulation)) {
				std::cout << "Simulation data is not equal \n";
				return false;
			}
			/*if (!(this->m_save == prot.m_save)) {
				std::cout << "Save Data is not equal \n";
				return false;
			}*/
			
			return true;
		}

		static std::vector<SimulationProtocoll> loadAll(const std::filesystem::path& root_directory)
		{
			const std::vector<std::filesystem::path> setup_directories =
				findSetupDirectories(root_directory);

			if (setup_directories.empty())
				throw std::runtime_error("No protocol setup directories found in: " + root_directory.string());

			std::vector<SimulationProtocoll> protocols{};
			protocols.reserve(setup_directories.size());

			for (const auto& directory : setup_directories) {
				SimulationProtocoll protocol{};

				protocol = readProtocoll(directory);

				protocols.push_back(std::move(protocol));
			}

			return protocols;
		}

		static Protocoll::SimulationProtocoll readProtocoll(const std::filesystem::path& path) {
			std::filesystem::path fs{ path };
			if (!std::filesystem::is_directory(fs))
				throw std::invalid_argument("Path must be a directory");

			std::filesystem::path xml{ fs / protocol_xml_name };
			std::filesystem::path binary{ fs / simulation_binary_name };
			std::filesystem::path save{ fs / save_path_name };

			if (!std::filesystem::exists(xml))
				throw std::invalid_argument(".xml does not exist");
			if (!std::filesystem::exists(binary)) {
				throw std::invalid_argument(".bin file does not exist");
			}
			
			SimulationProtocoll protocoll{};
			protocoll.m_camera = CameraData::read(xml.string());
			protocoll.m_calibration = CalibrationData::read(xml.string());
			protocoll.m_simulation = Simulation::read(binary.string());
			protocoll.m_save = SaveProtocoll::read(save.string());
			
			return protocoll;
		}
	};

	
}


#endif //UTILS_HPP
