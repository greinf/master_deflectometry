#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <Eigen/dense>
#include <filesystem>
#include <fstream>


struct RayStructure {
	const Eigen::Vector3d* dir;
	const Eigen::Vector3d* origin;
};

struct SamplingSetting {
	std::size_t samples_x{ 1 };
	std::size_t samples_y{ 1 };
};

namespace Protocoll {

	class CameraData {
	public:
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
				double NearPoint{};
				double FarPoint{};
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

				return { nearPoint, farPoint };
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
		bool read(const std::string& path) {
			cv::FileStorage fs{};

			if (!fs.open(path, cv::FileStorage::READ)) return false;

			cv::FileNode node = fs[std::string(m_keys[0])];
			if (node.empty()) return false;
			if (!node.isMap()) return false;

			cv::Mat distortion{}, intrinsic{};
			std::string name{};

			node[std::string(m_keys[1])] >> name;

			// We Just give the pointer if the name fits 
			for (const auto& obj : m_objectiveData) {
				if (obj->name == name) {
					objective = obj;
					break;
				}
			}

			node[std::string(m_keys[6])] >> distortion;
			node[std::string(m_keys[7])] >> intrinsic;
			node[std::string(m_keys[8])] >> used_f_num;

			const Eigen::Index distortion_sz{ static_cast<Eigen::Index>(distortion.total()) };

			const Eigen::Index intrinisc_row{ static_cast<Eigen::Index>(intrinsic.rows) };
			const Eigen::Index intrinsic_col{ static_cast<Eigen::Index>(intrinsic.cols) };

			if (intrinisc_row * intrinsic_col != 9) {
				std::cout << "Given Intrinsic Matrix seems to have wrong size \n" <<
					intrinisc_row * intrinsic_col << " Coefficients found";
				return false;
			}

			if (distortion_sz > 7) {
				std::cout << "Distortion Coefficient Vector has more elements than expected \n" <<
					distortion_sz << " Elements found \n";
				return false;
			}

			distortionCoefficients.resize(distortion_sz, 1);

			intrinsicMatrix.resize(intrinisc_row, intrinsic_col);

			for (Eigen::Index index = 0; index < static_cast<Eigen::Index>(distortion.total()); ++index)
			{
				distortionCoefficients(index, 0) = distortion.at<double>(index, 0);
			}

			for (Eigen::Index row = 0; row < static_cast<Eigen::Index>(intrinsic.rows); ++row) {
				for (Eigen::Index col = 0; col < static_cast<Eigen::Index>(intrinsic.cols); ++col) {
					intrinsicMatrix(row, col) = intrinsic.at<double>(row, col);
				}
			}

			fs.release();

			return true;
		}

		void save(const std::string& path) const {
			cv::FileStorage file{};

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
			file << "}";

			file.release();

			return;
		}

		// Members
		const ObjectiveData* objective{ nullptr };
		ObjectiveData::FocusPoints focus_points{};
		double used_f_num{};
		Eigen::Matrix3d intrinsicMatrix{};
		Eigen::VectorXd distortionCoefficients{};

	private:
		static constexpr std::array<std::string_view, 9> m_keys{
		std::string_view{"Camera_Data"},
		std::string_view{"Objective_Name"},
		std::string_view{"EnrancePupileDiameter"},
		std::string_view{"ExitPupileDiameter"},
		std::string_view{"Min_F_Number"},
		std::string_view{"Max_F_Number"},
		std::string_view{"Distortion_Coefficient"},
		std::string_view{"Camera_Matrix"},
		std::string_view{"Used_F_Number"}
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
		// Member
		SamplingSetting settings{};
		std::vector<Eigen::MatrixXd> m_simulated_images{};
		

		void save(const std::string& path) const {
			std::ofstream fs{ path, std::ios_base::out | std::ios_base::binary };
			if (!fs.is_open()) throw std::runtime_error("Could not open File");

			for (const auto& img : m_simulated_images) {
				fs << "BinaryImage" << '\n';
				fs << img.rows() << " " << img.cols() << " " << "double" << '\n';

				std::streamsize data_size{
					static_cast<std::streamsize>(sizeof(std::decay_t<decltype(img)>::Scalar)) *
					static_cast<std::streamsize>(img.size()) };

				fs.write(reinterpret_cast<const char*>(img.data()), data_size);

				fs << "\n\n";
			}
			fs.close();
		}

		bool read(const std::string& path) {
			std::ifstream fs{ path, std::ios_base::in | std::ios_base::binary };

			if (!fs.is_open()) return false;

			if (!m_simulated_images.empty()) {
				std::cout << "Already Contains images \n";
				return false;
			}

			std::vector<Eigen::MatrixXd> loaded{};

			while (true) {
				int rows, cols;
				std::string datatype, type;
				char newline;

				fs >> type;

				if (fs.eof()) break;

				if (type != std::string("BinaryImage")) {
					std::cerr << "Wrong datatype !!!\n";
					return false;
				}

				fs >> rows >> cols >> datatype;

				if (type != std::string("double")) {
					std::cout << "WARNING: The binary is not marked as double. Expect broken data \n";
				}

				// extract the newline after type
				fs.get(newline);

				std::streamsize data_sz{ rows * cols * sizeof(double) };

				Eigen::MatrixXd temp{};

				temp.resize(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));

				loaded.push_back(std::move(temp));

				fs.read(reinterpret_cast<char*>(loaded.back().data()), data_sz);
			}

			m_simulated_images = std::move(loaded);
			return true;
		}
	};


	class CalibrationData {
	public:
		// Member
		static constexpr std::array<std::string_view, 5> m_keys{
			std::string_view{"CalibrationBoard"},
			std::string_view{"pattern_x"},
			std::string_view{"pattern_y"},
			std::string_view{"pattern_width"},
			std::string_view{"translation_matrices"}
		};

		std::uint16_t pattern_x{};
		std::uint16_t pattern_y{};
		double pattern_width{};
		std::vector<Eigen::Matrix4d> translationMatrix{};

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

		bool read(const std::string& path) {
			cv::FileStorage fs{};

			if (!fs.open(path, cv::FileStorage::READ)) return false;
			cv::FileNode node = fs[std::string(m_keys[0])];
			if (node.empty()) return false;
			if (!node.isMap()) return false;

			node[std::string(m_keys[1])] >> pattern_x;
			node[std::string(m_keys[2])] >> pattern_y;
			node[std::string(m_keys[3])] >> pattern_width;

			if (!node[std::string(m_keys[4])].isSeq()) return false;

			cv::FileNodeIterator start = node[std::string(m_keys[4])].begin();
			cv::FileNodeIterator end = node[std::string(m_keys[4])].end();

			std::size_t n_elements = start.remaining();

			if (n_elements == 0) {
				std::cerr << "Sequence of translation Matrix is empty \n";
				return false;
			}

			translationMatrix.reserve(n_elements);

			for (auto it = start; it != end; ++it) {
				cv::Mat currentTranslation(4, 4, CV_64F);
				*it >> currentTranslation;

				Eigen::Matrix4d matrix{};

				for (std::size_t row = 0; row < currentTranslation.rows; ++row) {
					for (std::size_t col = 0; col < currentTranslation.cols; ++col) {
						matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
							currentTranslation.at<double>(row, col);

						translationMatrix.push_back(std::move(matrix));
					}
				}

				
			}
		}
	};

	// Controll Class 
	class SimulationProtocoll {
		//SimulationProtocoll()

	private:
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

		static constexpr std::string_view m_xml_name{ "Protocoll.xml" };
		static constexpr std::string_view m_binary_name{ "Binary.bin" };
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

		void save(const std::string& save_path) const {

			const std::string unique_name{ getDate() };

			const std::string fullpath(save_path + "/" + unique_name);

			std::filesystem::path active{};

			// Save_path contains path to empty directory
			if (createDirectories(fullpath))
				active = std::filesystem::path(fullpath);

			else {
				std::cout << "Creation of directory failed \n" <<
					"Try Fallback path\n";
				active = { m_fallBack };
			}

			if (!std::filesystem::is_directory(active))
				throw std::runtime_error("Path does not point to directory \n");

			// The Fallback is a last resort 
			// Therefore the fallback folder should be at all times empty 
			if (!std::filesystem::is_empty(active))
				throw std::runtime_error("Given Path does contain data \n");

			std::filesystem::path directory_xml{ active / m_xml_name };

			std::filesystem::path binary_data{ active / m_binary_name };

			cv::FileStorage storage{};
			if (!storage.open(directory_xml.string(), cv::FileStorage::WRITE)) {
				std::cerr << "Saving failed sry \n";
				return;
			}

			storage.release();

			m_camera.save(directory_xml.string());
			m_calibration.save(directory_xml.string());
			m_simulation.save(binary_data.string());
		}

		bool read(const std::string& path) {
			std::filesystem::path fs{ path };
			if (!std::filesystem::is_directory(fs)) return false;

			std::filesystem::path xml{ fs / std::string(m_xml_name) };
			std::filesystem::path binary{ fs / std::string(m_binary_name) };

			if (!std::filesystem::exists(xml)) 
				std::cout << ".xml file at " << xml.string() << " does not exist \n";
			else {
				if (!m_camera.read(xml.string())) 
					std::cout << "Readout of Camera Data Failed \n";
				if (!m_calibration.read(xml.string()))
					std::cout << "Readout of Calibration Data Failed \n";
			}
			
			if (!std::filesystem::exists(binary)) {
				std::cout << ".bin file at " << binary.string() << " does not exist \n";
			}
			else {
				m_simulation.read(binary.string());
			}
		}
	};
}



#endif //UTILS_HPP