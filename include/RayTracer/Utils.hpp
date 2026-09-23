#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
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

			return (valid).select(matrix, 0.0).isZero();
		}
	public:
		CameraData() = default;

		bool validAndSet() const {
			constexpr double eps{ 1e-9 };
			if (objective == nullptr) return false;
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

			const Eigen::Index distortion_sz{ static_cast<Eigen::Index>(distortion.total()) };

			const Eigen::Index intrinisc_row{ static_cast<Eigen::Index>(intrinsic.rows) };
			const Eigen::Index intrinsic_col{ static_cast<Eigen::Index>(intrinsic.cols) };

			if (intrinisc_row * intrinsic_col != 9) {
				std::cout << "Given Intrinsic Matrix seems to have wrong size \n" <<
					intrinisc_row * intrinsic_col << " Coefficients found";
				throw std::logic_error("Intrinsic Matrix can not have more than 9 coefficients");
			}

			if (distortion_sz > 7) {
				std::cout << "Distortion Coefficient Vector has more elements than expected \n" <<
					distortion_sz << " Elements found \n";
				throw std::logic_error("Distortion Vektor can not have more than 7 elements");
			}

			data.distortionCoefficients.resize(distortion_sz, 1);

			data.intrinsicMatrix.resize(intrinisc_row, intrinsic_col);

			for (Eigen::Index index = 0; index < static_cast<Eigen::Index>(distortion.total()); ++index)
			{
				data.distortionCoefficients(index, 0) = distortion.at<double>(index, 0);
			}

			for (Eigen::Index row = 0; row < static_cast<Eigen::Index>(intrinsic.rows); ++row) {
				for (Eigen::Index col = 0; col < static_cast<Eigen::Index>(intrinsic.cols); ++col) {
					data.intrinsicMatrix(row, col) = intrinsic.at<double>(row, col);
				}
			}

			fs.release();

			return data;
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
		Eigen::Matrix3d intrinsicMatrix{};
		Eigen::VectorXd distortionCoefficients{};

		bool operator==(const CameraData& data) const {
			if (!(this->objective == data.objective)) return false;
			if (!(this->focus_points == data.focus_points)) return false;
			if (this->used_f_num != data.used_f_num) return false;
			if (!this->intrinsicMatrix.isApprox(data.intrinsicMatrix, 1e-9)) return false;
			if (!this->distortionCoefficients.isApprox(data.distortionCoefficients, 1e-9)) return false;
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
		bool validAndSet() const {
			if (settings.samples_x == 0 || settings.samples_y == 0) return false;
			if (m_simulated_images.empty()) return false;
			if (m_simulated_images.size() < 1) return false;
			return true;
		}

		std::size_t get_Number_of_Images() const {
			return m_simulated_images.size();
		}
		
		// Member
		SamplingSetting settings{};
		std::vector<Eigen::MatrixXd> m_simulated_images{};

		bool operator==(const Simulation& data) const {
			if (!(this->settings == data.settings)) return false;
			if (this->m_simulated_images.size() != data.m_simulated_images.size()) return false;
			if (std::pair<const std::vector<Eigen::MatrixXd>::const_iterator, const std::vector<Eigen::MatrixXd>::const_iterator>
			{this->m_simulated_images.end(), data.m_simulated_images.end()} !=
				std::mismatch(this->m_simulated_images.begin(),
					this->m_simulated_images.end(),
					data.m_simulated_images.begin(),
					[](const Eigen::MatrixXd& this_img, const Eigen::MatrixXd& data_img) -> bool {
						return this_img.isApprox(data_img, 1e-9);
					})) return false;
			return true;
		}

		void show(const std::size_t i) const {
			const auto img = m_simulated_images.at(i);
			cv::Mat img_cv{};

			cv::eigen2cv(img, img_cv);

			cv::Mat scaledOutput{};

			const auto row = img.rows();
			const auto cols = img.cols();

			int row_out = static_cast<int>(row / settings.samples_x);
			int col_out = static_cast<int>(cols / settings.samples_y);

			cv::GaussianBlur(img_cv, 
				img_cv, 
				cv::Size{static_cast<int>(settings.samples_x + 2), static_cast<int>(settings.samples_y + 2)},
				0.0);

			cv::resize(img_cv, scaledOutput, cv::Size(col_out, row_out), 0.0, 0.0, cv::INTER_AREA);

			cv::normalize(img_cv, img_cv, 0, 255, cv::NORM_MINMAX, CV_8U);
			
			// Erlaubt das freie Skalieren des Fensters per Maus
			cv::namedWindow("Output", cv::WINDOW_NORMAL);

			cv::imshow("Output", img_cv);

			cv::waitKey(0);
		}

		void showAll() const {
			for (std::size_t i = 0; i < m_simulated_images.size(); ++i) {
				show(i);
			}
		}
		
		std::vector<Eigen::MatrixXd> prepareImagesforProcessing(
			const int kernel_bias = 2) const 
		{
			if (kernel_bias < 0) throw std::invalid_argument("Kernel size must be positive");
			if (m_simulated_images.empty()) {
				std::cout << "WARNING: Container is empty";
				return {};
			}
			if (settings.samples_x != settings.samples_y) {
				std::cout << "WARNING: Different Subsampling for x and y direction \n";
				return {};
			}

			const int kernel_size{ static_cast<int>(settings.samples_x) + kernel_bias };

			if (kernel_size == 1) return m_simulated_images;

			std::vector<Eigen::MatrixXd> output{};

			output.reserve(m_simulated_images.size());

			const int cam_rows{ 
				static_cast<int>(m_simulated_images.front().rows()) / 
				static_cast<int>(settings.samples_y) };

			const int cam_cols{ 
				static_cast<int>(m_simulated_images.front().cols()) / 
				static_cast<int>(settings.samples_x) };

			for (const auto& raw : m_simulated_images) {
				cv::Mat raw_cv{};
				cv::eigen2cv(raw, raw_cv);

				cv::Mat smoothed_cv{};
				cv::GaussianBlur(raw_cv, smoothed_cv, cv::Size{ kernel_size, kernel_size }, 0.0);

				cv::Mat downsized{};
				cv::resize(smoothed_cv, downsized, cv::Size{ cam_cols, cam_rows }, cv::INTER_AREA);

				Eigen::MatrixXd out{};
				cv::cv2eigen(downsized, out);

				if (out.rows() != cam_rows || out.cols() != cam_cols) throw std::logic_error("MASSIVE ERROR");
				output.push_back(out);
			}
			
			return output;
		}

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

		static Simulation read(const std::filesystem::path& path) {

			std::ifstream fs{ path, std::ios_base::in | std::ios_base::binary };

			if (!fs.is_open()) throw std::runtime_error("File is already open");

			Simulation data{};

			std::vector<Eigen::MatrixXd> loaded{};

			while (true) {
				int rows, cols;
				std::string datatype, type;
				char newline;

				fs >> type;
			
				if (fs.eof()) break;

				if (type != std::string("BinaryImage")) {
					throw std::logic_error("Wrong Datatype \n");
				}

				fs >> rows >> cols >> datatype;

				if (datatype != std::string("double")) {
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

			data.m_simulated_images = std::move(loaded);
			return data;
		}
	};


	class CalibrationData {
	public:
		// Member
		bool validAndSet() const {
			constexpr double eps = 1e-9;
			if (pattern_x == 0 || pattern_y == 0) return false;
			if (pattern_width < eps) return false;
			if (translationMatrix.empty()) return false;

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

			node[std::string(m_keys[1])] >> data.pattern_x;
			node[std::string(m_keys[2])] >> data.pattern_y;
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

				Eigen::Matrix4d matrix = Eigen::Matrix4d::Zero();

				for (int row = 0; row < currentTranslation.rows; ++row) {
					for (int col = 0; col < currentTranslation.cols; ++col) {
						matrix(row, col) = currentTranslation.at<double>(row, col);
					}
				}

				data.translationMatrix.push_back(std::move(matrix));
			}
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

			file >> path;

			int c{ file.get() };
			
			if (c != '\n')
				throw std::runtime_error("Datatype has a Proceeding Whitspace");

			file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

			SaveProtocoll save{};
			save.path = path;

			return save;
		}
	};


	// Controll Class 
	class SimulationProtocoll {
	public:
		SimulationProtocoll() = default;

		bool validAndSet() const 
		{
			if (m_camera.validAndSet()) return false;
			if (m_calibration.validAndSet()) return false;
			const std::size_t n_trans{ m_calibration.get_number_of_translation() };
			if (m_simulation.validAndSet()) return false;

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
				throw std::invalid_argument("Path must be a container");

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