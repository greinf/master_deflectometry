#ifndef PROTOCOLLBATCHIO_HPP
#define PROTOCOLLBATCHIO_HPP

#include "Utils.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Protocoll {

    namespace BatchIO {

        namespace fs = std::filesystem;

        inline constexpr const char* protocol_xml_name = "Protocoll.xml";
        inline constexpr const char* simulation_binary_name = "Binary.bin";

        // Checks if valid charakters are used for saving
        inline std::string sanitizePathComponent(std::string value)
        {
            for (char& c : value) {
                const unsigned char uc = static_cast<unsigned char>(c);
                if (!(std::isalnum(uc) || c == '-' || c == '_'))
                    c = '_';
            }

            if (value.empty())
                value = "UnknownObjective";

            return value;
        }

        //inline bool readAll(const fs::path& )

        inline std::string makeSetupDirectoryName(
            const SimulationProtocoll& protocol,
            const std::size_t setup_index,
            const std::size_t distortion_index)
        {
            const std::string objective_name =
                protocol.m_camera.objective != nullptr
                ? sanitizePathComponent(protocol.m_camera.objective->name)
                : std::string{ "UnknownObjective" };

            std::ostringstream ss;
            ss << std::setw(3) << std::setfill('0') << setup_index
                << '_' << objective_name
                << "_dist_" << std::setw(2) << std::setfill('0') << distortion_index;

            return ss.str();
        }

        // Saves the complete SyntheticCalibration output.
        // The root directory must be new or empty; existing data are never overwritten silently.
        inline void saveAll(
            std::vector<SimulationProtocoll>& protocols,
            const fs::path& root_directory)
        {
            if (protocols.empty())
                throw std::invalid_argument("No protocols supplied to BatchIO::saveAll");

            if (fs::exists(root_directory)) {
                if (!fs::is_directory(root_directory))
                    throw std::runtime_error("Batch path exists but is not a directory: " + root_directory.string());

                if (!fs::is_empty(root_directory))
                    throw std::runtime_error("Batch directory is not empty: " + root_directory.string());
            }
            else {
                if (!fs::create_directories(root_directory))
                    throw std::runtime_error("Could not create batch directory: " + root_directory.string());
            }

            // build() stores objective as outer loop and distortion as inner loop.
            // Keeping a per-objective counter gives readable folder names while the
            // global three-digit prefix preserves exact setup order.
            std::unordered_map<std::string, std::size_t> distortion_counter{};

            for (std::size_t setup_index = 0; setup_index < protocols.size(); ++setup_index) {
                auto& protocol = protocols[setup_index];

                const std::string objective_name =
                    protocol.m_camera.objective != nullptr
                    ? protocol.m_camera.objective->name
                    : std::string{ "UnknownObjective" };

                const std::size_t distortion_index = distortion_counter[objective_name]++;

                const fs::path setup_directory =
                    root_directory /
                    makeSetupDirectoryName(protocol, setup_index, distortion_index);

                protocol.saveProtocol(setup_directory);
            }
        }

        inline std::vector<fs::path> findSetupDirectories(const fs::path& root_directory)
        {
            if (!fs::is_directory(root_directory))
                throw std::runtime_error("Protocol root is not a directory: " + root_directory.string());

            std::vector<fs::path> result{};

            for (const auto& entry : fs::directory_iterator(root_directory)) {
                if (!entry.is_directory())
                    continue;

                if (fs::exists(entry.path() / protocol_xml_name))
                    result.push_back(entry.path());
            }
            
            // Names start with 000_, 001_, ... so lexical sort restores build order.
            std::sort(result.begin(), result.end(),
                [](const fs::path& lhs, const fs::path& rhs) {
                    return lhs.filename().string() < rhs.filename().string();
                });

            return result;
        }

        inline std::vector<SimulationProtocoll> loadAll(const fs::path& root_directory)
        {
            const std::vector<fs::path> setup_directories =
                findSetupDirectories(root_directory);

            if (setup_directories.empty())
                throw std::runtime_error("No protocol setup directories found in: " + root_directory.string());

            std::vector<SimulationProtocoll> protocols{};
            protocols.reserve(setup_directories.size());

            for (const auto& directory : setup_directories) {

                SimulationProtocoll protocol = SimulationProtocoll::readProtocoll(directory);
                // std::cout << protocol.m_save.path << std::endl;
                protocols.push_back(std::move(protocol));
            }

            return protocols;
        }

        // After rendering, write only Binary.bin back into the already existing batch.
        // This leaves camera data, distortion, board geometry and poses untouched.
        inline void saveSimulationResults(
            const std::vector<SimulationProtocoll>& protocols,
            const fs::path& root_directory)
        {
            const std::vector<fs::path> setup_directories =
                findSetupDirectories(root_directory);

            if (setup_directories.size() != protocols.size())
                throw std::runtime_error("Protocol count does not match setup-directory count");

            for (std::size_t i = 0; i < protocols.size(); ++i) {
                const fs::path binary_path =
                    setup_directories[i] / simulation_binary_name;

                protocols[i].m_simulation.save(binary_path.string());
            }
        }

        // Generic bridge to the raytracer. The callback receives the full setup,
        // its board pose and both indices. No raytracer-specific dependency is needed here.
        template<class Callable>
        void forEachPose(
            std::vector<SimulationProtocoll>& protocols,
            Callable&& callable)
        {
            for (std::size_t setup_index = 0; setup_index < protocols.size(); ++setup_index) {
                auto& protocol = protocols[setup_index];

                for (std::size_t pose_index = 0;
                    pose_index < protocol.m_calibration.translationMatrix.size();
                    ++pose_index)
                {
                    auto& pose = protocol.m_calibration.translationMatrix[pose_index];
                    callable(protocol, pose, setup_index, pose_index);
                }
            }
        }


        // Convenience variant for the actual raytracer pass. The renderer callable
        // must return one Eigen::MatrixXd image for the supplied setup + board pose.
        // Existing images are cleared so each pose produces exactly one image.
        template<class Renderer>
        void renderAll(
            std::vector<SimulationProtocoll>& protocols,
            Renderer& renderer)
        {
            for (std::size_t setup_index = 0; setup_index < protocols.size(); ++setup_index) {
                auto& protocol = protocols[setup_index];
                auto& images = protocol.m_simulation.m_simulated_images;
                images.clear();
                images.reserve(protocol.m_calibration.translationMatrix.size());

                for (std::size_t pose_index = 0;
                    pose_index < protocol.m_calibration.translationMatrix.size();
                    ++pose_index)
                {
                    renderer(protocol, setup_index, pose_index);
                }
            }
        }

    } // namespace BatchIO
} // namespace Protocoll

#endif // PROTOCOLLBATCHIO_HPP
