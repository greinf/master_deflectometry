#include "camera.hpp"
#include "config/CameraConfig.hpp"


namespace {
    void wait_for_enter()
    {
        std::cout << std::endl;
#if defined(_WIN32)
        system("pause");
#endif
    }
}

Camera::Camera(std::size_t preferredIndex) : 
    ICameraBackend{} // construct the base class 
{
    peak::Library::Initialize();

    auto& deviceManager = peak::DeviceManager::Instance();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    try {
        deviceManager.Update();

        if (deviceManager.Devices().empty()) {
            std::cout << "No device found. Exiting program.\n";
            wait_for_enter();
            peak::Library::Close();
            return;
        }
        // If devices do exist a camera config file is created.
        check_for_config(preferredIndex);

        const auto& devs = deviceManager.Devices();
        
        std::size_t selectedDevice = 0;

        if (devs.size() >= 2) {
            // wenn preferredIndex gültig ist: nutze den

            if (preferredIndex < devs.size()) {
                selectedDevice = preferredIndex;
            }
            else {
                // sonst wie bisher: Auswahl per Konsole
                std::cout << "Devices available:\n";
                std::size_t i = 0;
                for (const auto& d : devs) {
                    std::cout << i << ": " << d->ModelName() << " ("
                        << d->ParentInterface()->DisplayName() << "; "
                        << d->ParentInterface()->ParentSystem()->DisplayName() << " v."
                        << d->ParentInterface()->ParentSystem()->Version() << ")\n";
                    ++i;
                }

                std::cout << "\nSelect device index to open [0-" << devs.size() - 1 << "]: ";
                std::cin >> selectedDevice;

                if (std::cin.fail() || selectedDevice >= devs.size()) {
                    std::cout << "Invalid input! Using index 0.\n";
                    std::cin.clear();
                    std::cin.ignore(1000, '\n');
                    selectedDevice = 0;
                }
            }
        }

        
        // Device öffnen
        if (devs.at(selectedDevice)->IsOpenable()) {
            auto dev = devs.at(selectedDevice)->OpenDevice(peak::core::DeviceAccessType::Control);
            m_devices.push_back(dev);
            m_nodeMaps.push_back(dev->RemoteDevice()->NodeMaps().at(0));
            m_activeDeviceIndex = 0;
        }

        // erste NodeMap
        auto nm = m_nodeMaps.at(m_activeDeviceIndex);

        // Exposure-Time
        auto exposure_time =
            std::dynamic_pointer_cast<peak::core::nodes::FloatNode>(nm->FindNode("ExposureTime"));
        
        // Frame Rate
        auto frame_rate =
            std::dynamic_pointer_cast<peak::core::nodes::FloatNode>(nm->FindNode("AcquisitionFrameRateConv"));
        
        // Maximalauflösung
        int64_t w_max = nm->FindNode<peak::core::nodes::IntegerNode>("Width")->Maximum();
        int64_t h_max = nm->FindNode<peak::core::nodes::IntegerNode>("Height")->Maximum();
    
        logging_data(m_activeDeviceIndex);    
    }
    catch (const std::exception& e) {
        std::cout << "EXCEPTION in Camera ctor: " << e.what() << '\n';
    }
}

bool Camera::open(std::size_t prefferedIndex) 
{   
    try {
        setUpAcquisition(prefferedIndex);
    }
    catch (std::exception& e) { 
        std::cout << "EXCEPTION: " << e.what() << '\n';
        return false;
    }
    return true;
}

void Camera::close(std::size_t prefferedIndex)
{
    try {
        dataStream(prefferedIndex)->StopAcquisition();
        nodeMap(prefferedIndex)->FindNode<peak::core::nodes::IntegerNode>("TLParamsLocked")->SetValue(0);
        nodeMap(prefferedIndex)->FindNode<peak::core::nodes::CommandNode>("AcquisitionStop")->Execute();
        //m_dataStreams[prefferedIndex] = nullptr;
        //m_nodeMaps[prefferedIndex] = nullptr;
    }
    //dataStream(prefferedIndex)->
    catch (std::exception& e) { std::cout << "EXCEPTION: " << e.what() << '\n'; }
}



bool Camera::isacquisitionRunning(std::size_t deviceIndex ) {
    auto it = std::find_if(m_camera_data.begin(), m_camera_data.end(),
        [deviceIndex](std::shared_ptr<defl::CameraConfig> a) -> bool {
            return a->active_device_index == deviceIndex;
        });
    if (it == m_camera_data.end()) {
        throw std::runtime_error("Camera index not available");
    }
    else {
        return (*it)->acquisition_mode_active;
    }
}


std::vector<std::shared_ptr<defl::CameraConfig>>::iterator 
    Camera::check_for_config(std::size_t index) {
    auto it = std::find_if(m_camera_data.begin(), m_camera_data.end(),
        [index](std::shared_ptr<defl::CameraConfig> a) -> bool {
            return a->active_device_index == index;
        });
    if (it == m_camera_data.end()) {
        m_camera_data.push_back(std::make_shared<defl::CameraConfig>());
        return std::prev(m_camera_data.end());
    }
    else return it;
}

void Camera::logging_data(std::size_t camera_index)
{
    std::cout << "Loggin camera data \n";
    auto val_iterator = check_for_config(camera_index);
    try {
        (*val_iterator)->pixel_x = static_cast<int>(nodeMap(camera_index) ->
            FindNode<peak::core::nodes::IntegerNode>("Width")->Maximum());
        (*val_iterator)->pixel_y = static_cast<int>(nodeMap(camera_index) ->
            FindNode<peak::core::nodes::IntegerNode>("Height")->Maximum());
        (*val_iterator)->exposure_time = std::dynamic_pointer_cast<peak::core::nodes::FloatNode>(nodeMap(camera_index) ->
            FindNode("ExposureTime"))->Value();
        (*val_iterator)->frame_rate = std::dynamic_pointer_cast<peak::core::nodes::FloatNode>(nodeMap(camera_index) ->
            FindNode("AcquisitionFrameRateConv"))->Value();
        (*val_iterator)->gain = std::dynamic_pointer_cast<peak::core::nodes::FloatNode>(nodeMap(camera_index) ->
            FindNode("Gain"))->Value();
        if (dataStream(camera_index) != nullptr) (*val_iterator)->acquisition_mode_active = dataStream(camera_index)->IsGrabbing();
        (*val_iterator)->active_device_index = camera_index;
        (*val_iterator)->model_name = device(camera_index)->ModelName();
        (*val_iterator)->version = device(camera_index)->ParentInterface()->ParentSystem()->Version();
        (*val_iterator)->parent_system = device(camera_index)->ParentInterface()->ParentSystem()->DisplayName();
        if (dataStream(camera_index)) (*val_iterator)->acquisition_mode_active =
            dataStream(camera_index)->IsGrabbing();
        else (*val_iterator)->acquisition_mode_active = false;

        auto gainSelector =
            std::dynamic_pointer_cast<peak::core::nodes::EnumerationNode>(nodeMap(camera_index)->FindNode("GainSelector"));

        auto gain =
            std::dynamic_pointer_cast<peak::core::nodes::FloatNode>(nodeMap(camera_index)->FindNode("Gain"));
        gainSelector->SetCurrentEntry("Blue");
        (*val_iterator)->gainB = gain->Value();
        gainSelector->SetCurrentEntry("Green");
        (*val_iterator)->gainG = gain->Value();
        gainSelector->SetCurrentEntry("Red");
        (*val_iterator)->gainR = gain->Value();
        (*val_iterator)->pixel_format = nodeMap(camera_index)->
            FindNode<peak::core::nodes::EnumerationNode>("PixelFormat")->
            CurrentEntry()->Name();
    }
    catch (std::exception& e) { std::cout << "EXCEPTION: " << e.what() << '\n'; }

    // defl::CameraConfig has overloaded << operator
    std::cout << *(*val_iterator);
}


Camera::~Camera() {

    for (auto& ds : m_dataStreams) {
        if (!ds) continue;
        else if (!ds->IsGrabbing()) continue;
        try {
            ds->StopAcquisition();
        }
        catch (...) {
            std::cout << "Exception while stopping acquisition in ~Camera\n";
        }
    }

    peak::Library::Close();
}

cv::Mat Camera::grab(int timeout) {
    CV_Assert(isRunning());
    CV_Assert(timeout > 0);
    auto ds = m_dataStreams[0];
    auto buffer = ds->WaitForFinishedBuffer(timeout);
    cv::Mat dummy;
    if (!buffer) return dummy;
    else {
        cv::Mat view(static_cast<int>(buffer->Height()), static_cast<int>(buffer->Width()),
            CV_8UC1, (void*)buffer->BasePtr(), static_cast<int>(buffer->Width()));
        cv::Mat gray;

        cv::rotate(view, gray, cv::ROTATE_180);

        cv::cvtColor(gray, gray, cv::COLOR_BayerRG2GRAY);
        ds->QueueBuffer(buffer);
        return gray;
        
    }
}


bool Camera::adjustSettings(std::size_t i) {
    if (i >= m_nodeMaps.size()) return false;
    auto nm = m_nodeMaps.at(i);

    std::cout << "\n=== Adjust Settings: Exposure/Gain Analysis ===\n";

    using namespace peak::core::nodes;


    // Small functionality to search for nodes 
    for (const auto& node : nm->Nodes()) {
        const std::string name = node->Name();

        // case-insensitive search
        auto contains = [&](const std::string& key) {
            return std::search(
                name.begin(), name.end(),
                key.begin(), key.end(),
                [](char a, char b) {
                    return std::tolower(a) == std::tolower(b);
                }
            ) != name.end();
            };

        if (contains("Acquisition")) { // || contains("gain") 
            std::cout
                << name
                << ", readable=" << node->IsReadable()
                << "]\n";
        }
    }

    auto gainSelector =
        std::dynamic_pointer_cast<peak::core::nodes::EnumerationNode>(nm->FindNode("GainSelector"));

    // Gain
    auto gain =
        std::dynamic_pointer_cast<peak::core::nodes::FloatNode>(nm->FindNode("Gain"));

    if (!gainSelector || !gain) {
        std::cerr << "GainSelector or Gain node not found \n";
        return false;
    }

    if (gain && gainSelector) {
        try {
            std::array<std::string, (std::size_t)3> color{ "Red", "Green", "Blue" };
            std::array<double, (std::size_t)3> gain_val{ 1.198660134060, 1.0, 1.5666735106970 }; //   FH Dispaly Gain (03.01.2025) values 1.30462, 1.0, 1.50483 
            // BMZ Desktop 1.198660134060, 1.0, 1.5666735106970
            for (std::size_t i = 0; i < color.size(); ++i) {
                gainSelector->SetCurrentEntry(color[i]);
                gain->SetValue(gain_val[i]);
                std::cout << "Set " << color[i] << " gain to: " << gain_val[i] << '\n';;
            }

            for (std::size_t i = 0; i < color.size(); ++i) {
                gainSelector->SetCurrentEntry(color[i]);
                std::cout << color[i] << ": " << gain->Value() << '\n';
            }
        }
        catch (const std::exception& e) {
            std::cout << "EXCEPTION (Gain): " << e.what() << '\n';
            return false;
        }
    }

    

    if (gainSelector) {
        for (const auto& entry : gainSelector->AvailableEntries()) {
            std::cout << entry->DisplayName() << '\n';
        }
    }

    auto pixelFormatNode =
        std::dynamic_pointer_cast<EnumerationNode>(nm->FindNode("PixelFormat"));
    
    if (pixelFormatNode) {
        std::cout << "PixelFormat name Found: \n";
        for (const auto& entry : pixelFormatNode->AvailableEntries()) {
            std::cout << "Pixel Format " << entry->Name() << '\n';
        }
    }
    else std::cout << "PixelFormat Node not available \n";




    // ExposureMode
    auto exposure_mode =
        std::dynamic_pointer_cast<EnumerationNode>(nm->FindNode("ExposureMode"));

    if (!exposure_mode) {
        std::cerr << "Unable to cast ExposureMode node to EnumerationNode\n";
        return false;
    }

    std::cout << "Exposure Modes:\n";
    for (const auto& entry : exposure_mode->AvailableEntries()) {
        std::cout << "  " << entry->DisplayName() << '\n';
    }

    try {
        exposure_mode->SetCurrentEntry("Timed");
    }
    catch (const std::exception& e) {
        std::cout << "EXCEPTION (ExposureMode): " << e.what() << '\n';
        return false;
    }

    // Frame Rate
    auto frame_rate =
        std::dynamic_pointer_cast<FloatNode>(nm->FindNode("AcquisitionFrameRateConv"));
    if (frame_rate) {
        try {
            frame_rate->SetValue(4.5);
            std::cout << "Current Frame Rate " << frame_rate->Value() << '\n';
        }
        catch (const std::exception& e) {
            std::cout << "EXCEPTION (FrameRate): " << e.what() << '\n';
            return false;
        }
    }

    // Exposure Time
    auto exposure_time =
        std::dynamic_pointer_cast<FloatNode>(nm->FindNode("ExposureTime"));
    if (exposure_time) {
        try {
            exposure_time->SetValue(exposure_time->Maximum());
            std::cout << "Current exposure Time " << exposure_time->Value() << '\n';
        }
        catch (const std::exception& e) {
            std::cout << "EXCEPTION (ExposureTime): " << e.what() << '\n';
            return false;
        }
    }

    

    std::cout << "wait\n";
    return true;
}

bool Camera::PrepareAcquisition(std::size_t i) {
    try {
        if (i > m_devices.size()) return false;
        auto dataStreams = m_devices.at(i)->DataStreams();
        if (dataStreams.empty()) return false;

        auto ds = dataStreams.at(i)->OpenDataStream();
        m_dataStreams.push_back(ds);
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "EXCEPTION in PrepareAcquisition: " << e.what() << '\n';
    }
    return false;
}

bool Camera::SetRoi(int64_t x, int64_t y, int64_t width, int64_t height, std::size_t i)
{
    try {
        if (i >= m_nodeMaps.size()) return false;
        auto nm = m_nodeMaps.at(i);

        auto nodeOffsetX = nm->FindNode<peak::core::nodes::IntegerNode>("OffsetX");
        auto nodeOffsetY = nm->FindNode<peak::core::nodes::IntegerNode>("OffsetY");
        auto nodeWidth = nm->FindNode<peak::core::nodes::IntegerNode>("Width");
        auto nodeHeight = nm->FindNode<peak::core::nodes::IntegerNode>("Height");

        int64_t x_min = nodeOffsetX->Minimum();
        int64_t y_min = nodeOffsetY->Minimum();
        int64_t w_min = nodeWidth->Minimum();
        int64_t h_min = nodeHeight->Minimum();

        nodeOffsetX->SetValue(x_min);
        nodeOffsetY->SetValue(y_min);
        nodeWidth->SetValue(w_min);
        nodeHeight->SetValue(h_min);

        int64_t x_max = nodeOffsetX->Maximum();
        int64_t y_max = nodeOffsetY->Maximum();
        int64_t w_max = nodeWidth->Maximum();
        int64_t h_max = nodeHeight->Maximum();

        if ((x < x_min) || (y < y_min) ||
            (x > x_max) || (y > y_max))
        {
            return false;
        }
        if ((width < w_min) || (height < h_min) ||
            ((x + width) > w_max) || ((y + height) > h_max))
        {
            return false;
        }

        nodeOffsetX->SetValue(x);
        nodeOffsetY->SetValue(y);
        nodeWidth->SetValue(width);
        nodeHeight->SetValue(height);

        return true;
    }
    catch (const std::exception& e) {
        std::cout << "EXCEPTION in SetRoi: " << e.what() << '\n';
    }

    return false;
}

bool Camera::AllocAndAnnounceBuffers(std::size_t i) {
    try {
        if (i >= m_dataStreams.size()) return false;
        auto ds = m_dataStreams.at(i);
        if (!ds) return false;

        ds->Flush(peak::core::DataStreamFlushMode::DiscardAll);

        for (const auto& buffer : ds->AnnouncedBuffers())
            ds->RevokeBuffer(buffer);

        auto nm = m_nodeMaps.at(i);
        int64_t payloadSize =
            nm->FindNode<peak::core::nodes::IntegerNode>("PayloadSize")->Value();

        int numBuffersMinRequired = 
            static_cast<int>(ds->NumBuffersAnnouncedMinRequired());

        for (int n = 0; n < numBuffersMinRequired; ++n) {
            auto buffer = ds->AllocAndAnnounceBuffer(static_cast<size_t>(payloadSize), nullptr);
            ds->QueueBuffer(buffer);
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cout << "EXCEPTION in AllocAndAnnounceBuffers: " << e.what() << '\n';
    }

    return false;
}

bool Camera::StartAcquisition(std::size_t i) {
    try {
        if (i >= m_dataStreams.size()) return false;
        auto ds = m_dataStreams.at(i);
        auto nm = m_nodeMaps.at(i);

        ds->StartAcquisition(peak::core::AcquisitionStartMode::Default,
            peak::core::DataStream::INFINITE_NUMBER);
        nm->FindNode<peak::core::nodes::IntegerNode>("TLParamsLocked")->SetValue(1);
        nm->FindNode<peak::core::nodes::CommandNode>("AcquisitionStart")->Execute();

        return true;
    }
    catch (const std::exception& e) {
        std::cout << "EXCEPTION in StartAcquisition: " << e.what() << '\n';
    }

    return false;
}

void Camera::setUpAcquisition(std::size_t i) {
    if (!adjustSettings(i)) {
        std::cerr << "Exposure is not set!\n";
    }
    if (!PrepareAcquisition(i)) {
        std::cerr << "PrepareAcquisition failed!\n";
    }
    if (!SetRoi(0, 0, 1280, 1024, i)) {
        std::cerr << "Setting ROI failed!\n";
    }
    if (!AllocAndAnnounceBuffers(i)) {
        std::cerr << "Buffer allocation failed!\n";
    }
    if (!StartAcquisition(i)) {
        std::cerr << "Acquisition start failed!\n";
    }
    logging_data(i);
}

// Getter
std::shared_ptr<peak::core::DataStream> Camera::dataStream(std::size_t i) const {
    if (i >= m_dataStreams.size()) return { nullptr };
    return m_dataStreams.at(i);
}

std::shared_ptr<peak::core::NodeMap> Camera::nodeMap(std::size_t i) const {
    if (i >= m_nodeMaps.size()) return { nullptr };
    return m_nodeMaps.at(i);
}

std::shared_ptr<peak::core::Device> Camera::device(std::size_t i) const {
    if (i >= m_devices.size()) return { nullptr };
    return m_devices.at(i);
}

