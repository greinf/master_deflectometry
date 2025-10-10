#include "camera.hpp"

void wait_for_enter();
void showRawImage(const cv::Mat& mat);

Camera::Camera()
{
    peak::Library::Initialize();

    // create a device manager object
    auto& deviceManager = peak::DeviceManager::Instance();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    //std::cout << "entered Constructor " << std::endl;
    try
    {
        // update the deviceManager
        deviceManager.Update();

        // exit program if no device was found
        if (deviceManager.Devices().empty())
        {
            std::cout << "No device found. Exiting program." << std::endl << std::endl;
            wait_for_enter();
            // close peak library
            peak::Library::Close();
        }
        // deviceManager.Devices() gives Device Descriptors. These are NOT the active devices, more like a first descriptor 
        // of what could be opended. Hyrachie: System -> Interface -> DeviceDescriptor -> Device
        // 1. System ( ParentSystem() ) -> Represent hardware transport layer
        // 2. Interface ( ParentInterface() ) -> Represent a single interface on that system
        // 3. Device Descriptor -> Represent a camera found on that interface, BUT not yet opended. 
        // 4. Device -> The actual open Camera that one gets by calling deviceManager.Devices().at(i)->OpenDevice(...Devices Acces Type...)

        if (deviceManager.Devices().size() >= 2) {
            // list all available devices
            size_t i = 0;
            std::cout << "Devices available: " << std::endl;
            for (const auto& deviceDescriptor : deviceManager.Devices())
            {
                std::cout << i << ": " << deviceDescriptor->ModelName() << " ("
                    << deviceDescriptor->ParentInterface()->DisplayName() << "; "
                    << deviceDescriptor->ParentInterface()->ParentSystem()->DisplayName() << " v."
                    << deviceDescriptor->ParentInterface()->ParentSystem()->Version() << ")" << std::endl;
                ++i;
            }

            // select a device to open
            size_t selectedDevice = 0;

            // select a device to open via user input or remove these lines to always open the first available device
            std::cout << std::endl << "Select device index to open [0-" << deviceManager.Devices().size() - 1 << "]: ";
            std::cin >> selectedDevice;

            if (std::cin.fail())
            {
                std::cout << "Invalid input! Using index 0." << std::endl;
                std::cin.clear();
                std::cin.ignore(1000, '\n');
                selectedDevice = 0;
            }

            if (selectedDevice >= deviceManager.Devices().size())
            {
                std::cout << "Invalid index! Using index 0." << std::endl;
                selectedDevice = 0;
            }

            // open the selected device in member variable
            if (deviceManager.Devices().at(selectedDevice)->IsOpenable()) {
                m_device.push_back(deviceManager.Devices().at(selectedDevice)->OpenDevice(peak::core::DeviceAccessType::Control));
                m_nodemapRemoteDevice.push_back(m_device.at(selectedDevice)->RemoteDevice()->NodeMaps().at(0));

                //auto device = deviceManager.Devices().at(selectedDevice)->OpenDevice(peak::core::DeviceAccessType::Control);
                //auto nodeMapRemoteDevice = device->RemoteDevice()->NodeMaps().at(0);
            }
        }
        else if (deviceManager.Devices().size() == 1) {
            m_device.push_back(deviceManager.Devices().at(0)->OpenDevice(peak::core::DeviceAccessType::Control));
            m_nodemapRemoteDevice.push_back(m_device.at(0)->RemoteDevice()->NodeMaps().at(0));
        }
        
    }
    catch (const std::exception& e)
    {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
    }

}

Camera::~Camera() {
    for (auto& ds : m_dataStream) {
        try { ds->StopAcquisition(); }
        catch (...) { std::cout << "Camera Object destroyed " << std::endl; }
    }
    peak::Library::Close();
}

bool Camera::PrepareAcuqisition(std::size_t i) {
    try {
        auto dataStreams = m_device.at(i)->DataStreams();
        if (dataStreams.empty()) {
            return false;
        }
        m_dataStream.push_back(m_device.at(i)->DataStreams().at(0)->OpenDataStream());
        return true;
    }
    catch (std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
    }
    return false;
}

bool Camera::SetRoi(int64_t x, int64_t y, int64_t width, int64_t height, std::size_t i)
{
    try
    {
        // Get the minimum ROI and set it. After that there are no size restrictions anymore
        int64_t x_min = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("OffsetX")->Minimum();
        int64_t y_min = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("OffsetY")->Minimum();
        int64_t w_min = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("Width")->Minimum();
        int64_t h_min = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("Height")->Minimum();

        //Set the minimal ROI
        m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("OffsetX")->SetValue(x_min);
        m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("OffsetY")->SetValue(y_min);
        m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("Width")->SetValue(w_min);
        m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("Height")->SetValue(h_min);

        //Debugging 
        /*
        auto availableEntries = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::EnumerationNode>("PixelFormat")->AvailableEntries();
        std::cout << "Num Options: " << availableEntries.size() << '\n';
        for (const auto& entry : availableEntries) {
            std::cout << entry->Name() << std::endl;
        }
        */

        // Get the maximum ROI values
        int64_t x_max = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("OffsetX")->Maximum();
        int64_t y_max = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("OffsetY")->Maximum();
        int64_t w_max = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("Width")->Maximum();
        int64_t h_max = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("Height")->Maximum();
        
        //Check for maximum values
        //std::cout << "Maximum Width: " << w_max << '\n';
        //std::cout << "Maximum Hight: " << h_max << '\n';
        //auto pixFmt = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::EnumerationNode>("PixelFormat");
        //pixFmt->SetCurrentEntry(pixFmt->FindEntry("Mono8"));

        if ((x < x_min) || (y < y_min) || (x > x_max) || (y > y_max))
        {
            return false;
        }
        else if ((width < w_min) || (height < h_min) || ((x + width) > w_max) || ((y + height) > h_max))
        {
            return false;
        }
        else
        {
            // Now, set final AOI
            m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("OffsetX")->SetValue(x);
            m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("OffsetY")->SetValue(y);
            m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("Width")->SetValue(width);
            m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("Height")->SetValue(height);
            
            return true;
        }
    }
    catch (std::exception& e)
    {
        std::cout << "EXCEPTION " << e.what() << std::endl;
    }

    return false;
}

bool Camera::AllocAndAnnounceBuffers(std::size_t i) {
    try
    {
        if (m_dataStream.at(i))
        {
            // Flush queue and prepare all buffers for revoking
            m_dataStream.at(i)->Flush(peak::core::DataStreamFlushMode::DiscardAll);

            // Clear all old buffers
            for (const auto& buffer : m_dataStream.at(i)->AnnouncedBuffers())
            {
                m_dataStream.at(i)->RevokeBuffer(buffer);
            }

            int64_t payloadSize = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("PayloadSize")->Value();

            // Get number of minimum required buffers
            int numBuffersMinRequired = m_dataStream.at(i)->NumBuffersAnnouncedMinRequired();

            // Alloc buffers
            for (size_t count = 0; count < numBuffersMinRequired; count++)
            {
                auto buffer = m_dataStream.at(i)->AllocAndAnnounceBuffer(static_cast<size_t>(payloadSize), nullptr);
                m_dataStream.at(i)->QueueBuffer(buffer);
            }

            return true;
        }
    }
    catch (std::exception& e)
    {
        std::cout << "EXCEPTION " << e.what() << std::endl;
    }

    return false;
}

bool Camera::StartAcquisition(std::size_t i){
    try
    {
        m_dataStream.at(i)->StartAcquisition(peak::core::AcquisitionStartMode::Default, peak::core::DataStream::INFINITE_NUMBER);
        m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("TLParamsLocked")->SetValue(1);
        m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::CommandNode>("AcquisitionStart")->Execute();
        
        return true;
    }
    catch (std::exception& e)
    {
        std::cout << "EXCEPTION " << e.what() << std::endl;
    }

    return false;
}

void Camera::getFrames(std::size_t i) {  //Index for camera numeration
    //Create Buffer
    if (!PrepareAcuqisition()) {
        std::cout << "Acquisition Failed " << std::endl;
        return;
    }
    if (!SetRoi(0, 0, 1280, 1024)) //Here input the needed ROI 
    {
        std::cout << "Setting for ROI failed " << std::endl;
    }
    
    if (!AllocAndAnnounceBuffers()) {
        std::cout << "Buffer allocation failed " << std::endl;
    }

    if (!StartAcquisition()) {
        std::cout << "Acquisation failed " << std::endl;
    }

    auto worker = std::make_unique<AcquisitionWorker>(m_dataStream.at(i), m_nodemapRemoteDevice.at(i));
    worker->assignImageHandler(showRawImage);
    worker->start();

    std::cout << "Wait for Results ";
    char a;
    std::cin >> a;
    worker->stop();

}

void wait_for_enter()
{
    std::cout << std::endl;
#if defined(_WIN32)
    system("pause");
#endif
}

//Declare further functionality like saving pictures or ask for user input
// void *function_name*(const cv::Mat& mat){}

void showRawImage(const cv::Mat& mat) {
    cv::imshow("Frame", mat);
    cv::waitKey(10);
}
