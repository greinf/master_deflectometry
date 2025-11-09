#pragma once
#include "camera.hpp"
#include "flagHandler.hpp"

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
                //Camera running Flag is set here. Even if multiple camera are found. 
                runtime_flags.set_start_camera_running_flag();
                
            }
        }
        else if (deviceManager.Devices().size() == 1) {
            m_device.push_back(deviceManager.Devices().at(0)->OpenDevice(peak::core::DeviceAccessType::Control));
            m_nodemapRemoteDevice.push_back(m_device.at(0)->RemoteDevice()->NodeMaps().at(0));
            runtime_flags.set_start_camera_running_flag();
            //Camera running Flag is set here.
        }
        
        std::shared_ptr<peak::core::nodes::FloatNode> exposure_time =
            std::dynamic_pointer_cast<peak::core::nodes::FloatNode>(m_nodemapRemoteDevice.at(0)->FindNode("ExposureTime"));
        try {
            runtime_flags.camera_data.exposure_time = exposure_time->Value();
        }
        catch (std::exception& e) { std::cout << "EXCEPTION " << e.what() << std::endl;}

        std::cout << "wait";
        //return true;
        //Frame Rate
        std::shared_ptr<peak::core::nodes::FloatNode> frame_rate =
            std::dynamic_pointer_cast<peak::core::nodes::FloatNode>(m_nodemapRemoteDevice.at(0)->FindNode("AcquisitionFrameRateConv"));
        try {
            runtime_flags.camera_data.frame_rate = frame_rate->Value();
        }
        catch (std::exception& e) { std::cout << "EXCEPTION " << e.what() << std::endl; }

        int64_t w_max = m_nodemapRemoteDevice.at(0)->FindNode<peak::core::nodes::IntegerNode>("Width")->Maximum();
        int64_t h_max = m_nodemapRemoteDevice.at(0)->FindNode<peak::core::nodes::IntegerNode>("Height")->Maximum();
        // Set pixel Value for 
        runtime_flags.camera_data.pixel_x = w_max;
        runtime_flags.camera_data.pixel_y = h_max;

    }
    catch (const std::exception& e)
    {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        //Sets stop camera running flag
        runtime_flags.set_stop_camera_running_flag();
    }
}

Camera::~Camera() {
    runtime_flags.set_stop_camera_running_flag();
    for (auto& ds : m_dataStream) {
        try { ds->StopAcquisition(); }
        catch (...) { std::cout << "Camera Object destroyed " << std::endl; }
    }
    peak::Library::Close();
}

void Camera::grayValueCalibration(std::shared_ptr<Screen> screen) {

}


/*
//Signature

template<class T, class U>
std::shared_ptr<T> dynamic_pointer_cast(const std::shared_ptr<U>& r);

It checks at runtime whether the object that r points to is (polymorphically) a T.
-> If yes: returns a new shared_ptr<T> pointing to the same object (shares ownership, control block).
-> If no: returns an empty shared_ptr<T> (equivalent to nullptr).

*/

bool Camera::adjustSettings(std::size_t i) {
    // 
    auto nm = m_nodemapRemoteDevice.at(i);
    
    std::cout << "\n=== Adjust Settings: Exposure/Gain Analysis ===\n";

    
    //Exposure Mode
    std::shared_ptr<peak::core::nodes::EnumerationNode> exposure_mode = 
        std::dynamic_pointer_cast<peak::core::nodes::EnumerationNode>(nm->FindNode("ExposureMode"));
    
    if (!exposure_mode) { std::cerr << "Unable to upcast pointer to enumerationMode! \n"; }
    std::cout << "Esposure Modes: \n";
    for (const auto entry : exposure_mode->AvailableEntries()) {
        std::cout << entry->DisplayName() << '\n';
    }
    //This lines sets the Value of ExposureMode fix to timed. Even so it seems it is the only available entry
    try {
        exposure_mode->SetCurrentEntry("Timed");
    }
    catch (std::exception& e) { std::cout << "EXCEPTION " << e.what() << std::endl; return false; }

    //Exposure Time
    std::shared_ptr<peak::core::nodes::FloatNode> exposure_time =
        std::dynamic_pointer_cast<peak::core::nodes::FloatNode>(nm->FindNode("ExposureTime"));
    try {
        exposure_time->SetValue(exposure_time->Minimum() * 7000);
        std::cout << "Current exposure Time " << exposure_time->Value() << '\n';
        runtime_flags.camera_data.exposure_time = exposure_time->Value();
    }
    catch (std::exception& e) { std::cout << "EXCEPTION " << e.what() << std::endl; return false; }

    std::cout << "wait";
    //return true;
    //Frame Rate
    std::shared_ptr<peak::core::nodes::FloatNode> frame_rate = 
        std::dynamic_pointer_cast<peak::core::nodes::FloatNode>(nm->FindNode("AcquisitionFrameRateConv"));
    try {
        frame_rate->SetValue(frame_rate->Maximum()); //frame_rate->Maximum()
        std::cout << "Current Frame Rate " << frame_rate->Value() << '\n';
        runtime_flags.camera_data.frame_rate = frame_rate->Value();
    }

    catch (std::exception& e) { std::cout << "EXCEPTION " << e.what() << std::endl; return false; }

    return true;
    // This is for debuggin -> search node for given category with name.find()
    // To use it comment out -> return true above
    // after this the nodes (base class) try to get downcased to their given derivatives
    // this is possible because they´re shared_ptr and the class contais virtual functions
    // and with that a virtual table where one can look if the given ptr really is the derivative
    for (auto& node : nm->Nodes()) {
        const std::string name = node->Name();

        if (name.find("Exposure") != std::string::npos ||
            name.find("Gain") != std::string::npos ||
            name.find("Brightness") != std::string::npos ||
            name.find("Frame") != std::string::npos )
        {
            std::cout << name;

            try {
                using namespace peak::core::nodes;


                // Remember peak::core::NodeMap is a shared ptr. 
                // Becasue every node is derived from the base class peak::core::node
                // we can dynamic cast the give node to its derived form. std::dynamic_pointer_cast<>() return false
                // if the downcast was not possible. 
                if (auto n = std::dynamic_pointer_cast<IntegerNode>(node)) {
                    std::cout << " [Integer]"
                        << " value=" << n->Value()
                        << " min=" << n->Minimum()
                        << " max=" << n->Maximum();
                }
                else if (auto n = std::dynamic_pointer_cast<FloatNode>(node)) {
                    std::cout << " [Float]"
                        << " value=" << n->Value()
                        << " min=" << n->Minimum()
                        << " max=" << n->Maximum();
                }
                else if (auto n = std::dynamic_pointer_cast<EnumerationNode>(node)) {
                    std::string current = n->CurrentEntry()
                        ? n->CurrentEntry()->Name()
                        : "<none>";
                    std::cout << " [Enum] current=" << current;
                }
                else if (auto n = std::dynamic_pointer_cast<BooleanNode>(node)) {
                    std::cout << " [Bool] value=" << std::boolalpha << n->Value();
                }
                else {
                    std::cout << " [Unknown type]";
                }
            }
            catch (const std::exception& e) {
                std::cout << " [Access error: " << e.what() << "]";
            }

            std::cout << '\n';
        }
    }

    std::cout << "==============================================\n";
    return true;
    
}


bool Camera::PrepareAcquisition(std::size_t i) {
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

        //  ***** Debugging Settings *****
        /*
        auto availableEntries = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::EnumerationNode>("PixelFormat")->AvailableEntries();
        std::cout << "Num Options: " << availableEntries.size() << '\n';
        for (const auto& entry : availableEntries) {
            std::cout << entry->Name() << std::endl;
        }
        */

        /*
        
        /*
        //available nodes
        for (auto& nodes : m_nodemapRemoteDevice.at(i)->Nodes()) {
            if (nodes->DisplayName().find("") != std::string::npos) {
                std::cout << nodes->DisplayName() << '\n';
            }
        }
        */
        /* 

        Color transforms / sRGB / LUT ? if such nodes exist, disable or leave default (=off).

        BlackLevel ? if present, read/record it and subtract in processing.
        */

        // Get the maximum ROI values
        int64_t x_max = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("OffsetX")->Maximum();
        int64_t y_max = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("OffsetY")->Maximum();
        int64_t w_max = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("Width")->Maximum();
        int64_t h_max = m_nodemapRemoteDevice.at(i)->FindNode<peak::core::nodes::IntegerNode>("Height")->Maximum();
        

        // Set pixel Value for 
        runtime_flags.camera_data.pixel_x = w_max;
        runtime_flags.camera_data.pixel_y = h_max;

        // Check for maximum values
        // std::cout << "Maximum Width: " << w_max << '\n';
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

//Does set up the camera to acquire images. 
//If more than camera is used index gives possability to choose between cameras. 
//Return a worker class for controll over Image Acquisition.
void Camera::setUpAcquisition(std::size_t i) {  
    //Create Buffer

    if (!adjustSettings(i)) {
        std::cerr << "Exposure is not set ! \n";
    }
    if (!PrepareAcquisition(i)) {
        std::cerr << "Prepare Acquisition failed ! \n";
    }
    if (!SetRoi(0, 0, 1280, 1024, i)) //Here input the needed ROI 
    {
        std::cout << "Setting for ROI failed " << std::endl;
    }
    if (!AllocAndAnnounceBuffers(i)) {
        std::cout << "Buffer allocation failed " << std::endl;
    }
    if (!StartAcquisition(i)) {
        std::cout << "Acquisation failed " << std::endl;
    }
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


