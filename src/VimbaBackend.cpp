#include "VimbaBackend.hpp"
#include <chrono>

namespace {
    static int counter{};
    void ensureVimbaStarted() {
        if (counter > 1) throw std::runtime_error("Multiple Statups are forbidden");
        auto& sys = VmbCPP::VmbSystem::GetInstance();
        auto err = sys.Startup();
        if (err != VmbErrorSuccess)
            throw std::runtime_error("Vimba Startup failed");
        ++counter;
    }
    void ensureVimbaClosed() {
        if (counter <= 0) {
            std::cout << "Vimba was not started \n";
            return;
        }
        auto& sys = VmbCPP::VmbSystem::GetInstance();
        if (sys.Shutdown() != VmbErrorSuccess) {
            std::cout << "Vimba Shutdown Failed \n";
            return;
        }
        else {
            --counter;
            return;
        }

    }
}

VimbaBackend::VimbaBackend(std::size_t i):
    m_running{ false },
    m_ringBufferPtr{ nullptr } {
    ensureVimbaStarted();
    m_cam_index = i;
}

bool VimbaBackend::openCamera(std::size_t camera_index) {

    VmbCPP::VmbSystem& system = VmbCPP::VmbSystem::GetInstance();
    
    std::vector<VmbCPP::CameraPtr> val_cams;
    
    std::vector<VmbCPP::CameraPtr> cameras;
    if (VmbErrorSuccess == system.GetCameras(cameras))
    {
        for (VmbCPP::CameraPtrVector::iterator iter = cameras.begin();
            cameras.end() != iter;
            ++iter)
        {   
            std::string device_name;
                
            if (VmbErrorSuccess == (*iter)->GetName(device_name)) {
                std::cout << device_name << '\n';
                if(std::string::npos != device_name.find("Simulator", 0)) continue;
                // If the camera is a "Simulator" push into the m_cameras
                else {
                    val_cams.push_back(*iter);
                    std::string deviceID;
                    if (VmbErrorSuccess != (*iter)->GetExtendedID(deviceID)) {
                        std::cout << "Could not Acquire extended deviceID \n";
                        return false; // We need this ExtendedID here
                    }
                    m_availableIds.push_back(deviceID);
                }  
            }
        }
    }
    

    assert((val_cams.size() > camera_index) && "Index out of bounds for available cameras \n");
    assert((m_availableIds.size() > camera_index) && "Index out of bounds for available ID´s \n");

    VmbCPP::CameraPtr cam = val_cams[camera_index];

    // This part Checks before Connecting if we can Access all the function of the camera that we need. 
    VmbAccessModeType access_mode{};
    VmbErrorType errP2 = cam->GetPermittedAccess(access_mode);
    std::cout << "GetPermittedAccess(after open) err=" << errP2
        << " perm=" << (int)access_mode << "\n";

    switch (access_mode) {
    case(VmbAccessModeNone):
        std::cout << "Camera opened in a not readable or writeable state \n ";
        return false;
    case(VmbAccessModeRead):
        std::cout << "Camera only readalbe. Features must be set \n";
        return false;
    case(VmbAccessModeFull):
        std::cout << "Full camera Access. Features can be set \n";
        break;
    case(VmbAccessModeExclusive):
        std::cout << "Exclusive Access to the camera. Should be ok \n";
        break;
    case(VmbAccessModeExclusive + VmbAccessModeRead + VmbAccessModeFull):
        std::cout << "Exclusive Acces to the camera with read and write \n";
        break;
    case(VmbAccessModeUnknown):
        std::cout << "Unknown Acces Mode. We do not wnat unknown \n";
        return false;
    default:
        std::cout << "Unknown Access Mode. \n";
        return false;
    }

    if (cam->Open(VmbAccessModeExclusive) != VmbErrorSuccess) {
        std::cout << "Opening Camera Failed \n";
        return false;
    }
    else {
        cam->GetExtendedID(m_active_cam);
    }

    // one harmless read, no write
    std::string name;
    if (cam->GetName(name) != VmbErrorSuccess) {
        std::cout << "Failed getting the devie name \n";
    }
    else {
        std::cout << "Opened Device " << name << '\n';
    }
    
    // If we reach this point at least a camera should be opened.
    return true;
}

void VimbaBackend::closeCamera(std::size_t camera_index) {
    /*assert((m_cameras.size() > camera_index) && "Index out of bounds for available cameras \n");
    std::string name;
    m_cameras[camera_index]->GetName(name);
    std::cout << "Try to close the camera: " << name << std::endl;
    if (VmbErrorSuccess == m_cameras[camera_index]->Close())
    {
        std::cout << "Camera closed" << std::endl;
    }
    else std::cout << "Closing the camera failed! " << std::endl;*/
}

VimbaBackend::~VimbaBackend() {
    close();
    deleteBuffer();
    ensureVimbaClosed();
}


bool VimbaBackend::getFeature(const VmbCPP::CameraPtr& cam,
    const std::vector<std::string>& names,
    VmbCPP::FeaturePtr& out)
{
    for (const auto& n : names) {
        VmbCPP::FeaturePtr f;
        if (cam->GetFeatureByName(n.c_str(), f) == VmbErrorSuccess && f) {
            out = f;
            return true;
        }
    }
    out.reset();
    return false;
}

bool VimbaBackend::setEnum(const VmbCPP::CameraPtr& cam,
    const std::vector<std::string>& names,
    const char* value)
{
    assert(names.size());
    if (!cam) return false;
    VmbCPP::FeaturePtr f;
    if (!getFeature(cam, names, f))
        std::cout << "We weren´t even able to acquire the feature Vector \n";
    VmbBool_t writable = false, readable = false;
    if (f->IsWritable(writable) != VmbErrorSuccess)
        std::cout << "WARN: Accessing IsWriteable() failed for " << names[0] << std::endl;;
    if (f->IsReadable(readable) != VmbErrorSuccess)
        std::cout << "WARN: Accessing IsReadalbe() failed for " << names[0] << std::endl;

    std::cout << "Feature " << names[0] << ": \n" <<
        "Is Readable: " << readable << '\n' <<
        "Is Writeable: " << writable << '\n';
    // For enums Vimba typically accepts strings like "Off", "Continuous", etc.
    return f->SetValue(value) == VmbErrorSuccess;
}

VmbCPP::CameraPtr VimbaBackend::findCameraByID(const std::size_t i) {
    std::string wantedID = m_availableIds.at(i);
    VmbCPP::CameraPtrVector cameras;
    auto& system = VmbCPP::VmbSystem::GetInstance();

    if (system.GetCameras(cameras) != VmbErrorSuccess)
        return nullptr;

    for (const auto& cam : cameras) {
        std::string id;
        if (cam->GetExtendedID(id) == VmbErrorSuccess) {
            if (id == wantedID) {
                return cam;
            }
        }
    }
    return nullptr;
}


bool VimbaBackend::setFloatClamped(const VmbCPP::CameraPtr& cam,
    const std::vector<std::string>& names,
    double value)
{
    if (!cam) return false;
    VmbCPP::FeaturePtr f;
    if (!getFeature(cam, names, f)) return false;
    VmbBool_t writable = false, readable = false;
    if (f->IsWritable(writable) != VmbErrorSuccess)
        std::cout << "WARN: Accessing IsWriteable() failed for " << names[0] << std::endl;;
    if (f->IsReadable(readable) != VmbErrorSuccess)
        std::cout << "WARN: Accessing IsReadalbe() failed for " << names[0] << std::endl;

    // 2) Only if direct set fails, attempt range clamp
    double minV = 0.0, maxV = 0.0;
    if (f->GetRange(minV, maxV) == VmbErrorSuccess) {
        if (value < minV) value = minV;
        if (value > maxV) value = maxV;
        return f->SetValue(value) == VmbErrorSuccess;
    }
    return false;
}

bool VimbaBackend::setIntClamped(const VmbCPP::CameraPtr& cam,
    const std::vector<std::string>& names,
    VmbInt64_t value)
{
    if (!cam) return false;
    VmbCPP::FeaturePtr f;
    if (!getFeature(cam, names, f)) return false;
    VmbBool_t writable = false, readable = false;
    if (f->IsWritable(writable) != VmbErrorSuccess)
        std::cout << "WARN: Accessing IsWriteable() failed for " << names[0] << std::endl;;
    if (f->IsReadable(readable) != VmbErrorSuccess)
        std::cout << "WARN: Accessing IsReadalbe() failed for " << names[0] << std::endl;

    VmbInt64_t minV = 0, maxV = 0;
    if (f->GetRange(minV, maxV) == VmbErrorSuccess) {
        if (value < minV) value = minV;
        if (value > maxV) value = maxV;
    }
    return f->SetValue(value) == VmbErrorSuccess;
}

bool VimbaBackend::setBool(const VmbCPP::CameraPtr& cam,
    const std::vector<std::string>& names,
    bool value)
{
    VmbCPP::FeaturePtr f;
    if (!getFeature(cam, names, f)) return false;

    VmbBool_t writable = false, readable = false;
    if (f->IsWritable(writable) != VmbErrorSuccess)
        std::cout << "WARN: Accessing IsWriteable() failed for " << names[0] << std::endl;;
    if (f->IsReadable(readable) != VmbErrorSuccess)
        std::cout << "WARN: Accessing IsReadalbe() failed for " << names[0] << std::endl;

    return f->SetValue(value ? true : false) == VmbErrorSuccess;
}

void VimbaBackend::findFeatures(
    VmbCPP::CameraPtr& cam,
    std::vector<std::string> keys) const
{
    VmbCPP::FeaturePtrVector feats;
    cam->GetFeatures(feats);

    std::cout << " --- Search for features Keys --- \n";

    auto runCmd = [&](const char* name) {
        VmbCPP::FeaturePtr f;
        if (cam->GetFeatureByName(name, f) == VmbErrorSuccess && f) {
            auto err = f->RunCommand();
            std::cout << "RunCommand " << name << " err=" << err << "\n";
        }
        else {
            std::cout << "Cmd " << name << " NOT FOUND\n";
        }
        };

    auto getInt = [&](const char* name) {
        VmbCPP::FeaturePtr f;
        if (cam->GetFeatureByName(name, f) == VmbErrorSuccess && f) {
            VmbInt64_t v = 0;
            auto err = f->GetValue(v);
            std::cout << name << " err=" << err << " val=" << v << "\n";
        }
        };

    // searches for the featurs for the including the key
    for (const auto& ft : feats) {
        std::string names;
        ft->GetName(names);
        for (const auto& key : keys) {
            std::size_t pos = names.find(key);
            if (pos != std::string::npos) {
                std::cout << names << '\n';
                std::cout << "Available entries if enum: \n";
                VmbCPP::EnumEntryVector entries;

                // If enum feature give show the available entries. 
                // If float int or command feature no entries are available. 
                ft->GetEntries(entries);
                for (const auto& entry : entries) {
                    std::string description, name;
                    VmbInt64_t enumval;
                    if (entry.GetDescription(description) == VmbErrorSuccess)
                        std::cout << "Description: " << description << std::endl;
                    if (entry.GetName(name) == VmbErrorSuccess)
                        std::cout << "Name: " << name << std::endl;
                    if (entry.GetValue(enumval))
                        std::cout << "Acutall value: " << enumval << std::endl;
                }
                
                // Checks if readable, writeable and which datatype is it. 
                VmbBool_t r = false, w = false;
                ft->IsReadable(r); ft->IsWritable(w);
                std::string category;
                VmbFeatureDataType dtype;
                if (ft->GetDataType(dtype) != VmbErrorSuccess) dtype = VmbFeatureDataUnknown;
                std::cout << names
                    << " readable=" << (r ? 1 : 0)
                    << " writable=" << (w ? 1 : 0)
                    << " dtype=" << (int)dtype << std::endl;
                std::cout << '\n';
            }
            
        }
    }
    //getInt("TLParamsLocked");       // if exists, 1 means locked
    //runCmd("AcquisitionStop");      // if exists
    //runCmd("AcquisitionAbort");     // sometimes present
    //getInt("TLParamsLocked");
    std::cout << "Search finished \n";
    return;
}



bool VimbaBackend::forceFreerunTimedExposure(const VmbCPP::CameraPtr& cam)
{
    // Stop acquisition just in case 
    VmbCPP::FeaturePtr f;

   

    // For all entries of Trigger Selector set TriggerMode = OFF
    if (getFeature(cam, std::vector<std::string>{"TriggerSelector"}, f) == VmbErrorSuccess && f) {
        std::vector<VmbCPP::EnumEntry> entries; // => VmbCPP::EnumEntryVector
        if (f->GetEntries(entries) == VmbErrorSuccess && !entries.empty()) {
            for (const auto& entry : entries) {
                std::string entry_name;
                entry.GetName(entry_name);
                if (!setEnum(cam, std::vector<std::string>{"TriggerSelector"}, entry_name.c_str())) {
                    std::cout << "WARN: TriggerSelector to " << entry_name << " failed \n";
                    continue; // Just continue to the next entry. 
                }
                if (!setEnum(cam, std::vector<std::string>{"TriggerMode"}, "Off")) {
                    std::cout << "WARN: Could not set TriggerMode to Off for Entry: " << entry_name << std::endl;
                }
            }
        }
    }
    
    // Trigger Source may be set before the all TriggerEntries are set to Off. 

    if (!setEnum(cam, { "TriggerSource" }, "Freerun"))
        std::cout << "WARN: TriggerSource Freerun failed \n";

    return true;
}

bool VimbaBackend::setGain(const VmbCPP::CameraPtr& cam, double gain_dB)
{
    VmbCPP::FeaturePtr f;

    if (!setFloatClamped(cam, std::vector<std::string>{"Gain"}, gain_dB)) {
        std::cout << "Gain Value could not be Set \n";
        return false;
    }

    if (cam->GetFeatureByName("GainSelector", f) != VmbErrorSuccess) {
        std::cout << "Gain Selector could not be selected \n";
    }
    else {
        if (f->SetValue("All") == VmbErrorSuccess) {
            std::cout << "Gain Selector is set to all \n";
        }
        else {
            std::cout << "Gain Selecto but All is the only option \n";
        }
    }

    if (cam->GetFeatureByName("GainAuto", f) != VmbErrorSuccess) {
        std::cout << "Gain Auto was not selected \n";
        return false;
    }
    else {
        if (f->SetValue("Off") != VmbErrorSuccess) {
            std::cout << "AutoGain is not disabled! \n";
            return false;
        }
        else {
            std::cout << "AutoGain is disabled \n";
        }
    }

    return true;
}

bool VimbaBackend::setExposureAbsRobust(const VmbCPP::CameraPtr& cam, double targetUs)
{
    VmbCPP::FeaturePtr f;

    if (!setFloatClamped(cam, std::vector<std::string>{"ExposureTimeAbs"}, targetUs)) {
        std::cout << "ExposureTimeAbs could not be Set to the given value \n";
        return false;
    }

    if (getFeature(cam, std::vector<std::string>{"ExposureMode"}, f) == VmbErrorSuccess && f) {
        std::vector<VmbCPP::EnumEntry> entries;
        if (f->GetEntries(entries) == VmbErrorSuccess && !entries.empty()) {
            for (const auto& entry : entries) {
                std::string entry_name;
                entry.GetName(entry_name);
                if (!setEnum(cam, std::vector<std::string>{"ExposureMode"}, entry_name.c_str())) {
                    std::cout << "WARN: ExposureMode to " << entry_name << " failed \n";
                    continue; // Just continue to the next entry. 
                }
                if (!setEnum(cam, std::vector<std::string>{"ExposureAuto"}, "Off")) {
                    std::cout << "WARN: Could not set ExposureMode to Off for Entry: " << entry_name << std::endl;
                }
            }
        }
    }

    return true;
}

bool VimbaBackend::setGamma(
    const VmbCPP::CameraPtr& cam,
    double gamma) {
    
    if (!setFloatClamped(cam, std::vector<std::string>{"Gamma"}, gamma)) {
        std::cout << "Gamma Value could not be set \n";
        return false;
    }
    std::cout << "Successfull set Gamma Value \n";
    return true;
}

// --- Types with in the VmbFeatureptr
//typedef enum VmbFeatureDataType
//{
//    VmbFeatureDataUnknown = 0,        //!< Unknown feature type
//    VmbFeatureDataInt = 1,        //!< 64-bit integer feature
//    VmbFeatureDataFloat = 2,        //!< 64-bit floating point feature
//    VmbFeatureDataEnum = 3,        //!< Enumeration feature
//    VmbFeatureDataString = 4,        //!< String feature
//    VmbFeatureDataBool = 5,        //!< Boolean feature
//    VmbFeatureDataCommand = 6,        //!< Command feature
//    VmbFeatureDataRaw = 7,        //!< Raw (direct register access) feature
//    VmbFeatureDataNone = 8,        //!< Feature with no data
//};

bool VimbaBackend::adjustSettings(const std::size_t camera_index) {

    auto cam = findCameraByID(camera_index);
    if (!cam) return false;

    VmbCPP::FeaturePtr f;
    
    std::vector<std::string> key{"Frame", "Exposure"};
    // --- find Features can be used to search in the feature tree for string snippets ---
    //findFeatures(cam, key);
    
    bool forceFree = forceFreerunTimedExposure(cam);
    bool forceGain = setGain(cam, 0.0);
    bool exposure = setExposureAbsRobust(cam, 3500.0);
    bool gamma = setGamma(cam, 1.0);
    bool roi = setRoi(cam);
    bool format = setFormat(cam);
    checkPackagesize(cam);

    // If you want: log which ones failed (feature not present vs not writable)
    return forceFree && forceGain && exposure && roi && format;
}

void VimbaBackend::checkPackagesize(
    const VmbCPP::CameraPtr& cam) {
    if (!cam) return;
    VmbCPP::FeaturePtr pPacketSizeFeature;
    VmbError_t err = cam->GetFeatureByName("GevSCPSPacketSize", pPacketSizeFeature);

    if (VmbErrorSuccess != err) {
        std::cout << "Fehler: Feature GevSCPSPacketSize nicht gefunden! Code: " << err << std::endl;
        return;
    }

    VmbInt64_t val;
    pPacketSizeFeature->GetValue(val);
    std::cout << "Aktuelle Packet Size: " << val << std::endl;

}

bool VimbaBackend::setFormat(
    const VmbCPP::CameraPtr& cam)
{
    if (!cam) return false;
    if (!setEnum(cam, std::vector<std::string>{"PixelFormat"}, "Mono8")) return false;
    return true;
}

bool VimbaBackend::setRoi(
    const VmbCPP::CameraPtr& cam,
    int x_0,
    int y_0,
    int height,
    int width)
{
    if (!cam) return false;
    assert(x_0 >= 0 && y_0 >= 0 && height > 0 && width > 0 && "The ROI Values are out of Range");
    if (!setIntClamped(cam, std::vector<std::string>{"OffsetX"}, 0)) return false;
    if (!setIntClamped(cam, std::vector<std::string>{"OffsetY"}, 0)) return false;
    if (!setIntClamped(cam, std::vector<std::string>{"Height"}, 2056)) return false;  // max_values
    if (!setIntClamped(cam, std::vector<std::string>{"Width"}, 2464)) return false;   // max_values
    std::cout << "Roi set successfully \n";
    
    return true;
}


bool VimbaBackend::allocateBuffer(
    std::size_t i,
    int n_buf) {
    auto cam = findCameraByID(i);
    VmbCPP::FeaturePtr ft;
    VmbInt64_t nPLS;

    if (cam->GetFeatureByName("PayloadSize", ft) != VmbErrorSuccess && ft) {
        std::cout << "Could not get Feature PayloadSize \n";
        return false;
    }
    if (ft->GetValue(nPLS) != VmbErrorSuccess) {
        std::cout << "Could not get the Payloadsize \n";
        return false;
    }
    
    m_frame_ptr = std::vector<VmbCPP::FramePtr>(n_buf);

    if (m_ringBufferPtr == nullptr) {
        m_ringBufferPtr = new RingBuffer();
        std::cout << "Ring Buffer Object is allocated on the Heap \n";
    }

    for (std::vector<VmbCPP::FramePtr>::iterator iter = m_frame_ptr.begin();
        iter != m_frame_ptr.end(); ++iter) {
        (*iter).reset(new VmbCPP::Frame(nPLS));
    }

    return true;
}

bool VimbaBackend::runAcquisition(const VmbCPP::CameraPtr& cam) {
    std::unique_lock<std::mutex> lk(m_mut);

    if (m_ringBufferPtr == nullptr) {
        m_ringBufferPtr = new RingBuffer();
    }

    for (const auto& frames : m_frame_ptr) {
        //auto observer = std::make_shared<FrameObserver>(cam);
        if (frames->RegisterObserver(VmbCPP::IFrameObserverPtr(
                new FrameObserver(cam, m_ringBufferPtr, &m_stopping, &m_callbacksInFlight))) != VmbErrorSuccess) {
            std::cout << "Register Frames to the FramesObserver Failed \n";
            return false;
        }
        //frames->RegisterObserver(VmbCPP::IFrameObserverPtr(std::make_shared<FrameObserver>(cam)));
        if (cam->AnnounceFrame(frames) != VmbErrorSuccess) {
            std::cout << "Annoucing Frames to the camera Failed \n";
            return false;
        }
    }

    if (cam->StartCapture() != VmbErrorSuccess) {
        std::cout << "Camera StartUp failed \n";
        return false;
    }

    for (const auto& frame : m_frame_ptr) {
        if (cam->QueueFrame(frame) != VmbErrorSuccess) {
            std::cout << "Queue Frames failed \n";
            return false;
        }
    }
    
    VmbCPP::FeaturePtr pFeature;
    if (cam->GetFeatureByName("AcquisitionStart", pFeature) != VmbErrorSuccess && pFeature) return false;
    pFeature->RunCommand();
    m_running = true;
    return true;
}


bool VimbaBackend::open(std::size_t i)
{   
    if (openCamera(i)) {
        std::cout << "Opened Camera Sucessfull \n";
    }
    else return false;

    if (adjustSettings(i)) {
        std::cout << "Sucessfully changed the settings \n";
    }
    else return false;

    // Hardcoded the Buffer Ammount
    if (allocateBuffer(i, 10)) {
        std::cout << "Sucessfully allocated buffer \n";
    }
    else return false;

    auto cam = findCameraByID(i);

    if (!runAcquisition(cam)) {
        std::cout << "Starting Acquisition failed \n";
        return false;
    }

    logging(i);
    return true;
}

void VimbaBackend::close(std::size_t i) 
{
    auto cam = findCameraByID(m_cam_index);

    if (!cam) throw std::runtime_error("The CameraPtr is nullptr");
       
    m_stopping.store(true, std::memory_order_relaxed);
    
    if (m_running) {
        {
            VmbCPP::FeaturePtr ft;
            if (cam->GetFeatureByName("AcquisitionStop", ft) == VmbErrorSuccess && ft) {
                if (ft->RunCommand() != VmbErrorSuccess)
                    std::cout << "AcquisitionStop failed \n";
            }
        }
        if(cam->EndCapture() != VmbErrorSuccess)
            std::cout << "EndCapture failed \n";
        
        
        if(cam->FlushQueue() != VmbErrorSuccess)
            std::cout << "FlushQueue failed \n";
        if (cam->RevokeAllFrames() != VmbErrorSuccess)
            std::cout << "RevokeAllFrames failed \n";
        

        for (auto& frames : m_frame_ptr) {
            if (frames && frames->UnregisterObserver() != VmbErrorSuccess)
                std::cout << "UnregisterObserver failed \n";
        }
        m_frame_ptr.clear();
        deleteBuffer();

        if (cam->Close() != VmbErrorSuccess)
            std::cout << "Closing the camera failed \n";
    }
    

    m_running = false;
}

void VimbaBackend::logging(const std::size_t camera_index)
{
    auto check_read_val = [](const VmbCPP::CameraPtr& cam, VmbCPP::FeaturePtr& ft, const std::string& name, bool& readable)
        {
            readable = false;
            if (cam->GetFeatureByName(name.c_str(), ft) != VmbErrorSuccess) {
                std::cout << "Get FeaturePtr for " << name << " failed \n";
                return;
            }
            if (ft->IsReadable(readable) != VmbErrorSuccess)
            std::cout << "FeaturePtr " << name << " is not readalbe \n";
            return;
        };

    auto cam = findCameraByID(m_cam_index);
    if (!cam) throw std::runtime_error("The CameraPtr is nullptr");
    VmbCPP::FeaturePtr f;
    bool readable{ false };
    
    // Create Camera logging file 
    m_camera_data.push_back(defl::CameraConfig{});
    std::vector<defl::CameraConfig>::iterator it = std::prev(m_camera_data.end()); // working iterator for logging
   
    // Pixel_y
    check_read_val(cam, f, std::string{ "Height" }, readable);
    if (readable) {
        VmbInt64_t val{};
        if (f->GetValue(val) != VmbErrorSuccess) std::cout << "Logging Pixel Y failed \n";
        it->pixel_y = static_cast<int>(val);
    }
    // Pixel_x
    check_read_val(cam, f, std::string{ "Width" }, readable);
    if (readable) {
        VmbInt64_t val{};
        if (f->GetValue(val) != VmbErrorSuccess) std::cout << "Logging Pixel X failed \n";
        it->pixel_x = static_cast<int>(val);
    }
    // AcquisitionModeActive
    it->acquisition_mode_active = m_running;

    // ModeName
    std::string modename{};
    if (cam->GetName(modename) != VmbErrorSuccess) { std::cout << "Could not read Camera Name \n"; }
    it->model_name = modename;

    // Device Index
    std::string extendedId{}; 
    if (cam->GetExtendedID(extendedId) != VmbErrorSuccess) { std::cout << "Could not read ExtendedID \n"; }

    // Exposure Time
    check_read_val(cam, f, std::string{ "ExposureTimeAbs" }, readable);
    if (readable) {
        double exp_time{};
        if (f->GetValue(exp_time) != VmbErrorSuccess) { std::cout << "Could not read the Exposure time \n"; }
        it->exposure_time = exp_time;
    }

    // Frame Rate 
    check_read_val(cam, f, std::string{ "AcquisitionFrameRateAbs" }, readable);
    if (readable) {
        double frame_rate{};
        if (f->GetValue(frame_rate) != VmbErrorSuccess) { std::cout << "Could not read the Frame Rate \n"; }
        it->frame_rate = frame_rate;
    }
    // Gain 
    check_read_val(cam, f, std::string{ "Gain" }, readable);
    if (readable) {
        double gain{};
        if (f->GetValue(gain) != VmbErrorSuccess) { std::cout << "Could not read the Gain val \n"; }
        it->gain = gain;
    }

    // PixelFormat
    check_read_val(cam, f, std::string{ "PixelFormat" }, readable);
    if (readable) {
        std::string entry;
        if (f->GetValue(entry) != VmbErrorSuccess) { std::cout << "Could not read the PixelFormat entry \n"; }
        it->pixel_format = entry;
    }

    it->timestamp = std::chrono::system_clock::now();

    /*check_read_val(cam, f, std::string{})
    

   
    if(cam->GetFeatureByName()
    VmbBool_t writable = false, readable = false;
    if (f->IsWritable(writable) != VmbErrorSuccess)
        std::cout << "WARN: Accessing IsWriteable() failed for " << names[0] << std::endl;;
    if (f->IsReadable(readable) != VmbErrorSuccess)
        std::cout << "WARN: Accessing IsReadalbe() failed for " << names[0] << std::endl;
    config.pixel_x = */
}

void VimbaBackend::deleteBuffer() {
    if (m_ringBufferPtr == nullptr) return;
    delete m_ringBufferPtr;
    m_ringBufferPtr = nullptr;
}

cv::Mat VimbaBackend::grab(int) { 
    if (!m_running) {
        return {};
    }

    return m_ringBufferPtr->extract();
}

bool VimbaBackend::isRunning(std::size_t) { return m_running; }

std::vector<defl::CameraConfig> VimbaBackend::getCameraConfig() {
    return m_camera_data;
}
