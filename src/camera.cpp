#include "camera.hpp"

void wait_for_enter();

Camera::Camera()
{
    /*!
     * \file    open_camera.cpp
     * \author  IDS Imaging Development Systems GmbH
     * \date    2021-03-05
     * \since   1.0.0
     *
     * \brief   This application demonstrates how to use the device manager to open a camera
     *
     * \version 1.1.2
     *
     * Copyright (C) 2019 - 2025, IDS Imaging Development Systems GmbH.
     *
     * The information in this document is subject to change without notice
     * and should not be construed as a commitment by IDS Imaging Development Systems GmbH.
     * IDS Imaging Development Systems GmbH does not assume any responsibility for any errors
     * that may appear in this document.
     *
     * This document, or source code, is provided solely as an example of how to utilize
     * IDS Imaging Development Systems GmbH software libraries in a sample application.
     * IDS Imaging Development Systems GmbH does not assume any responsibility
     * for the use or reliability of any portion of this document.
     *
     * General permission to copy or modify is hereby granted.
     */
    //Taken from this file with small adjustmants

    // initialize peak library
    std::cout << "entered Constructor " << std::endl;
    peak::Library::Initialize();

    // create a device manager object
    auto& deviceManager = peak::DeviceManager::Instance();

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
            camera_ptr.push_back(deviceManager.Devices().at(selectedDevice)->OpenDevice(peak::core::DeviceAccessType::Control));

            //auto device = deviceManager.Devices().at(selectedDevice)->OpenDevice(peak::core::DeviceAccessType::Control);
            //auto nodeMapRemoteDevice = device->RemoteDevice()->NodeMaps().at(0);
        }
        else if (deviceManager.Devices().size() == 1) {
            camera_ptr.push_back(deviceManager.Devices().at(0)->OpenDevice(peak::core::DeviceAccessType::Control));
        }

    }
    catch (const std::exception& e)
    {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
    }
}

void wait_for_enter()
{
    std::cout << std::endl;
#if defined(_WIN32)
    system("pause");
#endif
}