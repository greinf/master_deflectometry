#pragma once
#include <thread>
#include <atomic>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "imageStore.hpp"
#include "ScreenDisplay.hpp"
#include "runtime/AcquisitionController.hpp"
#include "CameraNew.hpp"

class AcquisitionWorker {
public:
    AcquisitionWorker(std::shared_ptr<CameraN> cam,
        std::shared_ptr<ImageStore> store,
        std::shared_ptr<defl::AcquisitionController> controller,
        std::shared_ptr<ScreenDisplay> display = nullptr
        )
        : m_camera(std::move(cam))
        , m_store(std::move(store))
        , m_display(std::move(display))
        , m_controller(std::move(controller))
    {
    }

    ~AcquisitionWorker() {
        stop();
    }

    void start() {
        if (m_running.load()) return;
        m_running.store(true);
        //m_camera->open();
        m_thread = std::thread(&AcquisitionWorker::loop, this);
    }

    void stop() {
        m_running.store(false);
        if (m_thread.joinable())
            m_thread.join();
        m_camera->close();
    }

    // === NEW: REQUEST TO SAVE NEXT IMAGE(S) ===
    void requestSave(int count = 1) {
        m_pendingSaves.fetch_add(count, std::memory_order_relaxed);
    }

    // === NEW: SAVE A PICTURE AFTER X MILLISECONDS ===
    void requestTimedSave(int ms) {
        std::thread([this, ms]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            this->requestSave(1);
            }).detach();
    }

private:
    std::shared_ptr<CameraN> m_camera;
    std::shared_ptr<ImageStore> m_store;
    std::shared_ptr<ScreenDisplay> m_display;
    std::shared_ptr<defl::AcquisitionController> m_controller;

    std::thread m_thread;
    std::atomic<bool> m_running{ false };

    // === NEW: counter for pending save operations ===
    std::atomic<int> m_pendingSaves{ 0 };

    void loop() {
       
        while (m_running.load()) {
            try {
                cv::Mat view = m_camera->grab();

                // empty cv::Mat gets returned if buffer was empty
                if (view.empty()) continue;

                /*std::cout << "Channels " << view.channels() << '\n';
                std::cout << "Size " << view.size() << '\n';*/
                
                // Always store raw if wanted
                //m_store->add(FrameRole::RawFrame, gray);

                // Show live preview
                if (m_display)
                    m_display->showCamera(view);

                // === NEW: SAVE REQUEST LOGIC ===
                int saveNow = m_pendingSaves.load();
                if (saveNow > 0) {
                    std::unique_lock m_img_save (m_controller->mtx);
                    m_pendingSaves.fetch_sub(1);
                    m_store->add(m_controller->mode, view.clone());
                    m_img_save.unlock();
                    m_controller->cv.notify_one();
                }
            }
            catch (...) {
            }
        }
    }
};

