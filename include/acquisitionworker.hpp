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
    AcquisitionWorker(
        std::shared_ptr<CameraN> cam, // for now fine
        ImageStore& store,
        defl::AcquisitionController& controller,
        ScreenDisplay& display 
        )
        : m_camera(std::move(cam))
        , m_store(store)
        , m_display(display)
        , m_controller(controller)
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
    std::shared_ptr<CameraN> m_camera; // fine for now
    ImageStore& m_store;
    ScreenDisplay& m_display;
    defl::AcquisitionController& m_controller;

    std::thread m_thread;
    std::atomic<bool> m_running{ false };

    // === NEW: counter for pending save operations ===
    std::atomic<int> m_pendingSaves{ 0 };

    void loop() {
       
        while (m_running.load()) {
            try {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                std::vector<cv::Mat> view = m_camera->grab();

                // empty cv::Mat gets returned if buffer was empty
                if (view.empty()) continue;

                m_display.showCamera(view);

                // === NEW: SAVE REQUEST LOGIC ===
                int saveNow = m_pendingSaves.load();
                if (saveNow > 0) {
                    
                    std::unique_lock m_img_save (m_controller.mtx);
                    m_pendingSaves.fetch_sub(1);
                    for (const auto& img : view) {
                        m_store.add(m_controller.mode, img.clone());
                    }
                    m_img_save.unlock();
                    m_controller.cv.notify_one();
                }
            }
            catch (...) {
            }
        }
    }
};

