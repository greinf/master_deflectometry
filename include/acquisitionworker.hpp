#pragma once
#include <thread>
#include <atomic>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "camera.hpp"
#include "imageStore.hpp"
#include "ScreenDisplay.hpp"
#include "runtime/AcquisitionController.hpp"

class AcquisitionWorker {
public:
    AcquisitionWorker(std::shared_ptr<Camera> cam,
        std::shared_ptr<ImageStore> store,
        std::shared_ptr<defl::AcquisitionController> controller,
        std::shared_ptr<ScreenDisplay> display = nullptr
        )
        : m_camera(cam)
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
        m_thread = std::thread(&AcquisitionWorker::loop, this);
    }

    void stop() {
        m_running.store(false);
        if (m_thread.joinable())
            m_thread.join();
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
    std::shared_ptr<Camera> m_camera;
    std::shared_ptr<ImageStore> m_store;
    std::shared_ptr<ScreenDisplay> m_display;
    std::shared_ptr<defl::AcquisitionController> m_controller;

    std::thread m_thread;
    std::atomic<bool> m_running{ false };

    // === NEW: counter for pending save operations ===
    std::atomic<int> m_pendingSaves{ 0 };

    void loop() {
        auto ds = m_camera -> dataStream(0);
        auto nm = m_camera -> nodeMap(0);

        while (m_running.load()) {
            try {
                auto buffer = ds->WaitForFinishedBuffer(5000);
                if (!buffer) continue;

                cv::Mat view(buffer->Height(), buffer->Width(),
                    CV_8UC1, (void*)buffer->BasePtr(), buffer->Width());

                /*std::cout << "Channels " << view.channels() << '\n';
                std::cout << "Size " << view.size() << '\n';*/
                // convert bayer to gray
                cv::Mat gray;
                cv::rotate(view, gray, cv::ROTATE_180);

                
                cv::cvtColor(gray, gray, cv::COLOR_BayerRG2GRAY);
                
                // Always store raw if wanted
                //m_store->add(FrameRole::RawFrame, gray);

                // Show live preview
                if (m_display)
                    m_display->showCamera(gray);

                // === NEW: SAVE REQUEST LOGIC ===
                int saveNow = m_pendingSaves.load();
                if (saveNow > 0) {
                    std::unique_lock m_img_save (m_controller->mtx);
                    m_pendingSaves.fetch_sub(1);
                    m_store->add(m_controller->mode, gray.clone());
                    m_img_save.unlock();
                    m_controller->cv.notify_one();
                }

                ds->QueueBuffer(buffer);
            }
            catch (...) {
            }
        }
    }
};

