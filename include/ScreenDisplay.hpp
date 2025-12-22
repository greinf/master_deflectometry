#ifndef SCREENDISPLAY
#define SCREENDISPLAY
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <optional>

class ScreenDisplay {
public:
    ScreenDisplay() = default;
    ~ScreenDisplay() { stop(); }

    // Start UI-thread
    void start() {
        // if already running-> return 
        if (m_running.load()) return;
        m_running.store(true);
        m_thread = std::thread(&ScreenDisplay::uiLoop, this);
    }

    // Stop UI-thread
    void stop() {
        m_running.store(false);

        if (m_thread.joinable()) {
            // Wake up UI thread so waitKey doesn't block
            cv::waitKey(1);
            m_thread.join();
        }
    }

    // Push camera frame to display
    void showCamera(const cv::Mat& img) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            cv::Mat image;
            if (ptr) image = ptr(img);
            else image = img;
            image.copyTo(m_camFrame);
        }
    }

    // Push pattern frame to display
    void showPattern(const cv::Mat& img) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            img.copyTo(m_patternFrame);
        }
    }

    // Push processed image
    void showProcessed(const cv::Mat& img) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            img.copyTo(m_procFrame);
        }
    }

    void assignCameraPreProcessing(cv::Mat(*function_ptr)(const cv::Mat& a)) {
        ptr = function_ptr;
    }

private:
    // Function Pointer for Camera PreProcessing if needed
    cv::Mat(*ptr)(const cv::Mat& a) = nullptr;
    std::thread m_thread;
    std::atomic<bool> m_running{ false };
    std::mutex m_mutex;

    cv::Mat m_camFrame;
    cv::Mat m_patternFrame;
    cv::Mat m_procFrame;

    void uiLoop() {
        cv::namedWindow("Camera", cv::WINDOW_NORMAL);
        cv::namedWindow("Pattern", cv::WINDOW_NORMAL);
        cv::namedWindow("Processed", cv::WINDOW_NORMAL);

        cv::setWindowProperty("Camera", cv::WND_PROP_TOPMOST, 1);
        cv::setWindowProperty(
            "Pattern",
            cv::WND_PROP_FULLSCREEN,
            cv::WINDOW_FULLSCREEN
        );
        cv::moveWindow("Pattern", 1920, 0);

        while (m_running.load()) {
            cv::Mat cam, pat, proc;

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!m_camFrame.empty()) cam = m_camFrame.clone();
                if (!m_patternFrame.empty()) pat = m_patternFrame.clone();
                if (!m_procFrame.empty()) proc = m_procFrame.clone();
            }

            if (!cam.empty()) cv::imshow("Camera", cam);
            if (!pat.empty()) cv::imshow("Pattern", pat);
            if (!proc.empty()) cv::imshow("Processed", proc);

            // keeps windows responsive
            cv::waitKey(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        cv::destroyAllWindows();
    }
};

#endif 