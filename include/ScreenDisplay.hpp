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

    void showCamera(const std::vector<cv::Mat>& img)
    {
        CV_Assert(img.size() <= 2);
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<cv::Mat> images(img.size());

        for (std::size_t i = 0; i < img.size(); ++i) {
            if (img[i].empty()) continue;
            if (ptr) images[i] = ptr(img[i]);
            else images[i] = img[i];
            images[i] = ptr(img[i]);
            (i % 2) ? (images[1].copyTo(m_camFrame1)) : images[0].copyTo(m_camFrame0);
        }
    }

    // Push camera frame to display
    void showCamera(const cv::Mat& img) {
        {
            std::cout << "Frame at Screen Dispaly \n";
            std::lock_guard<std::mutex> lock(m_mutex);
            cv::Mat image;
            if (ptr) image = ptr(img);
            else image = img;
            image.copyTo(m_camFrame0);
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

    cv::Mat m_camFrame0;
    cv::Mat m_camFrame1;
    cv::Mat m_patternFrame;
    cv::Mat m_procFrame;

    void uiLoop() {
        cv::namedWindow("Camera1", cv::WINDOW_NORMAL);
        cv::namedWindow("Camera2", cv::WINDOW_NORMAL);
        cv::namedWindow("Pattern", cv::WINDOW_NORMAL);
        cv::namedWindow("Processed", cv::WINDOW_NORMAL);

        cv::setWindowProperty("Camera1", cv::WND_PROP_TOPMOST, 1);
        cv::setWindowProperty("Camera2", cv::WND_PROP_TOPMOST, 1);
        
        cv::moveWindow("Pattern", 1920, 0);
        cv::setWindowProperty(
            "Pattern",
            cv::WND_PROP_FULLSCREEN,
            cv::WINDOW_FULLSCREEN
        );

        while (m_running.load()) {
            cv::Mat cam0, cam1 , pat, proc;

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!m_camFrame0.empty()) cam0 = m_camFrame0.clone();
                if (!m_camFrame1.empty()) cam1 = m_camFrame1.clone();
                if (!m_patternFrame.empty()) pat = m_patternFrame.clone();
                if (!m_procFrame.empty()) proc = m_procFrame.clone();
            }

            if (!cam0.empty()) cv::imshow("Camera1", cam0);
            if (!cam1.empty()) cv::imshow("Camera2", cam1);
            if (!pat.empty()) cv::imshow("Pattern", pat);
            if (!proc.empty()) cv::imshow("Processed", proc);

            // keeps windows responsive
            cv::waitKey(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        cv::destroyAllWindows();
    }
};

#endif 