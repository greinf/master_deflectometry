#ifndef ACQUISITIONCONTROLLER
#define ACQUISITIONCONTROLLER
#include <atomic>
#include <mutex>
#include <condition_variable>

enum class FrameRole;


namespace defl {

    class AcquisitionController {
    public:
        std::atomic<bool> acquisition_active{ false };

        std::atomic<bool> pattern_processed{ false };

        std::mutex mtx;
        std::condition_variable cv;

        FrameRole mode;
        
        void reset() {
            acquisition_active.store(false);
            pattern_processed.store(false);
        }
    };

} // namespace defl

#endif