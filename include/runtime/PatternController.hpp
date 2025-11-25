#ifndef PATTERNCONTROLLER_H
#define PATTERNCONTROLLER_H
#include <atomic>

namespace defl {

    class PatternController {
    public:
        std::atomic<int> pictures_per_pattern{ 1 };
        std::atomic<int> number_of_shifts{ 1 };
        std::atomic<bool> pattern_iteration_finished{ false };

        void reset() {
            pictures_per_pattern.store(1);
            number_of_shifts.store(1);
            pattern_iteration_finished.store(false);
        }
    };

} // namespace defl


#endif