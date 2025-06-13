#ifndef __AR3DVTOOLS_TIMER_H__
#define __AR3DVTOOLS_TIMER_H__

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "statistics.h"

#define ENABLE_TIMING 1

/*
// Example usage:
#define ENABLE_TIMING 1
ar3dv::Timer my_timer("My Operation");
// Some expensive operation that should be timed.
my_timer.Stop();
// Print all values.
ar3dv::Timing::Print();
// just get last operation time cost;
std::cout << my_timer.Stop();
*
*/

namespace ar3dv {

    double CurrentTimestampS();


    // A class that has the timer interface but does nothing. Swapping this in
    // place of the Timer class (say with a typedef) should allow one to disable
    // timing. Because all of the functions are inline, they should just disappear.
    class DummyTimer {
    public:
        DummyTimer(size_t /*handle*/, bool construct_stopped = false) {
            static_cast<void>(construct_stopped);
        }
        DummyTimer(const std::string& /*tag*/, bool construct_stopped = false) {
            static_cast<void>(construct_stopped);
        }
        ~DummyTimer() {}
        void Start() {}
        double Stop() {
            return -1.0;
        }
        void Discard() {}
        bool IsTiming() const {
            return false;
        }
        size_t GetHandle() const {
            return 0u;
        }
    };

    class TimerImpl {
    public:
        TimerImpl(const std::string& tag, bool construct_stopped = false);
        ~TimerImpl();

        void Start();
        // Returns the amount of time passed between Start() and Stop().
        double Stop();
        void Discard();
        bool IsTiming() const;
        size_t GetHandle() const;

    private:
        std::chrono::time_point<std::chrono::system_clock> time_;

        bool is_timing_;
        size_t handle_;
        std::string tag_;
    };

    class Timing {
    public:
        typedef std::map<std::string, size_t> map_t;
        friend class TimerImpl;
        // Definition of static functions to query the timers.
        static size_t GetHandle(const std::string& tag);
        static std::string GetTag(size_t handle);
        static double GetTotalSeconds(size_t handle);
        static double GetTotalSeconds(const std::string& tag);
        static double GetMeanSeconds(size_t handle);
        static double GetMeanSeconds(const std::string& tag);
        static size_t GetNumSamples(size_t handle);
        static size_t GetNumSamples(const std::string& tag);
        static double GetVarianceSeconds(size_t handle);
        static double GetVarianceSeconds(const std::string& tag);
        static double GetMinSeconds(size_t handle);
        static double GetMinSeconds(const std::string& tag);
        static double GetMaxSeconds(size_t handle);
        static double GetMaxSeconds(const std::string& tag);
        static double GetHz(size_t handle);
        static double GetHz(const std::string& tag);
        static void WriteToYamlFile(const std::string& path);
        static void Print(std::ostream& out);  // NOLINT
        static std::string Print();
        static std::string SecondsToTimeString(double seconds);
        static void Reset();
        static const map_t& GetTimerImpls() {
            return Instance().tag_map_;
        }

    private:
        void AddTime(size_t handle, double seconds);

        static Timing& Instance();

        Timing();
        ~Timing();

        typedef std::vector<ar3dv::StatisticsMapValue> list_t;

        list_t timers_;
        map_t tag_map_;
        size_t max_tag_length_;
        std::mutex mutex_;
    };

#if ENABLE_TIMING
    typedef TimerImpl Timer;
#else
    typedef DummyTimer Timer;
#endif

}  // namespace ar3dv
#endif //__AR3DVTOOLS_TIMER_H__