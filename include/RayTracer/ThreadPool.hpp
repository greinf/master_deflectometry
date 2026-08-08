#ifndef THREADPOOL_HPP
#define	THREADPOOL_HPP

#include <type_traits>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <vector>
#include <functional>
#include <queue>
#include <future>
#include <utility>
#include <memory>
#include <algorithm>
#include <atomic>
#include <tuple>


class ThreadPool {
private:
	ThreadPool()
	{
		n_workers = std::max(std::thread::hardware_concurrency(), static_cast<unsigned int>(1));
	};

	std::size_t n_workers{};
	std::atomic<std::size_t> m_active_threads{};
	std::atomic<bool> m_stop{ true };
	std::vector<std::thread> m_thread{};

	std::mutex mtx{};
	std::condition_variable cv{};

	std::queue<std::function<void()>> m_task{};

public:
	ThreadPool(const ThreadPool&) = delete;
	ThreadPool(ThreadPool&&) noexcept = delete;

	ThreadPool& operator=(const ThreadPool&) = delete;
	ThreadPool& operator=(ThreadPool&&) noexcept = delete;

	static ThreadPool& instance() {
		static ThreadPool inst{};
		return inst;
	}

	void start()
	{
		{
			std::lock_guard<std::mutex> lock(mtx);

			if (!m_stop.load(std::memory_order_relaxed))
				return; // already running

			m_stop.store(false, std::memory_order_relaxed);
		}

		for (std::size_t i = 0; i < n_workers; ++i) {
			m_thread.emplace_back(
				[this]() {
					while (true) {

						std::function<void()> task;

						{
							std::unique_lock<std::mutex> lock{ mtx };

							cv.wait(lock, [this]() {
								return m_stop.load(std::memory_order_relaxed)
									|| !m_task.empty();
								});

							// stop requested AND nothing left to do
							if (m_stop.load(std::memory_order_relaxed)
								&& m_task.empty())
							{
								return;
							}

							task = std::move(m_task.front());
							m_task.pop();

							m_active_threads.fetch_add(
								1,
								std::memory_order_relaxed);
						}

						task();

						m_active_threads.fetch_sub(
							1,
							std::memory_order_relaxed);
					}
				});
		}
	}
	
	std::size_t currentAvailableWorkers() {
		return n_workers - m_active_threads.load(std::memory_order_relaxed);
	}

	std::size_t currentActiveThreads() {
		return m_active_threads.load(std::memory_order_relaxed);
	}

	void stop() {
		{
			std::lock_guard<std::mutex> lock(mtx);
			if (m_stop.load(std::memory_order_relaxed)) return;
			m_stop.store(true, std::memory_order_relaxed);
		}

		cv.notify_all();

		for (std::thread& worker : m_thread) {
			if (worker.joinable()) {
				worker.join();
			}
		}
		m_thread.clear();
	}

	template<typename T, typename ... Args> 
	std::shared_future<typename std::invoke_result<T, Args...>::type> queueTask(T&& t, Args&& ... args)
	{
		using returnType = typename std::invoke_result<T, Args...>::type;

		// auto func = std::forward<T>(t);
		// auto arguments = std::make_tuple(std::forward<Args>(args)...);

		std::shared_ptr<std::packaged_task<returnType()>> task = 
			std::make_shared<std::packaged_task<returnType()>>(
				[func = std::forward<T>(t),
				arguments = std::make_tuple(args...)]() mutable
				{
					return std::apply(func, arguments);
				}
			);

		std::shared_future<returnType> result = task->get_future().share();

		{
			std::lock_guard<std::mutex> lock(mtx);

			if (m_stop.load(std::memory_order_relaxed)) {
				throw std::runtime_error(
					"Cannot enqueue on stopped ThreadPool");
			}

			m_task.emplace([task]() {
				(*task)();
				});
		}

		cv.notify_one();

		return result;
	}
};

#endif