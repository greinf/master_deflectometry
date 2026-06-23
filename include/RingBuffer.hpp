#ifndef RINGBUFFER_HPP
#define RINGBUFFER_HPP


#include <iterator>
#include <opencv2/opencv.hpp>
#include <new>
#include <atomic>
// Ring Buffer Class
// Buffer size Hardcoded to 10 Frames

struct RawData {
	uchar* data_ptr{nullptr};
	std::size_t bytes{ };  // for Full Frames 
	int height{};
	int width{};
	std::mutex mtx{};
	bool free{ true };

	~RawData() {
		delete[] data_ptr;
	}
	void clear() {
		height = 0;
		width = 0;
		free = true;
	}
};

class RingBuffer
{
public:
	bool extractData(
		const VmbCPP::FramePtr& pFrame,
		const std::size_t index);

	RingBuffer(
		const std::size_t buffer_size, 
		const std::size_t payload);

	cv::Mat extractfromRing();

	std::size_t m_size{ 0 };

	std::atomic<std::size_t> m_counter{0};

	~RingBuffer();

	// Safety this class should not be copied.
	RingBuffer(const RingBuffer&) = delete;
	RingBuffer(RingBuffer&&) = delete;
	RingBuffer& operator=(const RingBuffer&) = delete;
	RingBuffer& operator=(RingBuffer&&) = delete;

	void start() { m_started = true; }

private:

	bool m_started{ false };

	bool out_working{ false };

	std::vector<RawData> m_raw{};

	std::atomic<int> m_last_success{ -1 };
	
	void startUp(
		const std::size_t buffer_size,
		const std::size_t payload_per_buf
	);
	
	RawData m_out{};
};

inline RingBuffer::RingBuffer(
	const std::size_t buffer_size,
	const std::size_t payload_per_buf)
	:m_size{ buffer_size }
{
	startUp(buffer_size, payload_per_buf);
	std::cout << "Ring Buffer Memory allocated \n";
}

inline RingBuffer::~RingBuffer() {}

inline void RingBuffer::startUp(
	const std::size_t buffer_size,
	const std::size_t payload_per_buf)
{
	assert(m_raw.empty() && "Container must be empty \n");
	// Create the out RawData one time
	m_out.data_ptr = new uchar[payload_per_buf];

	m_out.bytes = payload_per_buf;

	m_raw = std::vector<RawData>(buffer_size);

	for (auto& raw : m_raw) {
		raw.data_ptr = new uchar[payload_per_buf];
		raw.bytes = payload_per_buf;
	}
}

inline bool RingBuffer::extractData(
	const VmbCPP::FramePtr& pFrame,
	const std::size_t index)
{
	if (index >= m_raw.size()) {
		std::cout << "Index RingBuffer out of bounds \n";
		std::cout << "index " << index << '\n';
		return false;
	}

	std::scoped_lock lock(m_raw[index].mtx);

	//std::cout << "Extract Data " << '\n' <<
	//	"Index " << index << '\n';

	VmbUint32_t bufferSize = 0;
	if (pFrame->GetBufferSize(bufferSize) != VmbErrorSuccess || bufferSize == 0)
		throw std::runtime_error("GetBufferSize failed");

	// slot capacity should match bufferSize (PayloadSize) OR be >= imageSize
	if (static_cast<std::size_t>(bufferSize) != m_raw[index].bytes)
		throw std::runtime_error("Slot size mismatch");

	VmbUint32_t width = 0, height = 0;
	if (pFrame ->GetWidth(width) != VmbErrorSuccess || width == 0) return false;
	if (pFrame -> GetHeight(height) != VmbErrorSuccess || height == 0) return false;
	
	m_raw[index].height = static_cast<int>(height);
	m_raw[index].width = static_cast<int>(width);
	
	uchar* src_data_ptr;
	if (pFrame->GetImage(src_data_ptr) != VmbErrorSuccess || src_data_ptr == nullptr)
		return false;

	// copy only what actually arrived
	std::memcpy(m_raw[index].data_ptr, src_data_ptr, bufferSize);

	m_last_success.store(static_cast<int>(index),
		std::memory_order_release);

	return true;
}

inline cv::Mat RingBuffer::extractfromRing()
{
	int i = m_last_success.load(std::memory_order_acquire);
	if (i < 0) return{};

	const std::size_t index{ static_cast<std::size_t>(i) };
	
	//std::cout << "Convert and Out " << '\n' << "Instance " <<
	//	"Index " << index << '\n';


	cv::Mat img;
	{
		std::scoped_lock lock(m_raw[index].mtx, m_out.mtx);

		m_out.free = false;
		std::memcpy(m_out.data_ptr, m_raw[index].data_ptr, m_raw[index].bytes);
		m_out.bytes = m_raw[index].bytes;
		m_out.height = m_raw[index].height;
		m_out.width = m_raw[index].width;

		img = cv::Mat(m_out.height, m_out.width, CV_8UC1, m_out.data_ptr).clone();

		m_out.free = true;
	}

	//std::cout << "Succesfull index extraction at " << i << '\n';

	return img;
}


#endif