#ifndef RINGBUFFER_HPP
#define RINGBUFFER_HPP

#include <iterator>
// Ring Buffer Class
// Buffer size Hardcoded to 10 Frames
class RingBuffer
{
public:
	void getFrame(const VmbCPP::FramePtr& pFrame);

	RingBuffer();

	cv::Mat extract();

	~RingBuffer() = default;

	// Safety this class should not be copied.
	RingBuffer(const RingBuffer&) = delete;
	RingBuffer(RingBuffer&&) = delete;
	RingBuffer& operator=(const RingBuffer&) = delete;
	RingBuffer& operator=(RingBuffer&&) = delete;
private:
	static const std::size_t m_size{ 15 };
	std::vector<cv::Mat> m_ring{};
	// We say that the iterator always should point at one element after the last Frame. 
	std::vector<cv::Mat>::iterator m_iter;
	std::mutex mtx{};

	cv::Mat convertToMat(const VmbCPP::FramePtr& pFrame);
	
	void clear();

	void startUp();

	void allocate(const cv::Mat frame);
};

inline RingBuffer::RingBuffer() {
	startUp();
}

inline void RingBuffer::startUp() {
	// If m_ring also has capacity > 0
	if (m_ring.begin() != m_ring.end()) {
		std::cout << "WARN: Ring Buffer already holds elements which are deleted \n";
	}
	m_ring = std::vector<cv::Mat>(m_size);
	m_iter = m_ring.end();
}

inline void RingBuffer::allocate(const cv::Mat frame) {
	std::lock_guard<std::mutex> locked(mtx);
	if (m_iter == m_ring.end()) {
		m_iter = m_ring.begin();
	}
	(*m_iter) = frame;
	std::advance(m_iter, 1);
	return;
}

inline void RingBuffer::clear() {
	m_ring.clear();
}

inline void RingBuffer::getFrame(const VmbCPP::FramePtr& pFrame) {
	assert(m_ring.size() > 0 && "The ringBuffer must be created");
	cv::Mat image = convertToMat(pFrame);
	allocate(image);
	return;
}

inline cv::Mat RingBuffer::extract() {
	std::lock_guard<std::mutex> lock(mtx);
	return *std::prev(m_iter);
}

inline cv::Mat RingBuffer::convertToMat(const VmbCPP::FramePtr& pFrame) {
	VmbUint32_t width, height;
	pFrame->GetWidth(width);
	pFrame->GetHeight(height);

	VmbUchar_t* pData = nullptr;
	pFrame->GetImage(pData); // Get the raw pointer to the pixel data

	// Assuming the camera is set to Mono8
	return cv::Mat(height, width, CV_8UC1, pData).clone();
}


#endif