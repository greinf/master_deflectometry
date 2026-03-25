#ifndef GRAYCODEDECODER_HPP
#define GRAYCODEDECODER_HPP	

class ImageProcessing;
struct GrayCodeConfig;

class GrayCodeDecoder {
	ImageProcessing& m_img_processing;
	

	
public:
	GrayCodeDecoder(ImageProcessing& imgProcess)
		:m_img_processing{imgProcess}
	{ }

	static void grayPoint(GrayCodeConfig&);





};





#endif