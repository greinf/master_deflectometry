#ifndef CVDEPTHTRAITS_H
#define CVDEPTHTRAITS_H
#include <opencv2/opencv.hpp>


/*Compile time traits mapping*/
template<int depth> 
struct CvDepthTraits {};

// using: defines a typemember. These are declared at compile time.
// This lives therefore in the type domain and not value domain. 

template<> struct CvDepthTraits<CV_8U> { using value_type = uchar; };
template<> struct CvDepthTraits<CV_8S> { using value_type = char; };
template<> struct CvDepthTraits<CV_16U> { using value_type = ushort; };
template<> struct CvDepthTraits<CV_16S> { using value_type = short; };
template<> struct CvDepthTraits<CV_32S> { using value_type = int; };
template<> struct CvDepthTraits<CV_32F> { using value_type = float; };
template<> struct CvDepthTraits<CV_64F> { using value_type = double; };

#endif