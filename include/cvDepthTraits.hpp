#ifndef CVDEPTHTRAITS_H
#define CVDEPTHTRAITS_H
#include <opencv2/opencv.hpp>


template<typename T>
struct is_vec_or_array : std::false_type {};

template<typename T, typename alloc>
struct is_vec_or_array<std::vector<T, alloc>> : std::true_type {};

template<typename T, std::size_t N>
struct is_vec_or_array<std::array<T, N>> : std::true_type {};


#endif