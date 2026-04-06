
// Depracted code

/*
// This is the first implementation of a template elipsis. The function takes a arbitrary ammount of cv::MAts and a function pointer.
// Becasue the cv::Mat datatype can not be decued at compile time this functions template calls a second inner function where,
// for the datatypes it is differntiatied.
template<typename func, typename... Mats>
auto ImageProcessing::forEachPixel(func, Mats&&... mats) {
    static_assert(sizeof...(mats) > 0, "Need at least one matrix."); //compile time check
    auto first = std::get<0>(std::forward_as_tuple(std::forward<Mats>(mats)));
    const cv::Size size = first.size();
    const int type = first.type();
    (assert(size == mats.size() && type == mats.type()), ...); //compiler expands at compile time, checkupt at runtime.
    // giving a conditional datatype back, std::is_same<>::value is static function
    // if condition is true give back the first datatype, if false give back the second datatype
    // But mats.size() and mats.type() are runtime functions, therefore static assert does not work.

    // This line does not work becasue in decltype it is assumed that .at<float> for each datatype
    // This line typename std::conditional -> typename is necessary to tell the compiler that a type is named. not a value. wihtout std::conditional<...>::type could be interpreted as static
    // typenmae std::remove_reference -> the function itself is template class and ::type nested type alias.
    // Whenever <T>::type (or something) is used and it is not a static member function we have to use typename becasue the comiler can not now.
    // using T = typename std::conditional<
    //    std::is_same<Func, float(*)(float,float) >>::value, float, typename std::remove_reference<decltype(first.at<float>(0, 0))>::type>::type;

    // typenabhängiger Name.
    int type = mat.type();
    int depth = CV_MAT_DEPTH(type);  //Macros to extract depth (datatype)
    int channels = CV_MAT_CN(type);  //Macros to define extract how manc channels are there

    // type dependend name here, therefore "typename" before. Also this line does not work. becasue type is runtime constant at the
    // template deduction would hapen at comile time the value is not available when the programm runs.
    //using T = typename CvDepthTraits<CV_MAT_DEPTH(type)>::value_type;

    switch (depth) {
    case CV_8U:  return forEachPixelImpl<CV_8U>(func, std::forward<Mats>(mats)...);
    case CV_8S:  return forEachPixelImpl<CV_8S>(func, std::forward<Mats>(mats)...);
    case CV_16U: return forEachPixelImpl<CV_16U>(func, std::forward<Mats>(mats)...);
    case CV_16S: return forEachPixelImpl<CV_16S>(func, std::forward<Mats>(mats)...);
    case CV_32S: return forEachPixelImpl<CV_32S>(func, std::forward<Mats>(mats)...);
    case CV_32F: return forEachPixelImpl<CV_32F>(func, std::forward<Mats>(mats)...);
    case CV_64F: return forEachPixelImpl<CV_64F>(func, std::forward<Mats>(mats)...);
    default:
        throw std::runtime_error("Unsupported depth.");
    }
}


template<int Depth, typename Func, typename... Mats>
cv::Mat ImageProcessing::forEachPixelImpl(const Func& func, Mats&&... mats) {
    using T = typename CvDepthTraits<Depth>::value_type;   //again typename necessary because of ...<dependen>::...
    auto&& first = std::get<0>(std::forward_as_tuple(mats...));

    cv::Mat result(first.size(), first.type());

    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            result.at<T>(y, x) = func(mats.at<T>(y, x)...);
        }
    }
    return result;
}
*/

//void ImageProcessing::saveImages_png(const std::string& path_s) {
//    std::filesystem::path path{ path_s };
//    if (!std::filesystem::exists(path)) {
//        if (!std::filesystem::create_directory(path)) std::cerr << "Creating the directory failed. \n";
//        std::cout << "Created Directory for the .png files \n";
//    }
//    else {
//        std::cout << "Directory for .png files already exists \n" << "Possible override. Do you want to proceed? \n [Y/N]";
//        char c;
//        bool cond{ true };
//        do {
//            std::cin >> c;
//            if (!std::cin.fail()) {
//                std::cin.clear();
//                std::cin.ignore(100, '\n');
//            }
//            if (std::toupper(c) == 'N') {
//                return;
//            }
//            if (std::toupper(c) == 'Y') {
//                cond = false;
//            }
//        } while (cond);
//    }
//    //Create save structure
//    std::array<std::string, 7> strings{ "Full_frames", "Mean_frames", "Base Intensity", "Contrast",
//        "WrappedPhase", "unwrappedPhase", "reprojectionError" };
//
//    for (std::size_t i = 0; i < strings.size(); ++i)
//    {
//        if (!std::filesystem::create_directory(path / strings[i])) {
//            std::cout << "Creating directory " << (path / strings[i]).string() << " failed \n";
//        }
//        switch (i) {
//        case(0): { save_png(path / strings[i], m_frames); break; }
//        case(1): { save_png(path / strings[i], m_raw_phase); break; }
//        case(2): { save_png(path / strings[i], m_baseIntensity); break; }
//        case(3): { save_png(path / strings[i], m_contrast); break; }
//        case(4): { save_png(path / strings[i], m_wrapped_phase); break; }
//        case(5): { save_png(path / strings[i], m_unwrapped_phase); break; }
//        case(6): { save_png(path / strings[i], m_reprojection_error_img); break; }
//
//        }
//    }
//}



////Try to write first allcoator
//template <typename T, std::size_t Alignment = 32>
//struct AlignedAllocator {
//    using value_type = T;
//
//    AlignedAllocator() noexcept = default;
//    template<class U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
//
//    T* allocate(std::size_t n) {
//        void* ptr = nullptr;
//        if (ptr = std::aligned_alloc() != 0)
//            throw std::bad_alloc();
//        return reinterpret_cast<T*>(ptr);
//    }
//
//    void deallocate(T* p, std::size_t) noexcept {
//        free(p);
//    }
//};


//
////Just a small example code to do generic programming with iterators
//template <class InputIt>
//typename std::iterator_traits<InputIt>::difference_type
//distance(InputIt first, InputIt last) {
//    using category = typename std::iterator_traits<InputIt>::iterator_category;
//
//    if constexpr (std::is_same_v<category, std::random_access_iterator_tag>)
//        return last - first;     // fast O(1)
//    else {
//        typename std::iterator_traits<InputIt>::difference_type n = 0;
//        for (; first != last; ++first) ++n;   // slow O(n)
//        return n;
//    }
//}


//void ImageProcessing::goldsteinUnwrap() {
//    cv::Mat unwrapped1;
//    cv::Mat unwrapped2;
//    for (auto& m : m_unwrapped_phase) {
//        cv::normalize(m, m, 0, 1, cv::NORM_MINMAX, CV_32F);
//    }
//    cv::Mat mask1 = (m_contrast[0] > 0.2f); // for example: keep only valid contrast regions
//    mask1.convertTo(mask1, CV_8U);           // ensure binary mask
//    cv::Mat mask2 = (m_contrast[1] > 0.2f);
//    mask2.convertTo(mask2, CV_8U);
//    goldsteinUnwrapCV(m_wrapped_phase[0], unwrapped1, mask1);
//    goldsteinUnwrapCV(m_wrapped_phase[1], unwrapped2, mask2);
//
//    cv::normalize(unwrapped1, m_unwrapped_phase[0], 0, 255, cv::NORM_MINMAX, CV_8U);
//    cv::normalize(unwrapped2, m_unwrapped_phase[1], 0, 255, cv::NORM_MINMAX, CV_8U);
//    cv::imshow("Goldstein Unwrapped Phase 1", m_unwrapped_phase.at(0));
//    cv::imshow("Goldstein Unwrapped Phase 2", m_unwrapped_phase.at(1));
//    cv::waitKey(0);
//}
