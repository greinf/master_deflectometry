// Source - https://stackoverflow.com/q/68727478
// Posted by ChrisZZ
// Retrieved 2025-12-13, License - CC BY-SA 4.0

// my own implementation of parallel_for_, copied from OpenCV source code
void my_parallel_for_(const cv::Range& range, const cv::ParallelLoopBody& body)
{
    #pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int i = range.start; i < range.end; ++i)
        body(cv::Range(i, i + 1));
}

// The functor that performs nearest neighbor resizing, copied from opencv source
class resizeNNInvoker : public cv::ParallelLoopBody
{
public:
    resizeNNInvoker(const cv::Mat& _src, cv::Mat &_dst, int *_x_ofs, double _ify) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs),
        ify(_ify)
    {
    }

    virtual void operator() (const cv::Range& range) const CV_OVERRIDE
    {
        //printf("--- resizeNNInvoker get called\n");
        cv::Size ssize = src.size(), dsize = dst.size();
        int y, x, pix_size = (int)src.elemSize();

        for( y = range.start; y < range.end; y++ )
        {
            uchar* D = dst.data + dst.step*y;
            int sy = std::min(cvFloor(y*ify), ssize.height-1);
            const uchar* S = src.ptr(sy);

            switch( pix_size )
            {
            case 1:
                for( x = 0; x <= dsize.width - 2; x += 2 )
                {
                    uchar t0 = S[x_ofs[x]];
                    uchar t1 = S[x_ofs[x+1]];
                    D[x] = t0;
                    D[x+1] = t1;
                }

                for( ; x < dsize.width; x++ )
                    D[x] = S[x_ofs[x]];
                break;
            case 2:
                for( x = 0; x < dsize.width; x++ )
                    *(ushort*)(D + x*2) = *(ushort*)(S + x_ofs[x]);
                break;
            case 3:
                for( x = 0; x < dsize.width; x++, D += 3 )
                {
                    const uchar* _tS = S + x_ofs[x];
                    D[0] = _tS[0]; D[1] = _tS[1]; D[2] = _tS[2];
                }
                break;
            case 4:
                for( x = 0; x < dsize.width; x++ )
                    *(int*)(D + x*4) = *(int*)(S + x_ofs[x]);
                break;
            case 6:
                for( x = 0; x < dsize.width; x++, D += 6 )
                {
                    const ushort* _tS = (const ushort*)(S + x_ofs[x]);
                    ushort* _tD = (ushort*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            case 8:
                for( x = 0; x < dsize.width; x++, D += 8 )
                {
                    const int* _tS = (const int*)(S + x_ofs[x]);
                    int* _tD = (int*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1];
                }
                break;
            case 12:
                for( x = 0; x < dsize.width; x++, D += 12 )
                {
                    const int* _tS = (const int*)(S + x_ofs[x]);
                    int* _tD = (int*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            default:
                for( x = 0; x < dsize.width; x++, D += pix_size )
                {
                    const uchar* _tS = S + x_ofs[x];
                    for (int k = 0; k < pix_size; k++)
                        D[k] = _tS[k];
                }
            }
        }
    }

private:
    const cv::Mat& src;
    cv::Mat& dst;
    int* x_ofs;
    double ify;

    resizeNNInvoker(const resizeNNInvoker&);
    resizeNNInvoker& operator=(const resizeNNInvoker&);
};

// The entry function that calls nearest neighbor resizing with openmp multi-thread
void resize_nearest(const uchar* src_buf, int src_height, int src_width, int src_linebytes, uchar* dst_buf, int dst_height, int dst_width, int dst_linebytes, const Option& )
{
    cv::Size src_size;
    src_size.height = src_height;
    src_size.width = src_width;
    cv::Mat src(src_size, CV_8UC3, const_cast<uchar*>(src_buf));

    cv::Size dst_size;
    dst_size.height = dst_height;
    dst_size.width = dst_width;
    cv::Mat dst(dst_size, CV_8UC3, dst_buf);

    cv::Size ssize = src.size(), dsize = dst.size();

    double inv_scale_x = (double)dsize.width/ssize.width;
    double inv_scale_y = (double)dsize.height/ssize.height;
    double fx = inv_scale_x;
    double fy = inv_scale_y;

    cv::AutoBuffer<int> _x_ofs(dsize.width);
    int* x_ofs = _x_ofs.data();
    int pix_size = (int)src.elemSize();
    double ifx = 1./fx, ify = 1./fy;
    int x;

    for( x = 0; x < dsize.width; x++ )
    {
        int sx = cvFloor(x*ifx);
        x_ofs[x] = std::min(sx, ssize.width-1)*pix_size;
    }

    cv::Range range(0, dsize.height);

    // !! define the instance of resizeNNInvoker functor.
    resizeNNInvoker invoker(src, dst, x_ofs, ify);

#if 0
    cv::parallel_for_(range, invoker);   //!! use opencv's, cost 3.24 ms
#elif 0
    my_parallel_for_(range, invoker);    //!! use own implementation, cost 7.67 ms
#else
    set_omp_dynamic(1);    //!! use inplace-implementation, cost 7.75 ms
    cv::Range stripeRange = range;
    #pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int i = stripeRange.start; i < stripeRange.end; ++i)
        invoker(cv::Range(i, i + 1));
#endif
}
