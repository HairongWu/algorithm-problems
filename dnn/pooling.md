# POOLING

```cpp

/*
max pooling corner func, left/right/top/bottom
*/
template <typename T>
void MaxPoolingCorner(const T* src, long iw, long ih, T* dst, long ow, long kw, long kh, long stride_w, long stride_h,
                      long pad_w, long pad_h, long l, long r, long t, long b) {
    for (long oy = t; oy < b; ++oy) {
        for (long ox = l; ox < r; ++ox) {
            Float4 vmax(-FLT_MAX);

            const long srcOriginX = ox * stride_w - pad_w;
            const long srcOriginY = oy * stride_h - pad_h;
            const long kxs        = MAX(0, -srcOriginX);
            const long kxe        = MIN(kw, iw - srcOriginX);
            const long kys        = MAX(0, -srcOriginY);
            const long kye        = MIN(kh, ih - srcOriginY);
            const auto src_ptr    = src + (srcOriginY * iw + srcOriginX) * 4;
            auto dst_ptr          = dst + (oy * ow + ox) * 4;

            for (long ky = kys; ky < kye; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * 4;
                for (long kx = kxs; kx < kxe; kx++) {
                    vmax = Float4::max(vmax, Float4::load(src_ptr_h + kx * 4));
                }
            }

            Float4::save(dst_ptr, vmax);
        }
    }
}

/*
max pooling 3x3s2 kernel
*/
template <typename T>
void MaxPoolingCenter3x3s2(const T* src, long iw, long ih, T* dst, long ow, long oh, long pad_w, long pad_h, long l,
                           long r, long t, long b) {
    for (long oy = t; oy < b; ++oy) {
        for (long ox = l; ox < r; ++ox) {
            Float4 vmax(-FLT_MAX);

            const long src_offset_x = ox * 2 - pad_w;
            const long src_offset_y = oy * 2 - pad_h;
            const auto src_ptr      = src + (src_offset_y * iw + src_offset_x) * 4;
            auto dst_ptr            = dst + (oy * ow + ox) * 4;

            for (long ky = 0; ky < 3; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * 4;
                vmax                 = Float4::max(vmax, Float4::load(src_ptr_h + 0 * 4));
                vmax                 = Float4::max(vmax, Float4::load(src_ptr_h + 1 * 4));
                vmax                 = Float4::max(vmax, Float4::load(src_ptr_h + 2 * 4));
            }
            Float4::save(dst_ptr, vmax);
        }
    }
}

/*
general max pooling center kernel
*/
template <typename T>
void MaxPoolingCenter(const T* src, long iw, long ih, T* dst, long ow, long oh, long kw, long kh, long stride_w,
                      long stride_h, long pad_w, long pad_h, long l, long r, long t, long b) {
    for (long oy = t; oy < b; ++oy) {
        for (long ox = l; ox < r; ++ox) {
            Float4 vmax(-FLT_MAX);

            const long src_offset_x = ox * stride_w - pad_w;
            const long src_offset_y = oy * stride_h - pad_h;
            const auto src_ptr      = src + (src_offset_y * iw + src_offset_x) * 4;
            auto dst_ptr            = dst + (oy * ow + ox) * 4;

            for (long ky = 0; ky < kh; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * 4;
                for (long kx = 0; kx < kw; kx++) {
                    vmax = Float4::max(vmax, Float4::load(src_ptr_h + kx * 4));
                }
            }

            Float4::save(dst_ptr, vmax);
        }
    }
}

/*
max pooling func, process four corners and center
*/
template <typename T>
void MaxPooling(const T* src, long iw, long ih, T* dst, long ow, long oh, long kw, long kh, long stride_w,
                long stride_h, long pad_w, long pad_h, long l, long r, long t, long b) {
    // top corner
    MaxPoolingCorner<T>(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, 0, ow, 0, t);
    if (kw == 3 && kh == 3 && stride_h == 2 && stride_w == 2) {
        MaxPoolingCenter3x3s2<T>(src, iw, ih, dst, ow, oh, pad_w, pad_h, l, r, t, b);
    } else {
        MaxPoolingCenter<T>(src, iw, ih, dst, ow, oh, kw, kh, stride_w, stride_h, pad_w, pad_h, l, r, t, b);
    }

    // bottom corner
    MaxPoolingCorner<T>(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, 0, ow, b, oh);
    // left corner
    MaxPoolingCorner<T>(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, 0, l, t, b);
    // right corner
    MaxPoolingCorner<T>(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, r, ow, t, b);
}
/*
general avg pooling func
*/
template <typename T>
void AvgPooling(const T* src, long iw, long ih, T* dst, long ow, long oh, long kw, long kh, long stride_w,
                long stride_h, long pad_w, long pad_h) {
    for (long oy = 0; oy < oh; ++oy) {
        for (long ox = 0; ox < ow; ++ox) {
            Float4 vavg(0.f);

            const long srcOriginX    = ox * stride_w - pad_w;
            const long srcOriginY    = oy * stride_h - pad_h;
            const long kxs           = MAX(0, -srcOriginX);
            const long kxe           = MIN(kw, iw - srcOriginX);
            const long kys           = MAX(0, -srcOriginY);
            const long kye           = MIN(kh, ih - srcOriginY);
            const float kernel_count = 1.0 / ((kxe - kxs) * (kye - kys));
            const auto src_ptr       = src + (srcOriginY * iw + srcOriginX) * 4;
            auto dst_ptr             = dst + (oy * ow + ox) * 4;

            for (long ky = kys; ky < kye; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * 4;
                for (long kx = kxs; kx < kxe; kx++) {
                    vavg = vavg + Float4::load(src_ptr_h + kx * 4);
                }
            }

            vavg = vavg * Float4(kernel_count);
            Float4::save(dst_ptr, vavg);
        }
    }
}
/*
general max pooling int8 kernel
*/
void MaxPoolingINT8(const int8_t* src, long iw, long ih, int8_t* dst, long ow, long oh, long c_r4, long kw, long kh,
                    long stride_w, long stride_h, long pad_w, long pad_h) {
    OMP_PARALLEL_FOR_GUIDED_
    for (long oy = 0; oy < oh; ++oy) {
        for (long ox = 0; ox < ow; ++ox) {
            const long srcOriginX = ox * stride_w - pad_w;
            const long srcOriginY = oy * stride_h - pad_h;
            const long kxs        = MAX(0, -srcOriginX);
            const long kxe        = MIN(kw, iw - srcOriginX);
            const long kys        = MAX(0, -srcOriginY);
            const long kye        = MIN(kh, ih - srcOriginY);
            long oc               = 0;
#ifdef TNN_USE_NEON
            for (; oc < c_r4 - 4; oc += 8) {
                const auto src_ptr = src + (srcOriginY * iw + srcOriginX) * c_r4 + oc;
                auto dst_ptr       = dst + (oy * ow + ox) * c_r4 + oc;
                int8x8_t max_reg   = vdup_n_s8(-127);
                // find kernel_w * kernel_h max value
                for (long ky = kys; ky < kye; ++ky) {
                    const auto src_ptr_h = src_ptr + (ky * iw) * c_r4;
                    long kx              = kxs;
                    for (; kx < kxe; kx++) {
                        const auto srcPtrStart = src_ptr_h + kx * c_r4;
                        max_reg                = vmax_s8(max_reg, vld1_s8(srcPtrStart));
                    }
                }
                vst1_s8(dst_ptr, max_reg);
            }
#endif
            for (; oc < c_r4; oc += 4) {
                int8_t maxValue[4] = {-127, -127, -127, -127};
                const auto src_ptr = src + (srcOriginY * iw + srcOriginX) * c_r4 + oc;
                auto dst_ptr       = dst + (oy * ow + ox) * c_r4 + oc;
                // find kernel_w * kernel_h max value
                for (long ky = kys; ky < kye; ++ky) {
                    const auto src_ptr_h = src_ptr + (ky * iw) * c_r4;
                    long kx              = kxs;
                    for (; kx < kxe; ++kx) {
                        const auto srcPtrStart = src_ptr_h + kx * c_r4;
                        for (long j = 0; j < 4; ++j) {
                            maxValue[j] = MAX(maxValue[j], srcPtrStart[j]);
                        }
                    }
                }
                // output
                *(int32_t*)dst_ptr = *(int32_t*)maxValue;
            }
        }
    }
}

/*
general avg pooling int8 kernel
*/
void AvgPoolingINT8(const int8_t* src, long iw, long ih, int8_t* dst, long ow, long oh, long c_r4, long kw, long kh,
                    long stride_w, long stride_h, long pad_w, long pad_h) {
    for (long oy = 0; oy < oh; ++oy) {
        for (long ox = 0; ox < ow; ++ox) {
            const long srcOriginX   = ox * stride_w - pad_w;
            const long srcOriginY   = oy * stride_h - pad_h;
            const long kxs          = MAX(0, -srcOriginX);
            const long kxe          = MIN(kw, iw - srcOriginX);
            const long kys          = MAX(0, -srcOriginY);
            const long kye          = MIN(kh, ih - srcOriginY);
            const long kernel_count = (kxe - kxs) * (kye - kys);
            long oc                 = 0;
#ifdef TNN_USE_NEON
            int16_t sum[8];
            for (; oc < c_r4 - 4; oc += 8) {
                int16x8_t avg_reg  = vdupq_n_s16(0);
                const auto src_ptr = src + (srcOriginY * iw + srcOriginX) * c_r4 + oc;
                auto dst_ptr       = dst + (oy * ow + ox) * c_r4 + oc;
                // find kernel_w * kernel_h avg value
                for (long ky = kys; ky < kye; ++ky) {
                    const auto src_ptr_h = src_ptr + (ky * iw) * c_r4;
                    long kx              = kxs;
                    for (; kx < kxe; kx++) {
                        const auto srcPtrStart = src_ptr_h + kx * c_r4;
                        int16x8_t cur_val      = vmovl_s8(vld1_s8(srcPtrStart));
                        avg_reg                = vaddq_s16(avg_reg, cur_val);
                    }
                }
                vst1q_s16(sum, avg_reg);
                for (long j = 0; j < 8; j++) {
                    dst_ptr[j] = sum[j] / kernel_count;
                }
            }
#endif
            for (; oc < c_r4; oc += 4) {
                int16_t sum[4]     = {0, 0, 0, 0};
                const auto src_ptr = src + (srcOriginY * iw + srcOriginX) * c_r4 + oc;
                auto dst_ptr       = dst + (oy * ow + ox) * c_r4 + oc;
                // find kernel_w * kernel_h avg value
                for (long ky = kys; ky < kye; ++ky) {
                    const auto src_ptr_h = src_ptr + (ky * iw) * c_r4;
                    long kx              = kxs;

                    for (; kx < kxe; ++kx) {
                        const auto srcPtrStart = src_ptr_h + kx * c_r4;
                        for (long j = 0; j < 4; ++j) {
                            sum[j] += srcPtrStart[j];
                        }
                    }
                }
                // output
                for (long j = 0; j < 4; j++) {
                    dst_ptr[j] = static_cast<int8_t>(sum[j] / kernel_count);
                }
            }
        }
    }
}
Status arm_neon_pooling(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs, std::vector<int> dims_input, std::vector<int> dims_output, int data_type, int pool_type) {
    auto oc_4       = UP_DIV(dims_output[1], 4);
    auto batch      = dims_output[0];

    // run
    if (data_type == DATA_TYPE_FLOAT) {
        auto input_plane_stride  = 4 * k_param_->iw * k_param_->ih;
        auto output_plane_stride = 4 * k_param_->ow * k_param_->oh;

        for (int plane = (int)0; plane < batch * oc_4; plane++) {
            if (pool_type == 0) {
                MaxPooling(reinterpret_cast<float *>(input_ptr) + plane * input_plane_stride, k_param_->iw,
                           k_param_->ih, reinterpret_cast<float *>(output_ptr) + output_plane_stride * plane,
                           k_param_->ow, k_param_->oh, param->kernels[0], param->kernels[1], param->strides[0],
                           param->strides[1], param->pads[0], param->pads[2], corner_l_, corner_r_, corner_t_,
                           corner_b_);
            } else {
                AvgPooling(reinterpret_cast<float *>(input_ptr) + plane * input_plane_stride, k_param_->iw,
                           k_param_->ih, reinterpret_cast<float *>(output_ptr) + output_plane_stride * plane,
                           k_param_->ow, k_param_->oh, param->kernels[0], param->kernels[1], param->strides[0],
                           param->strides[1], param->pads[0], param->pads[2]);
            }
        }
    } else if (data_type == DATA_TYPE_BFP16) {
        auto input_plane_stride  = 4 * k_param_->iw * k_param_->ih;
        auto output_plane_stride = 4 * k_param_->ow * k_param_->oh;

        for (int plane = (int)0; plane < batch * oc_4; plane++) {
            if (pool_type == 0) {
                MaxPooling(reinterpret_cast<bfp16_t *>(input_ptr) + plane * input_plane_stride, k_param_->iw,
                           k_param_->ih, reinterpret_cast<bfp16_t *>(output_ptr) + output_plane_stride * plane,
                           k_param_->ow, k_param_->oh, param->kernels[0], param->kernels[1], param->strides[0],
                           param->strides[1], param->pads[0], param->pads[2], corner_l_, corner_r_, corner_t_,
                           corner_b_);
            } else {
                AvgPooling(reinterpret_cast<bfp16_t *>(input_ptr) + plane * input_plane_stride, k_param_->iw,
                           k_param_->ih, reinterpret_cast<bfp16_t *>(output_ptr) + output_plane_stride * plane,
                           k_param_->ow, k_param_->oh, param->kernels[0], param->kernels[1], param->strides[0],
                           param->strides[1], param->pads[0], param->pads[2]);
            }
        }
    } else {
        // INT8
        for (int n = 0; n < batch; n++) {
            auto input_batch_stride  = k_param_->iw * k_param_->ih * oc_4 * 4;
            auto output_batch_stride = k_param_->ow * k_param_->oh * oc_4 * 4;
            if (pool_type == 0) {
                MaxPoolingINT8(reinterpret_cast<int8_t *>(input_ptr) + n * input_batch_stride, k_param_->iw,
                               k_param_->ih, reinterpret_cast<int8_t *>(output_ptr) + n * output_batch_stride,
                               k_param_->ow, k_param_->oh, oc_4 * 4, param->kernels[0], param->kernels[1],
                               param->strides[0], param->strides[1], param->pads[0], param->pads[2]);
            } else {
                AvgPoolingINT8(reinterpret_cast<int8_t *>(input_ptr) + n * input_batch_stride, k_param_->iw,
                               k_param_->ih, reinterpret_cast<int8_t *>(output_ptr) + n * output_batch_stride,
                               k_param_->ow, k_param_->oh, oc_4 * 4, param->kernels[0], param->kernels[1],
                               param->strides[0], param->strides[1], param->pads[0], param->pads[2]);
            }
        }
    }

    return TNN_OK;
}
```