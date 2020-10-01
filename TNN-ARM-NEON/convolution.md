# Conv2D layer
This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.

```cpp
void set_block_size(int &a_block, int &b_block, int l2_size, const int plane_num, const int oc_r4, const int ic_r4,
                    int byte_size) {
    const int l1cache = 32 * 1024 / byte_size;
    if (plane_num >= oc_r4) {
        // inner kernel also a first, safe in l1 cache
        a_block = MAX(l1cache / ic_r4 - ARM_SGEMM_TILE_N, 1);
        // b safe in l2 cache
        int l2_size_b = l2_size / ic_r4 - a_block;
        b_block       = MIN(l2_size_b, oc_r4);
    } else {
        if (plane_num < l2_size / ic_r4 - ARM_SGEMM_TILE_N) {
            a_block = plane_num;
        } else {
            a_block = MAX(l2_size / ic_r4 - ARM_SGEMM_TILE_N, 1);
        }
        b_block = ARM_SGEMM_TILE_N;
    }
    b_block = ROUND_UP(b_block, ARM_SGEMM_TILE_N);
    a_block = ROUND_UP(a_block, ARM_SGEMM_TILE_M);
}

/*
pack func, can be treated as img2col in conv1x1
*/
template <typename T>
static void PackLine(T *dst, T *src, int ih, int iw, int oh, int ow, int c_r4, int pad_h, int pad_w, int stride_h,
                     int stride_w) {
    if (pad_h != 0 || pad_w != 0)
        memset(dst, 0, c_r4 * oh * ow * sizeof(T));

    dst += pad_h * ow * 4 + pad_w * 4;

    for (int c = 0; c < c_r4; c += 4) {
        auto dst_c = dst + c * oh * ow;
        auto src_c = src + c * ih * iw;
        if (stride_w == 1 && stride_h == 1) {
            for (int h = 0; h < ih; h++) {
                memcpy(dst_c + h * ow * 4, src_c + h * iw * 4, iw * 4 * sizeof(T));
            }
        } else if (pad_w == 0 && pad_h == 0) {
            for (int h = 0; h < oh; h++) {
                auto dst_h = dst_c + h * ow * 4;
                auto src_h = src_c + h * stride_h * iw * 4;
                for (int w = 0; w < ow; w++) {
                    Float4::save(dst_h + w * 4, Float4::load(src_h + w * stride_w * 4));
                }
            }
        } else {
            Float4 zeros(0);
            for (int h = 0; h < oh; h++) {
                int sh = h * stride_h - pad_h;
                if (sh >= 0 && sh < ih) {
                    auto dst_h = dst_c + (h - pad_h) * ow * 4;
                    auto src_h = src_c + (h * stride_h - pad_h) * iw * 4;
                    for (int w = 0; w < ow; w++) {
                        int sw = w * stride_w - pad_w;
                        if (sw >= 0 && sw < iw) {
                            Float4::save(dst_h + (w - pad_w) * 4, Float4::load(src_h + (w * stride_w - pad_w) * 4));
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
Status arm_neon_conv_1x1(const T* inputs, const T* outputs, std::vector<int> dims_input, std::vector<int> dims_output, T* tmp_dst,float* buffer_weight_, float* bias_ptr, int activation_type) {
    const int batch = dims_output[0];
    auto ic4        = UP_DIV(dims_input[1], 4);
    auto oc4        = UP_DIV(dims_output[1], 4);

    int src_z_step = k_param_->iw * k_param_->ih * 4;
    int dst_z_step = k_param_->ow * k_param_->oh * 4;
    int plane_num  = k_param_->ow * k_param_->oh;

    /*
    get a_block & b_block based on l2 cache size(512K most of the time)
    */
    int a_block, b_block;
    set_block_size(a_block, b_block, 512 * 1024 / data_byte_size, plane_num, oc4 * 4, ic4 * 4, data_byte_size);
    /*
    pack inputs when pads or strides are not equal to one
    */
    if ((k_param_->ih != k_param_->oh) || (k_param_->iw != k_param_->ow)) {
        work_space_size += ic4 * 4 * dims_output[2] * dims_output[3] * data_byte_size;
        work_space   = tmp_dst + ic4 * 4 * dims_output[2] * dims_output[3];

        PackLine(tmp_dst, src_origin, k_param_->ih, k_param_->iw, k_param_->oh, k_param_->ow, k_param_->ic_r4,
                 conv_param->pads[2], conv_param->pads[0], conv_param->strides[1], conv_param->strides[0]);
        src_origin = tmp_dst;
    }

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto input_ptr  = src_origin + batch_idx * k_param_->iw * k_param_->ih * ROUND_UP(dims_input[1], 4);
        auto output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * ROUND_UP(dims_output[1], 4);

        /*
        call different sgemm func based on input and weight size
        */
        if (plane_num > oc4 * 4) {
            sgemm_repack_lhs(output_ptr, input_ptr, buffer_weight_, ic4, oc4, plane_num, dst_z_step,
                             a_block, b_block, work_space, bias_ptr, activation_type);
        } else {
            sgemm_repack_rhs(output_ptr, input_ptr, buffer_weight_, ic4, oc4, plane_num, dst_z_step,
                             a_block, b_block, work_space, bias_ptr, activation_type);
        }
    }

    return TNN_OK;
}


template <typename T>
Status arm_neon_conv_3x3(const T* inputs, const T* outputs, std::vector<int> dims_input, std::vector<int> dims_output) {
    const int batch = dims_output[0];

    auto w_unit      = UP_DIV(k_param_->ow, dst_unit_);
    auto h_unit      = UP_DIV(k_param_->oh, dst_unit_);
    auto title_count = UP_DIV(w_unit * h_unit, ARM_SGEMM_TILE_M);

    T *src_origin = input;
    T *dst_origin = output;

    int max_num_threads          = OMP_MAX_THREADS_NUM_;
    int transform_num_per_thread = src_unit_ * src_unit_ * 4;
    int work_num_per_thread      = (k_param_->ic_r4 * 2 + k_param_->oc_r4) * src_unit_ * src_unit_ * ARM_SGEMM_TILE_M;

    auto tranform_buf_size = max_num_threads * transform_num_per_thread * sizeof(float);
    auto work_buf_size     = work_num_per_thread * sizeof(float);

    // gemm kernel need bias pointer
    auto fake_bias_size = k_param_->oc_r4 * sizeof(float);
    float *work_sapce   = reinterpret_cast<float *>(
        context_->GetSharedWorkSpace(tranform_buf_size + work_buf_size + fake_bias_size + NEON_KERNEL_EXTRA_LOAD));
    float *fake_bias    = reinterpret_cast<float *>(work_sapce);
    T *transform_buffer = reinterpret_cast<T *>(work_sapce + fake_bias_size / sizeof(float));
    work_sapce += tranform_buf_size / sizeof(float) + fake_bias_size / sizeof(float);

    // memset fake bias data to get correct results
    memset(fake_bias, 0, fake_bias_size);

    if (DstTransformFunc_ == nullptr || SrcTransformFunc_ == nullptr) {
        return TNNERR_COMMON_ERROR;
    }

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto input_ptr  = src_origin + batch_idx * k_param_->iw * k_param_->ih * k_param_->ic_r4;
        auto output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r4;

        for (int t_idx = 0; t_idx < title_count; t_idx++) {
            auto _src_origin = work_sapce;
            auto _dst_origin = _src_origin + k_param_->ic_r4 * src_unit_ * src_unit_ * ARM_SGEMM_TILE_M;
            auto repack_buf  = _dst_origin + k_param_->oc_r4 * src_unit_ * src_unit_ * ARM_SGEMM_TILE_M;

            int x_idx    = t_idx * ARM_SGEMM_TILE_M;
            int x_remain = w_unit * h_unit - x_idx;
            int x_c      = x_remain > ARM_SGEMM_TILE_M ? ARM_SGEMM_TILE_M : x_remain;

            int src_z_step = k_param_->iw * k_param_->ih * 4;
            int dst_z_step = x_c * src_unit_ * src_unit_ * 4;

            OMP_PARALLEL_FOR_
            for (int z = 0; z < k_param_->ic_r4 / 4; z++) {
                int tid         = OMP_TID_;
                auto mid_buffer = transform_buffer + tid * transform_num_per_thread;
                auto src_z      = input_ptr + z * src_z_step;
                auto dst_z      = _src_origin + z * dst_z_step;
                for (int x_i = 0; x_i < x_c; x_i++) {
                    int idx   = x_idx + x_i;
                    int w_idx = idx % w_unit;
                    int h_idx = idx / w_unit;

                    int src_x = w_idx * dst_unit_ - conv_param->pads[0];
                    int src_y = h_idx * dst_unit_ - conv_param->pads[2];
                    int sy    = MAX(0, src_y) - src_y;
                    int ey    = MIN(src_y + src_unit_, k_param_->ih) - src_y;
                    int sx    = MAX(0, src_x) - src_x;
                    int ex    = MIN(src_x + src_unit_, k_param_->iw) - src_x;
                    int count = (ex - sx) * 4;

                    // source transform start
                    auto src_start       = src_z + (src_x + src_y * k_param_->iw) * 4;
                    auto dst_start       = dst_z + x_i * src_unit_ * src_unit_ * 4;
                    T *transform_src     = nullptr;
                    float *transform_dst = dst_start;
                    int h_stride0        = 0;
                    int h_stride1        = 4 * src_unit_;

                    if (ex - sx == src_unit_ && ey - sy == src_unit_) {
                        transform_src = src_start;
                        h_stride0     = 4 * k_param_->iw;
                    } else {
                        memset(mid_buffer, 0, src_unit_ * src_unit_ * 4 * data_byte_size);
                        if (count > 0) {
                            for (int yy = sy; yy < ey; yy++) {
                                auto dst_yy = mid_buffer + yy * src_unit_ * 4 + sx * 4;
                                auto src_yy = src_start + 4 * k_param_->iw * yy + sx * 4;
                                memcpy(dst_yy, src_yy, count * data_byte_size);
                            }
                        }

                        transform_src = mid_buffer;
                        h_stride0     = 4 * src_unit_;
                    }

                    SrcTransformFunc_(transform_src, transform_dst, 4, h_stride0);
                    // source transform end
                }

                /*
                repack data format to nchw for gemm func
                total data num : ic * tile * unit * unit
                */
                auto repack_z = repack_buf + 4 * x_c * z;
                for (int i = 0; i < src_unit_ * src_unit_; i++) {
                    auto repack_dst = repack_z + i * x_c * k_param_->ic_r4;
                    auto repack_src = dst_z + i * 4;
                    load_repack(repack_dst, repack_src, x_c, src_unit_ * src_unit_ * 4);
                }
            }

            // gemm multi (n8 for armv8, n4 for armv7)
            OMP_PARALLEL_FOR_
            for (int i = 0; i < src_unit_ * src_unit_; i++) {
                GEMM_FUNC(_dst_origin + i * 4 * x_c, repack_buf + i * k_param_->ic_r4 * x_c,
                          reinterpret_cast<float *>(k_param_->fil_ptr) + i * k_param_->ic_r4 * k_param_->oc_r4,
                          k_param_->ic_r4 / 4, x_c * src_unit_ * src_unit_ * 4, k_param_->oc_r4 / 4, x_c, fake_bias, 0);
            }

            src_z_step = x_c * src_unit_ * src_unit_ * 4;
            dst_z_step = k_param_->ow * k_param_->oh * 4;

            OMP_PARALLEL_FOR_
            for (int z = 0; z < k_param_->oc_r4 / 4; z++) {
                int tid         = OMP_TID_;
                auto mid_buffer = transform_buffer + tid * transform_num_per_thread;
                auto src_z      = _dst_origin + z * src_z_step;
                auto dst_z      = output_ptr + z * dst_z_step;
                for (int x_i = 0; x_i < x_c; x_i++) {
                    int idx   = x_idx + x_i;
                    int w_idx = idx % w_unit;
                    int h_idx = idx / w_unit;

                    int dst_x = w_idx * dst_unit_;
                    int dst_y = h_idx * dst_unit_;
                    int ey    = MIN(dst_y + dst_unit_, k_param_->oh) - dst_y;
                    int ex    = MIN(dst_x + dst_unit_, k_param_->ow) - dst_x;

                    int count = ex * 4;
                    // dst transform start
                    auto src_start       = src_z + x_i * 4;
                    auto dst_start       = dst_z + 4 * (dst_x + dst_y * k_param_->ow);
                    float *transform_src = src_start;
                    T *transform_dst     = nullptr;
                    int h_stride0        = 4 * dst_unit_;
                    int h_stride1        = 0;

                    if (ex == dst_unit_) {
                        transform_dst = dst_start;
                        h_stride1     = 4 * k_param_->ow;
                    } else {
                        transform_dst = mid_buffer;
                        h_stride1     = 4 * dst_unit_;
                    }

                    DstTransformFunc_(transform_src, transform_dst, x_c * 4, h_stride1, ey);

                    if (ex != dst_unit_) {
                        for (int yy = 0; yy < ey; ++yy) {
                            auto dst_yy = dst_start + yy * 4 * k_param_->ow;
                            auto src_yy = mid_buffer + yy * 4 * dst_unit_;
                            memcpy(dst_yy, src_yy, count * data_byte_size);
                        }
                    }
                    // dst transform end
                }
            }
        }
    }

    PostExec<T>(outputs);

    return TNN_OK;
}

template <typename T>
Status arm_neon_conv_c3(const T* inputs, const T* outputs, std::vector<int> dims_input, std::vector<int> dims_output) {
    int kernel_x               = conv_param->kernels[0];
    int kernel_y               = conv_param->kernels[1];
    int dilate_y_step          = k_param_->iw * 4 * conv_param->dialations[1];
    int dilate_x_step          = 4 * conv_param->dialations[0];

    int weight_z_step = kernel_y * kernel_x * 12;

    T *src_origin = input;
    T *dst_origin =output;

    int max_num_threads = OMP_MAX_THREADS_NUM_;

    int src_xc = 1 + (k_param_->ow - 1) * conv_param->strides[0] + conv_param->dialations[0] * (kernel_x - 1);
    int workspace_per_thread = src_xc * kernel_y * k_param_->ic_r4 * data_byte_size;
    T *work_space = reinterpret_cast<T *>(context_->GetSharedWorkSpace(max_num_threads * workspace_per_thread));

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto input_ptr  = src_origin + batch_idx * k_param_->iw * k_param_->ih * k_param_->ic_r4;
        auto output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r4;
        int src_start_x = 0 - conv_param->pads[0];
        int src_end_x   = src_start_x + src_xc >= k_param_->iw ? k_param_->iw : src_start_x + src_xc;

        int dst_offset = 0;
        if (src_start_x < 0) {
            dst_offset  = -src_start_x;
            src_start_x = 0;
        }
        int copy_count = src_end_x - src_start_x;
        auto src_x     = input_ptr + 4 * src_start_x;

        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < k_param_->oh; dy++) {
            int thread_id = OMP_TID_;

            auto work_space_t = work_space + thread_id * workspace_per_thread / sizeof(T);
            memset(work_space_t, 0, workspace_per_thread);
            int src_start_y = dy * conv_param->strides[1] - conv_param->pads[2];
            int sfy         = MAX(0, (UP_DIV(-src_start_y, conv_param->dialations[1])));
            int efy         = MIN(kernel_y, UP_DIV(k_param_->ih - src_start_y, conv_param->dialations[1]));

            // copy make board
            for (int ky = sfy; ky < efy; ky++) {
                int sy     = src_start_y + ky * conv_param->dialations[1];
                auto src_y = src_x + 4 * sy * k_param_->iw;
                auto dst_y = work_space_t + (ky * src_xc + dst_offset) * 4;
                memcpy(dst_y, src_y, copy_count * 4 * data_byte_size);
            }
            for (int dz = 0; dz < k_param_->oc_r4 / 4; dz++) {
                auto dst_z =
                    reinterpret_cast<T *>(output_ptr) + dz * k_param_->ow * k_param_->oh * 4 + k_param_->ow * 4 * dy;
                auto weight_dz = reinterpret_cast<float *>(k_param_->fil_ptr) + dz * weight_z_step;
                // process one line at a time
                GemmSlidewC3(dst_z, reinterpret_cast<T *>(work_space_t), weight_dz, k_param_->ow,
                             conv_param->strides[0] * 4, kernel_x, kernel_y, dilate_x_step, src_xc * 4);
            }
        }
    }

    PostExec<T>(outputs);

    return TNN_OK;
}

template <typename T>
Status arm_neon_conv(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

    const int batch = dims_output[0];
    const int group = conv_param->group;
    auto ic = dims_input[1], input_slice = UP_DIV(dims_input[1], 4);
    auto oc = dims_output[1], output_slice = UP_DIV(dims_output[1], 4), output_slice_per_group = output_slice / group;

    auto gic                    = dims_input[1] / group;
    auto goc                    = dims_output[1] / group;
    auto gic_4                  = UP_DIV(gic, 4);
    auto goc_4                  = UP_DIV(goc, 4);
    auto input_bytes_per_group  = k_param_->iw * k_param_->ih * gic_4 * 4 * data_byte_size;
    auto output_bytes_per_group = k_param_->ow * k_param_->oh * goc_4 * 4 * data_byte_size;

    int dilate_y_step = k_param_->iw * 4 * conv_param->dialations[1];
    int dilate_x_step = 4 * conv_param->dialations[0];

    int src_z_step    = k_param_->iw * k_param_->ih * 4;
    int weight_z_step = conv_param->kernels[1] * conv_param->kernels[0] * gic_4 * 16;

    T *input_orign = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));
    T *dst_origin  = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));

    int max_num_threads = OMP_MAX_THREADS_NUM_;

    int x_count = UP_DIV(k_param_->ow, CONVOLUTION_TILED_NUMBER);
    int src_xc  = 1 + (CONVOLUTION_TILED_NUMBER - 1) * conv_param->strides[0] +
                conv_param->dialations[0] * (conv_param->kernels[0] - 1);
    int workspace_per_thread = src_xc * conv_param->kernels[1] * ROUND_UP(dims_input[1], 4) * data_byte_size;
    RawBuffer i_buffer;
    RawBuffer o_buffer;

    T *work_space = reinterpret_cast<T *>(context_->GetSharedWorkSpace(max_num_threads * workspace_per_thread));

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        T *input_ptr;
        T *output_ptr;

        /*
        first unpack input tensor to nchw data format
        pack data to make sure every group chanel algin4
        */
        if (gic_4 != (gic / 4) && group != 1) {
            RawBuffer i_temp_buffer_(group * input_bytes_per_group);
            RawBuffer temp_buffer(group * input_bytes_per_group);
            i_buffer  = i_temp_buffer_;
            input_ptr = i_buffer.force_to<T *>();

            UnpackC4(temp_buffer.force_to<T *>(),
                     input_orign + batch_idx * k_param_->iw * k_param_->ih * ROUND_UP(ic, 4),
                     k_param_->iw * k_param_->ih, ic);
            for (int g = 0; g < group; g++) {
                PackC4(input_ptr + g * input_bytes_per_group / 4,
                       temp_buffer.force_to<T *>() + g * k_param_->iw * k_param_->ih * gic, k_param_->iw * k_param_->ih,
                       gic);
            }
        } else {
            input_ptr = input_orign + batch_idx * k_param_->iw * k_param_->ih * ROUND_UP(ic, 4);
        }

        if (goc_4 != (goc / 4) && group != 1) {
            RawBuffer o_temp_buffer_(group * output_bytes_per_group);
            o_buffer   = o_temp_buffer_;
            output_ptr = o_buffer.force_to<T *>();
        } else {
            output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * ROUND_UP(oc, 4);
        }

        for (int g = 0; g < group; g++) {
            auto input_g_ptr  = input_ptr + g * k_param_->iw * k_param_->ih * gic_4 * 4;
            auto output_g_ptr = output_ptr + g * k_param_->ow * k_param_->oh * goc_4 * 4;
            auto w_g_offset   = g * goc_4 * weight_z_step;
            OMP_PARALLEL_FOR_
            for (int x = 0; x < x_count; x++) {
                int thread_id = OMP_TID_;

                auto work_space_t = work_space + thread_id * workspace_per_thread / sizeof(T);

                int x_idx    = (int)x * CONVOLUTION_TILED_NUMBER;
                int x_remain = k_param_->ow - x_idx;
                int x_c      = x_remain > CONVOLUTION_TILED_NUMBER ? CONVOLUTION_TILED_NUMBER : x_remain;
                int src_xc =
                    1 + (x_c - 1) * conv_param->strides[0] + conv_param->dialations[0] * (conv_param->kernels[0] - 1);
                int d_x         = x_idx;
                int src_start_x = d_x * conv_param->strides[0] - conv_param->pads[0];
                int src_end_x   = src_start_x + src_xc >= k_param_->iw ? k_param_->iw : src_start_x + src_xc;

                int dst_offset = 0;
                if (src_start_x < 0) {
                    dst_offset  = -src_start_x;
                    src_start_x = 0;
                }
                int copy_count = src_end_x - src_start_x;
                auto src_x     = input_g_ptr + 4 * src_start_x;

                for (int dy = 0; dy < k_param_->oh; dy++) {
                    /*
                    copy make board, data in workspace are dirty, should be clear first
                    */
                    memset(work_space_t, 0, workspace_per_thread);
                    int src_start_y = dy * conv_param->strides[1] - conv_param->pads[2];
                    int sfy         = MAX(0, (UP_DIV(-src_start_y, conv_param->dialations[1])));
                    int efy =
                        MIN(conv_param->kernels[1], UP_DIV(k_param_->ih - src_start_y, conv_param->dialations[1]));

                    for (int sz = 0; sz < gic_4; sz++) {
                        auto dst_z = work_space_t + sz * src_xc * conv_param->kernels[1] * 4;
                        auto src_z = src_x + sz * src_z_step;
                        for (int ky = sfy; ky < efy; ky++) {
                            int sy     = src_start_y + ky * conv_param->dialations[1];
                            auto src_y = src_z + 4 * sy * k_param_->iw;
                            auto dst_y = dst_z + (ky * src_xc + dst_offset) * 4;
                            memcpy(dst_y, src_y, copy_count * 4 * sizeof(T));
                        }
                    }

                    // output: tile x oc
                    for (int dz = 0; dz < goc_4; dz++) {
                        auto dst_z =
                            output_g_ptr + dz * k_param_->ow * k_param_->oh * 4 + x_idx * 4 + k_param_->ow * 4 * dy;
                        auto weight_dz = reinterpret_cast<float *>(k_param_->fil_ptr) + w_g_offset + dz * weight_z_step;

                        ConvCommonO4(dst_z, work_space_t, weight_dz, x_c, conv_param->strides[0] * 4, gic_4,
                                     src_xc * 4 * conv_param->kernels[1], conv_param->kernels[0],
                                     conv_param->kernels[1], dilate_x_step, src_xc * 4);
                    }
                }
            }
        }

        /*
        first unpack every group output data to get nchw data format
        pack data to make sure output tensor channel algin4 and continuously
        */
        if (goc_4 != (goc / 4) && group != 1) {
            RawBuffer temp_buffer(group * output_bytes_per_group);
            for (int g = 0; g < group; g++) {
                UnpackC4(temp_buffer.force_to<T *>() + g * k_param_->ow * k_param_->oh * goc,
                         output_ptr + g * k_param_->ow * k_param_->oh * goc_4 * 4, k_param_->ow * k_param_->oh, goc);
            }
            PackC4(dst_origin + batch_idx * k_param_->ow * k_param_->oh * ROUND_UP(oc, 4), temp_buffer.force_to<T *>(),
                   k_param_->ow * k_param_->oh, oc);
        }
    }

    PostExec<T>(outputs);

    return TNN_OK;
}
```