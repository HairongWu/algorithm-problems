# CNN

```cpp

template <typename T>
Status arm_neon_conv(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs, std::vector<int> dims_input, std::vector<int> dims_output, int batch, int group, int std::vector<int> dialations) {
    auto ic = dims_input[1], input_slice = UP_DIV(dims_input[1], 4);
    auto oc = dims_output[1], output_slice = UP_DIV(dims_output[1], 4), output_slice_per_group = output_slice / group;

    auto gic                    = dims_input[1] / group;
    auto goc                    = dims_output[1] / group;
    auto gic_4                  = UP_DIV(gic, 4);
    auto goc_4                  = UP_DIV(goc, 4);
    auto input_bytes_per_group  = k_param_->iw * k_param_->ih * gic_4 * 4 * data_byte_size;
    auto output_bytes_per_group = k_param_->ow * k_param_->oh * goc_4 * 4 * data_byte_size;

    int dilate_y_step = k_param_->iw * 4 * dialations[1];
    int dilate_x_step = 4 * dialations[0];

    int src_z_step    = k_param_->iw * k_param_->ih * 4;
    int weight_z_step = conv_param->kernels[1] * conv_param->kernels[0] * gic_4 * 16;

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