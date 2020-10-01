# Conv2D layer

```cpp

template <typename T>
Status ArmConvLayerDepthwiseS1::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

    const int batch    = dims_output[0];
    int dst_depth_quad = UP_DIV(dims_output[1], 4);
    int dst_z_step     = k_param_->ow * k_param_->oh;
    int src_z_step     = k_param_->iw * k_param_->ih;
    int pad_l          = conv_param->pads[0];
    int pad_r          = conv_param->pads[1];
    int pad_t          = conv_param->pads[2];
    int pad_b          = conv_param->pads[3];
    int weight_z_step  = conv_param->kernels[0] * conv_param->kernels[1];

    auto *src_origin         = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));
    auto *dst_origin         = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));
    int max_num_threads      = OMP_MAX_THREADS_NUM_;
    int workspace_per_thread = conv_param->kernels[1] * (k_param_->iw + pad_l + pad_r) * 4 * data_byte_size;

    if (!SlideFunc_) {
        LOGE("Error: ConvDw slide func is nil\n");
        return Status(TNNERR_LAYER_ERR, "Error: ConvDw slide func is nil");
    }

    if (pad_t > conv_param->kernels[1]) {
        LOGE("ERROR: ConvDw pad_t must small than kernel_h\n");
        return Status(TNNERR_LAYER_ERR, "ERROR: ConvDw pad_t must small than kernel_h");
    }

    auto work_space = reinterpret_cast<T *>(context_->GetSharedWorkSpace(max_num_threads * workspace_per_thread));

    /*
    [ATTENTION]
    data in workspace are dirty, must be clear first
    */
    memset(work_space, 0, max_num_threads * workspace_per_thread);
    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto src_ptr = src_origin + batch_idx * k_param_->iw * k_param_->ih * k_param_->ic_r4;
        auto dst_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r4;

        OMP_PARALLEL_FOR_
        for (int dz = 0; dz < k_param_->oc_r4; dz += 4) {
            auto *dst_z                       = dst_ptr + dst_z_step * dz;
            auto *src_z                       = src_ptr + src_z_step * dz;
            const auto *weight_dz             = reinterpret_cast<float *>(k_param_->fil_ptr) + dz * weight_z_step;
            int thread_id                     = OMP_TID_;
            auto thread_work_space            = work_space + thread_id * workspace_per_thread / data_byte_size;
            T *cache_line[MAX_CACHE_LINE_NUM] = {nullptr};
            for (int i = 0; i < conv_param->kernels[1]; i++) {
                cache_line[i] = thread_work_space + i * (k_param_->iw + pad_l + pad_r) * 4;
            }

            auto src_y = src_z;
            auto dst_y = dst_z;
            // memset pat_t lines
            for (int ky = 0; ky < pad_t; ky++) {
                memset(cache_line[ky] + pad_l * 4, 0, k_param_->iw * 4 * data_byte_size);
            }
            // load mid lines
            for (int ky = pad_t; ky < conv_param->kernels[1] - 1; ky++) {
                memcpy(cache_line[ky] + pad_l * 4, src_y, k_param_->iw * 4 * data_byte_size);
                src_y += k_param_->iw * 4;
            }
            for (int dy = 0; dy < k_param_->oh - pad_b; dy++) {
                // load only one line every loop
                memcpy(cache_line[conv_param->kernels[1] - 1] + pad_l * 4, src_y, k_param_->iw * 4 * data_byte_size);
                // kernel func
                SlideFunc_(dst_y, (void **)cache_line, weight_dz, k_param_->ow);

                src_y += k_param_->iw * 4;
                dst_y += k_param_->ow * 4;
                cache_lines_slide(cache_line, conv_param->kernels[1]);
            }
            // memset pad_b lines
            for (int ky = pad_b; ky > 0; ky--) {
                memset(cache_line[conv_param->kernels[1] - 1] + pad_l * 4, 0, k_param_->iw * 4 * data_byte_size);
                // kernel func
                SlideFunc_(dst_y, (void **)cache_line, weight_dz, k_param_->ow);

                dst_y += k_param_->ow * 4;
                cache_lines_slide(cache_line, conv_param->kernels[1]);
            }
        }
    }

    PostExec<T>(outputs);

    return TNN_OK;
}

template <typename T>
Status ArmConvLayerDepthwise::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    const int batch    = dims_output[0];
    int dst_depth_quad = UP_DIV(dims_output[1], 4);
    int dst_z_step     = k_param_->ow * k_param_->oh;
    int src_z_step     = k_param_->iw * k_param_->ih;
    int dilate_y_step  = k_param_->iw * 4 * param->dialations[1];
    int dilate_x_step  = 4 * param->dialations[0];
    int weight_z_step  = param->kernels[0] * param->kernels[1];

    int l = 0, t = 0, r = k_param_->ow, b = k_param_->oh;
    for (; l * param->strides[0] - param->pads[0] < 0; l++)
        ;
    for (; t * param->strides[1] - param->pads[2] < 0; t++)
        ;
    for (; (r - 1) * param->strides[0] - param->pads[0] + param->kernels[0] * param->dialations[0] > k_param_->iw &&
            r > l; r--)
        ;
    for (; (b - 1) * param->strides[1] - param->pads[2] + param->kernels[1] * param->dialations[1] > k_param_->ih &&
            b > t; b--)
        ;

    // lamda function to process left/right/top/bottom corner
    auto RunCorner = [=](T *dst_z, const T *src_z, const float *weight_dz, int left, int top, int right, int bottom) {
        for (int dy = top; dy < bottom; ++dy) {
            auto *dst_y        = dst_z + dy * k_param_->ow * 4;
            int srcStartY      = dy * param->strides[1] - param->pads[2];
            const auto *src_dy = src_z + srcStartY * k_param_->iw * 4;
            int sfy            = MAX(0, (UP_DIV(-srcStartY, param->dialations[1])));
            int efy            = MIN(param->kernels[1], UP_DIV(k_param_->ih - srcStartY, param->dialations[1]));
            for (int dx = left; dx < right; ++dx) {
                auto *dst_x        = dst_y + 4 * dx;
                int srcStartX      = dx * param->strides[0] - param->pads[0];
                const auto *src_dx = src_dy + srcStartX * 4;
                int sfx            = MAX(0, (UP_DIV(-srcStartX, param->dialations[0])));
                int efx            = MIN(param->kernels[0], UP_DIV(k_param_->iw - srcStartX, param->dialations[0]));
                DepthwiseUnit(dst_x,
                              src_dx + (sfx * param->dialations[0] + sfy * param->dialations[1] * k_param_->iw) * 4,
                              weight_dz + 4 * (param->kernels[0] * sfy + sfx), efx - sfx, efy - sfy,
                              4 * param->kernels[0], dilate_x_step, dilate_y_step);
            }
        }
    };

    auto *src_origin = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));
    auto *dst_origin = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));
    auto dw_full     = DepthwiseConv<T>;
    /*
    convdw3x3 stride >= 2 here
    convdw3x3s1 has separate kernel in convdws1 acc
    */
    if (param->kernels[0] == 3 && param->kernels[1] == 3) {
        dw_full = DepthwiseConv3x3<T>;
    }
    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto src_ptr = src_origin + batch_idx * k_param_->iw * k_param_->ih * k_param_->ic_r4;
        auto dst_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r4;

        OMP_PARALLEL_FOR_
        for (int dz = 0; dz < k_param_->oc_r4; dz += 4) {
            auto *dst_z     = dst_ptr + dst_z_step * dz;
            auto *src_z     = src_ptr + src_z_step * dz;
            auto *weight_dz = reinterpret_cast<float *>(k_param_->fil_ptr) + dz * weight_z_step;
            auto *bias_z    = reinterpret_cast<T *>(k_param_->bias) + dz;

            RunCorner(dst_z, src_z, weight_dz, 0, 0, k_param_->ow, t);
            RunCorner(dst_z, src_z, weight_dz, 0, b, k_param_->ow, k_param_->oh);
            RunCorner(dst_z, src_z, weight_dz, 0, t, l, b);
            RunCorner(dst_z, src_z, weight_dz, r, t, k_param_->ow, b);

            if (r > l && b > t) {
                dw_full(dst_z + t * k_param_->ow * 4 + l * 4,
                        src_z + (t * param->strides[1] - param->pads[2]) * k_param_->iw * 4 +
                            (l * param->strides[0] - param->pads[0]) * 4,
                        weight_dz, r - l, param->strides[0] * 4, param->kernels[0], param->kernels[1], dilate_x_step,
                        dilate_y_step, b - t, k_param_->iw * 4 * param->strides[1], k_param_->ow * 4);
            }
        }
    }

    PostExec<T>(outputs);

    return TNN_OK;
}
```