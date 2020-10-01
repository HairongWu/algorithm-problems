# Conv2D layer
This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.

```cpp

Status ArmConvInt8LayerCommon::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    auto input  = inputs[0];
    auto output = outputs[0];

    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;
    const int batch  = dims_output[0];
    auto ic          = dims_input[1];
    auto ic_calc     = ic < 4 ? ic : k_param_->ic_r4;

    int8_t *input_data  = reinterpret_cast<int8_t *>(GetBlobHandlePtr(input->GetHandle()));
    int8_t *output_data = reinterpret_cast<int8_t *>(GetBlobHandlePtr(output->GetHandle()));

    const int crs_div8   = UP_DIV(ic_calc * conv_param->kernels[1] * conv_param->kernels[0], 8);
    const int tile_count = UP_DIV(k_param_->oh * k_param_->ow, NEON_INT8CONV_TILE_HW);
    for (int n = 0; n < batch; ++n) {
        const auto input_batch = input_data + n * k_param_->iw * k_param_->ih * k_param_->ic_r4;
        auto output_batch      = output_data + n * k_param_->ow * k_param_->oh * k_param_->oc_r4;

        OMP_PARALLEL_FOR_GUIDED_
        for (int t_idx = 0; t_idx < tile_count; t_idx++) {
            int thread_id          = OMP_TID_;
            int8_t *input_kernel   = nullptr;
            const int hw_start     = t_idx * NEON_INT8CONV_TILE_HW;
            const int real_hw_tile = MIN(k_param_->oh * k_param_->ow - hw_start, NEON_INT8CONV_TILE_HW);
            auto gemm_work_space   = buffer_gemm_work_space_.force_to<int8_t *>();
            // im2col
            if (im_col_func_) {
                input_kernel = buffer_im2col_.force_to<int8_t *>() + crs_div8 * NEON_INT8CONV_TILE_HW * 8 * thread_id;
                im_col_func_(input_kernel, input_batch, conv_param, hw_start, real_hw_tile, crs_div8, k_param_.get());
            } else {
                input_kernel = input_batch + hw_start * ic_calc;
            }
            auto output_kernel = output_batch + hw_start * k_param_->oc_r4;
            // gemm int8
            if (real_hw_tile == NEON_INT8CONV_TILE_HW) {
                GemmInt8(output_kernel, input_kernel, gemm_work_space, reinterpret_cast<int8_t *>(k_param_->fil_ptr),
                         reinterpret_cast<int32_t *>(k_param_->bias), k_param_->scale, crs_div8, crs_div8 * 8,
                         k_param_->oc_r4);
            } else {
                int8_t *outptr_tmp =
                    buffer_tmpout_.force_to<int8_t *>() + k_param_->oc_r4 * NEON_INT8CONV_TILE_HW * thread_id;
                GemmInt8(outptr_tmp, input_kernel, gemm_work_space, reinterpret_cast<int8_t *>(k_param_->fil_ptr),
                         reinterpret_cast<int32_t *>(k_param_->bias), k_param_->scale, crs_div8, crs_div8 * 8,
                         k_param_->oc_r4);
                memcpy(output_kernel, outptr_tmp, real_hw_tile * k_param_->oc_r4);
            }
        }
        // only support relu activation
        if (conv_param->activation_type == ActivationType_ReLU) {
            ReluInt8(output_batch, output_batch, k_param_->ow * k_param_->oh * k_param_->oc_r4);
        }
    }
    return TNN_OK;
}
```