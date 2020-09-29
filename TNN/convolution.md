# CNN

```cpp

template <typename T>
Status arm_neon_conv_1x1(const T* inputs, const T* outputs, std::vector<int> dims_input, std::vector<int> dims_output, T* work_space) {
    const int batch = dims_output[0];
    auto ic4        = UP_DIV(dims_input[1], 4);
    auto oc4        = UP_DIV(dims_output[1], 4);

    int src_z_step = k_param_->iw * k_param_->ih * 4;
    int dst_z_step = k_param_->ow * k_param_->oh * 4;
    int plane_num  = k_param_->ow * k_param_->oh;

    /*
    get a_block & b_block based on l2 cache size(512K most of the time)
    */
    int max_num_threads = OMP_MAX_THREADS_NUM_;
    int threadbuf_num   = plane_num > oc4 * 4 ? max_num_threads : 1;
    int a_block, b_block;
    set_block_size(a_block, b_block, 512 * 1024 / data_byte_size, plane_num, oc4 * 4, ic4 * 4, data_byte_size);
    /*
    pack inputs when pads or strides are not equal to one
    */
    if ((k_param_->ih != k_param_->oh) || (k_param_->iw != k_param_->ow)) {
        work_space_size += ic4 * 4 * dims_output[2] * dims_output[3] * data_byte_size;
        auto tmp_dst = reinterpret_cast<T *>(context_->GetSharedWorkSpace(work_space_size + NEON_KERNEL_EXTRA_LOAD));
        work_space   = tmp_dst + ic4 * 4 * dims_output[2] * dims_output[3];

        PackLine(tmp_dst, src_origin, k_param_->ih, k_param_->iw, k_param_->oh, k_param_->ow, k_param_->ic_r4,
                 conv_param->pads[2], conv_param->pads[0], conv_param->strides[1], conv_param->strides[0]);
        src_origin = tmp_dst;
    }

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto input_ptr  = src_origin + batch_idx * k_param_->iw * k_param_->ih * ROUND_UP(dims_input[1], 4);
        auto output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * ROUND_UP(dims_output[1], 4);
        auto bias_ptr   = reinterpret_cast<float *>(k_param_->bias);

        /*
        call different sgemm func based on input and weight size
        */
        if (plane_num > oc4 * 4) {
            sgemm_repack_lhs(output_ptr, input_ptr, buffer_weight_.force_to<float *>(), ic4, oc4, plane_num, dst_z_step,
                             a_block, b_block, work_space, bias_ptr, conv_param->activation_type);
        } else {
            sgemm_repack_rhs(output_ptr, input_ptr, buffer_weight_.force_to<float *>(), ic4, oc4, plane_num, dst_z_step,
                             a_block, b_block, work_space, bias_ptr, conv_param->activation_type);
        }
    }

    return TNN_OK;
}
```