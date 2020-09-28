# CNN

```cpp
/*
directly copy in c4 mode, nc4hw4 format
*/
template <typename T>
int concat_channel_c4(Blob *output, const std::vector<Blob *> &inputs) {
    bool concat_c4     = true;
    auto dims_output   = output->GetBlobDesc().dims;
    auto output_stride = dims_output[2] * dims_output[3] * ROUND_UP(dims_output[1], 4);

    auto *output_origin = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));

    for (int n = 0; n < dims_output[0]; n++) {
        auto *output_ptr = output_origin + n * output_stride;
        for (int b = 0; b < inputs.size(); b++) {
            auto input        = inputs[b];
            auto dims_input   = input->GetBlobDesc().dims;
            auto input_stride = dims_input[2] * dims_input[3] * ROUND_UP(dims_input[1], 4);
            auto input_ptr    = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle())) + n * input_stride;
            memcpy(output_ptr, input_ptr, input_stride * sizeof(T));
            output_ptr += input_stride;
        }
    }

    return 0;
}

/*
need extra buf to pack and unpack, channel not align with 4, nc4hw4 format
*/
template <typename T>
int concat_channel(Blob *output, const std::vector<Blob *> &inputs, T *unpack_buf) {
    auto dims_output   = output->GetBlobDesc().dims;
    auto output_stride = dims_output[2] * dims_output[3] * ROUND_UP(dims_output[1], 4);
    auto output_width  = dims_output[3];
    auto output_height = dims_output[2];

    auto *output_origin = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));

    for (int n = 0; n < dims_output[0]; n++) {
        auto *output_ptr = output_origin + n * output_stride;
        auto *unpack_ptr = unpack_buf;
        int area         = output_height * output_width;
        for (int b = 0; b < inputs.size(); b++) {
            auto input      = inputs[b];
            auto dims_input = input->GetBlobDesc().dims;
            auto c_r4       = ROUND_UP(dims_input[1], 4);
            auto input_ptr  = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle())) + n * c_r4 * area;
            UnpackC4(unpack_ptr, input_ptr, area, dims_input[1]);
            unpack_ptr += dims_input[1] * area;
        }
        PackC4(output_ptr, unpack_buf, area, dims_output[1]);
    }

    return 0;
}

/*
concat channel int8, nhwc format
*/
static int concat_channel_i8(Blob *output, const std::vector<Blob *> &inputs) {
    auto dims_output = output->GetBlobDesc().dims;
    int full_hw      = dims_output[2] * dims_output[3];
    auto oc_c4       = ROUND_UP(dims_output[1], 4);

    int8_t *output_origin = reinterpret_cast<int8_t *>(GetBlobHandlePtr(output->GetHandle()));
    for (int n = 0; n < dims_output[0]; n++) {
        int c_offset = 0;
        for (int b = 0; b < inputs.size(); b++) {
            auto input_channel = inputs[b]->GetBlobDesc().dims[1];
            auto ic_c4         = ROUND_UP(input_channel, 4);
            auto input_ptr = reinterpret_cast<int8_t *>(GetBlobHandlePtr(inputs[b]->GetHandle())) + n * ic_c4 * full_hw;
            auto output_ptr = output_origin + n * full_hw * oc_c4 + c_offset;
            for (int cur_hw = 0; cur_hw < full_hw; cur_hw++) {
                memcpy(output_ptr + cur_hw * oc_c4, input_ptr + cur_hw * ic_c4, input_channel);
            }
            c_offset += input_channel;
        }
    }

    return 0;
}

/*
concat common int8, nhwc format
*/
static int concat_common_i8(Blob *output, const std::vector<Blob *> &inputs, int axis) {
    auto output_dims             = output->GetBlobDesc().dims;
    DimsVector round_output_dims = {output_dims[0], output_dims[2], output_dims[3], ROUND_UP(output_dims[1], 4)};
    auto slice_count             = DimsVectorUtils::Count(round_output_dims, 0, axis - 1);
    auto output_stride           = DimsVectorUtils::Count(round_output_dims, axis - 1);
    auto *output_origin          = reinterpret_cast<int8_t *>(GetBlobHandlePtr(output->GetHandle()));

    for (int n = 0; n < slice_count; n++) {
        auto output_ptr = output_origin + n * output_stride;
        for (int b = 0; b < inputs.size(); b++) {
            auto input                  = inputs[b];
            auto input_dims             = input->GetBlobDesc().dims;
            DimsVector round_input_dims = {input_dims[0], input_dims[2], input_dims[3], ROUND_UP(input_dims[1], 4)};
            auto input_stride           = DimsVectorUtils::Count(round_input_dims, axis - 1);
            auto input_ptr = reinterpret_cast<int8_t *>(GetBlobHandlePtr(input->GetHandle())) + n * input_stride;
            memcpy(output_ptr, input_ptr, input_stride * sizeof(int8_t));
            output_ptr += input_stride;
        }
    }

    return 0;
}

/*
concat channel common(float & bf16), nc4hw4 format
*/
template <typename T>
static int concat_common(Blob *output, const std::vector<Blob *> &inputs, int axis) {
    auto output_dims             = output->GetBlobDesc().dims;
    DimsVector round_output_dims = {output_dims[0], UP_DIV(output_dims[1], 4), output_dims[2], output_dims[3], 4};
    auto slice_count             = DimsVectorUtils::Count(round_output_dims, 0, axis);
    auto output_stride           = DimsVectorUtils::Count(round_output_dims, axis);
    auto *output_origin          = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));

    for (int n = 0; n < slice_count; n++) {
        auto output_ptr = output_origin + n * output_stride;
        for (int b = 0; b < inputs.size(); b++) {
            auto input                  = inputs[b];
            auto input_dims             = input->GetBlobDesc().dims;
            DimsVector round_input_dims = {input_dims[0], UP_DIV(input_dims[1], 4), input_dims[2], input_dims[3], 4};
            auto input_stride           = DimsVectorUtils::Count(round_input_dims, axis);
            auto input_ptr = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle())) + n * input_stride;
            memcpy(output_ptr, input_ptr, input_stride * sizeof(T));
            output_ptr += input_stride;
        }
    }

    return 0;
}
Status arm_neon_concat(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs, int axis, int data_type, std::vector<int> dims_output) {
    bool concat_c4 = true;
    for (int i = 0; i < inputs.size() - 1; i++) {
        if (inputs[i]->GetBlobDesc().dims[1] % 4 != 0) {
            concat_c4 = false;
            break;
        }
    }

    switch (axis) {
        case 1:
            if (data_type == DATA_TYPE_FLOAT) {
                if (concat_c4) {
                    concat_channel_c4<float>(outputs[0], inputs);
                } else {
                    auto output_stride = dims_output[2] * dims_output[3] * ROUND_UP(dims_output[1], 4);
                    float *unpack_buf =
                        static_cast<float *>(context_->GetSharedWorkSpace(output_stride * sizeof(float)));
                    concat_channel<float>(outputs[0], inputs, unpack_buf);
                }
            } else if (data_type == DATA_TYPE_INT8) {
                concat_channel_i8(outputs[0], inputs);
            } else if (data_type == DATA_TYPE_BFP16) {
                if (concat_c4) {
                    concat_channel_c4<bfp16_t>(outputs[0], inputs);
                } else {
                    auto output_stride = dims_output[2] * dims_output[3] * ROUND_UP(dims_output[1], 4);
                    bfp16_t *unpack_buf =
                        static_cast<bfp16_t *>(context_->GetSharedWorkSpace(output_stride * sizeof(bfp16_t)));
                    concat_channel<bfp16_t>(outputs[0], inputs, unpack_buf);
                }
            } else {
                return TNNERR_LAYER_ERR;
            }
            break;
        case 2:
        case 3:
            if (data_type == DATA_TYPE_FLOAT) {
                concat_common<float>(outputs[0], inputs, axis);
            } else if (data_type == DATA_TYPE_BFP16) {
                concat_common<bfp16_t>(outputs[0], inputs, axis);
            } else if (data_type == DATA_TYPE_INT8) {
                concat_common_i8(outputs[0], inputs, axis);
            } else {
                return TNNERR_LAYER_ERR;
            }
            break;
        default:
            LOGE("Error: Concat only support on axis 1");
            break;
    }

    return TNN_OK;
}

```