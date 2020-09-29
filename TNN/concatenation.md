# CNN

```cpp
/*
directly copy in c4 mode, nc4hw4 format
*/
template <typename T>
int concat_channel_c4(T *output, const std::vector<T *> &inputs, std::vector<int> dims_output) {
    bool concat_c4     = true;
    auto output_stride = dims_output[2] * dims_output[3] * ROUND_UP(dims_output[1], 4);

    auto *output_origin = output;

    for (int n = 0; n < dims_output[0]; n++) {
        auto *output_ptr = output_origin + n * output_stride;
        for (int b = 0; b < inputs.size(); b++) {
            auto input        = inputs[b];
            auto input_stride = dims_input[2] * dims_input[3] * ROUND_UP(dims_input[1], 4);
            auto input_ptr    = input + n * input_stride;
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
int concat_channel(T *output, const std::vector<T *> &inputs, T *unpack_buf, std::vector<int> dims_input, std::vector<int> dims_output) {
    auto output_stride = dims_output[2] * dims_output[3] * ROUND_UP(dims_output[1], 4);
    auto output_width  = dims_output[3];
    auto output_height = dims_output[2];

    auto *output_origin = output;

    for (int n = 0; n < dims_output[0]; n++) {
        auto *output_ptr = output_origin + n * output_stride;
        auto *unpack_ptr = unpack_buf;
        int area         = output_height * output_width;
        for (int b = 0; b < inputs.size(); b++) {
            auto input      = inputs[b];
            auto c_r4       = ROUND_UP(dims_input[1], 4);
            auto input_ptr  = input + n * c_r4 * area;
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
static int concat_channel_i8(int8_t *output, const std::vector<int8_t *> &inputs, std::vector<int> dims_output) {
    int full_hw      = dims_output[2] * dims_output[3];
    auto oc_c4       = ROUND_UP(dims_output[1], 4);

    int8_t *output_origin = output;
    for (int n = 0; n < dims_output[0]; n++) {
        int c_offset = 0;
        for (int b = 0; b < inputs.size(); b++) {
            auto input_channel = inputs[b]->GetBlobDesc().dims[1];
            auto ic_c4         = ROUND_UP(input_channel, 4);
            auto input_ptr = inputs[b] + n * ic_c4 * full_hw;
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
static int concat_common_i8(int8_t *output, const std::vector<int8_t *> &inputs, int axis, std::vector<int> dims_output) {
    DimsVector round_output_dims = {output_dims[0], output_dims[2], output_dims[3], ROUND_UP(output_dims[1], 4)};
    auto slice_count             = DimsVectorUtils::Count(round_output_dims, 0, axis - 1);
    auto output_stride           = DimsVectorUtils::Count(round_output_dims, axis - 1);
    auto *output_origin          = output;

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
static int concat_common(T *output, const std::vector<T *> &inputs, int axis, std::vector<int> dims_output) {
    DimsVector round_output_dims = {output_dims[0], UP_DIV(output_dims[1], 4), output_dims[2], output_dims[3], 4};
    auto slice_count             = DimsVectorUtils::Count(round_output_dims, 0, axis);
    auto output_stride           = DimsVectorUtils::Count(round_output_dims, axis);
    auto *output_origin          = output;

    for (int n = 0; n < slice_count; n++) {
        auto output_ptr = output_origin + n * output_stride;
        for (int b = 0; b < inputs.size(); b++) {
            auto input                  = inputs[b];
            auto input_dims             = input->GetBlobDesc().dims;
            DimsVector round_input_dims = {input_dims[0], UP_DIV(input_dims[1], 4), input_dims[2], input_dims[3], 4};
            auto input_stride           = DimsVectorUtils::Count(round_input_dims, axis);
            auto input_ptr = input + n * input_stride;
            memcpy(output_ptr, input_ptr, input_stride * sizeof(T));
            output_ptr += input_stride;
        }
    }

    return 0;
}
template <typename T>
Status arm_neon_concat(const T* inputs, const T* outputs, int axis, std::vector<int> dims_output, bool concat_c4) {
    switch (axis) {
        case 1:
            if (concat_c4) {
                concat_channel_c4<T>(outputs[0], inputs);
            } else {
                auto output_stride = dims_output[2] * dims_output[3] * ROUND_UP(dims_output[1], 4);
                T *unpack_buf =
                        static_cast<float *>(context_->GetSharedWorkSpace(output_stride * sizeof(T)));
                concat_channel<T>(outputs[0], inputs, unpack_buf);
            }
            break;
        case 2:
        case 3:
            concat_common<T>(outputs[0], inputs, axis);
            break;
        default:
            LOGE("Error: Concat only support on axis 1");
            break;
    }

    return TNN_OK;
}
Status arm_neon_concat_int8(const int8_t* inputs, const int8_t *outputs, int axis) {

    switch (axis) {
        case 1:
            concat_channel_i8(outputs[0], inputs);
            break;
        case 2:
        case 3:
            concat_common_i8(outputs[0], inputs, axis);
            break;
        default:
            LOGE("Error: Concat only support on axis 1");
            break;
    }

    return TNN_OK;
}
```