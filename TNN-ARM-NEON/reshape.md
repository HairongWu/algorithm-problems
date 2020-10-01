# Reshape
The Reshape layer can be used to change the dimensions of its input, without changing its data.
```cpp

//input format DATA_FORMAT_NC4HW4
// reshape_type:
// onnx caffe reshape(nchw): 0
// Tensorflow TFLite reshape(nhwc): 1
template <typename T>
Status arm_neon_reshape(const float * inputs, const float * outputs, std::vector<int> dims_input, std::vector<int> dims_output, int reshape_type, float* workspace_) {
    auto ic    = dims_input[1];
    auto ic_r4 = ROUND_UP(dims_input[1], 4);
    auto ih    = dims_input[2];
    auto iw    = dims_input[3];
    auto oc    = dims_output[1];
    auto oc_r4 = ROUND_UP(dims_output[1], 4);
    auto oh    = dims_output[2];
    auto ow    = dims_output[3];

    auto input_plane     = ic * ih * iw;
    auto input_plane_r4  = ic_r4 * ih * iw;
    auto output_plane    = oc * oh * ow;
    auto output_plane_r4 = oc_r4 * oh * ow;

    for (int b = 0; b < dims_input[0]; b++) {
        auto input_data = inputs + b * input_plane_r4;
        auto workspace_data = workspace_ + b * input_plane;
        if (reshape_type == 0)
            UnpackC4(workspace_data, input_data, ih * iw, ic);
        else if (reshape_type == 1)
            UnpackC4ToNHWC(workspace_data, input_data, ih * iw, ic);
        else
            return Status(TNNERR_LAYER_ERR, "Unsupport reshape type");
    }
    for (int b = 0; b < dims_output[0]; b++) {
        auto workspace_data = workspace_ + b * output_plane;
        auto output_data = outputs + b * output_plane_r4;
        if (reshape_type == 0)
            PackC4(output_data, workspace_data, oh * ow, oc);
        else if (reshape_type == 1)
            PackC4FromNHWC(output_data, workspace_data, oh * ow, oc);
        else
            return Status(TNNERR_LAYER_ERR, "Unsupport reshape type");
    }

    return TNN_OK;
};

```