# RELU
In the context of artificial neural networks, the rectifier is an activation function defined as the positive part of its argument:
$$f(x)=\max(0, x)$$

> Advantages
- Biological plausibility: One-sided, compared to the antisymmetry of tanh.
- Sparse activation: For example, in a randomly initialized network, only about 50% of hidden units are activated (have a non-zero output).
- Better gradient propagation: Fewer vanishing gradient problems compared to sigmoidal activation functions that saturate in both directions.
- Efficient computation: Only comparison, addition and multiplication.
- Scale-invariant: $\max(0,ax)=a\max(0,x)\text{ for }a\geq 0$
  
> Potential problems
- Non-differentiable at zero; however, it is differentiable anywhere else, and the value of the derivative at zero can be arbitrarily chosen to be 0 or 1.
- Not zero-centered.
- Unbounded.
- Dying ReLU problem: ReLU neurons can sometimes be pushed into states in which they become inactive for essentially all inputs. 
```cpp


//input & output data_format is NCHW
void arm_neon_relu_int8(int8_t* dst, const int8_t* src, std::vector<int> dims) {
    long len = dims[0] * ROUND_UP(dims[1], 4) * dims[2] * dims[3];
    long idx = 0;

    int8x8_t zero = vdup_n_s8(0);
    idx           = len - len % 8;
    for (long i = 0; i < idx; i += 8) {
        int8x8_t val = vld1_s8(src + i);
        vst1_s8(dst + i, vmax_s8(val, zero));
    }

    for (; idx < len; idx++) {
        dst[idx] = MAX(0, src[idx]);
    }
}

void arm_neon_relu_float(float* dst, const float* src, std::vector<int> dims) {
    long len = dims[0] * ROUND_UP(dims[1], 4) * dims[2] * dims[3];
    float32x4_t vzero = vdupq_n_f32(0);
    for (int i = 0; i < len; i += 4) {
        vst1q_f32(dst + i, vmaxq_f32(vld1q_f32(src + i), vzero));
    }
}

void arm_neon_relu_bfp16(bfp16_t* dst, const bfp16_t* src, std::vector<int> dims) {
    long len = dims[0] * ROUND_UP(dims[1], 4) * dims[2] * dims[3];
    float32x4_t vzero = vdupq_n_f32(0);
    for (int i = 0; i < len; i += 4) {
        arm_neon_save(dst + i, vmaxq_f32(arm_neon_load(src + i), vzero));
    }
}
```