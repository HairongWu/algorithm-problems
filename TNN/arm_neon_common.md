# RELU

```cpp
#define ROUND_UP(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y) * (int)(y))

#define UP_DIV(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y))

#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define MIN(x, y) ((x) < (y) ? (x) : (y))

typedef union {
    float f;
    uint32_t u;
} cvt_32b;

typedef struct bfp16_struct {
public:
    uint16_t w = 0;

    bfp16_struct() : w(0) {}

    bfp16_struct(float vf) {
        cvt_32b c;
        c.f = vf;
        w   = c.u >> 16;
    }

    operator const float() const {
        cvt_32b c;
        c.u = w << 16;
        return c.f;
    }
} bfp16_t;

static float32x4_t arm_neon_load(const bfp16_t* addr) {
        float32x4_t v;
#if __aarch64__
        asm volatile(
            "ld1    {v1.4h}, [%1]\n"
            "shll   %0.4s, v1.4h, #16\n"
            : "=w"(v.value)
            : "r"(addr)
            : "memory", "v1");
#else   // __aarch64__
        asm volatile(
            "vld1.u16    d1, [%1]\n"
            "vshll.u16  %q0, d1, #16\n"
            : "=w"(v.value)
            : "r"(addr)
            : "memory", "v1");
#endif  // __aarch64__
        return v;
    }
    static void arm_neon_save(bfp16_t* addr, const float32x4_t& v) {
#if __aarch64__
        asm volatile(
            "shrn   v1.4h, %1.4s, #16\n"
            "st1    {v1.4h}, [%0]\n"
            : "+r"(addr)
            : "w"(v.value)
            : "memory", "v1");
#else   // __aarch64__
        asm volatile(
            "vshrn.u32  d1, %q1, #16\n"
            "vst1.u16    d1, [%0]\n"
            : "+r"(addr)
            : "w"(v.value)
            : "memory", "v1");
#endif  // __aarch64__
}

int UnpackNeon(float *dst, const float *src, size_t hw, size_t channel) {
    float32x4x4_t v;
    for (int c = 0; c < channel; c += 4) {
        auto src_c = src + c * hw;
        auto dst_c = dst + c * hw;
        for (int cur_hw = 0; cur_hw < hw; cur_hw += 4) {
            v = vld4q_f32(src_c + cur_hw * 4);
            vst1q_f32(dst_c + cur_hw, v.val[0]);
            vst1q_f32(dst_c + cur_hw + hw * 1, v.val[1]);
            vst1q_f32(dst_c + cur_hw + hw * 2, v.val[2]);
            vst1q_f32(dst_c + cur_hw + hw * 3, v.val[3]);
        }
    }

    return 0;
}
int UnpackNeonNHWC(float *dst, const float *src, size_t hw, size_t channel) {
    if ((hw == 1) && (channel % 4 == 0)) {
        memcpy(dst, src, hw * channel * sizeof(float));
        return 0;
    }

    auto cc = (channel>>2<<2);
    float32x4_t v;
    for (int c = 0; c < cc; c += 4) {
        auto dst_c = dst + c;
        auto src_c = src + c * hw;
        for (int cur_hw = 0; cur_hw < hw; ++cur_hw) {
            v = vld1q_f32(src_c);
            vst1q_f32(dst_c, v);
            src_c += 4;
            dst_c += channel;
        }
    }

    int remain = channel % 4;
    if (remain) {
        auto dst_c = dst + cc;
        auto src_c = src + cc * hw;
        for (int cur_hw = 0; cur_hw < hw; ++cur_hw) {
            v = vld1q_f32(src_c);
            for (int r = 0; r < remain; ++r)
                *(dst_c + r) = v[r];
            src_c += 4;
            dst_c += channel;
        }
    }

    return 0;
}
int PackNeonNHWC(float *dst, const float *src, size_t hw, size_t channel) {
    if ((hw == 1) && (channel % 4 == 0)) {
        memcpy(dst, src, hw * channel * sizeof(float));
        return 0;
    }

    auto cc = (channel>>2<<2);
    float32x4_t v;
    for (int c = 0; c < cc; c += 4) {
        auto src_c = src + c;
        auto dst_c = dst + c * hw;
        for (int cur_hw = 0; cur_hw < hw; ++cur_hw) {
            v = vld1q_f32(src_c);
            vst1q_f32(dst_c, v);
            src_c += channel;
            dst_c += 4;
        }
    }

    int remain = channel % 4;
    if (remain) {
        auto src_c = src + cc;
        auto dst_c = dst + cc * hw;
        for (int cur_hw = 0; cur_hw < hw; ++cur_hw) {
            v = vdupq_n_f32(0);
            for (int r = 0; r < remain; ++r)
                v[r] = *(src_c + r);
            vst1q_f32(dst_c, v);
            src_c += channel;
            dst_c += 4;
        }
    }

    return 0;
}
template <typename Tin, typename Tout>
int UnpackC4(Tout *dst, const Tin *src, size_t hw, size_t channel) {
#ifdef TNN_USE_NEON
    if (std::is_same<Tin, float>::value && std::is_same<Tout, float>::value) {
        if (channel % 4 == 0 && hw % 4 == 0) {
            return UnpackNeon((float *)dst, (const float *)src, hw, channel);
        }
    }
#endif
    int cur_hw;
    int c;
    int idx = 0;
    for (c = 0; c < channel; ++c) {
        int plane         = c / 4;
        const auto *src_c = plane * hw * 4 + src;
        int offset        = c % 4;
        for (cur_hw = 0; cur_hw < hw; ++cur_hw) {
            dst[idx++] = src_c[4 * cur_hw + offset];
        }
    }
    return 0;
}

template <typename Tin, typename Tout>
int PackC4(Tout *dst, const Tin *src, size_t hw, size_t channel) {
#ifdef TNN_USE_NEON
    if (std::is_same<Tin, float>::value && std::is_same<Tout, float>::value) {
        if (channel % 4 == 0 && hw % 4 == 0) {
            return PackNeon((float *)dst, (const float *)src, hw, channel);
        } else if (channel == 3 && hw % 4 == 0) {
            return PackNeonC3((float *)dst, (const float *)src, hw, channel);
        }
    }
#endif
    int c, cur_hw;
    int idx = 0;
    memset(dst, 0, hw * UP_DIV(channel, 4) * 4 * sizeof(Tout));
    for (c = 0; c < channel; ++c) {
        int plane      = c / 4;
        auto *dstPlane = plane * hw * 4 + dst;
        int offset     = c % 4;
        for (cur_hw = 0; cur_hw < hw; ++cur_hw) {
            dstPlane[4 * cur_hw + offset] = src[idx++];
        }
    }

    return 0;
}
template <typename Tin, typename Tout>
int UnpackC4ToNHWC(Tout *dst, const Tin *src, size_t hw, size_t channel) {
#ifdef TNN_USE_NEON
    if (std::is_same<Tin, float>::value && std::is_same<Tout, float>::value) {
        return UnpackNeonNHWC((float *)dst, (const float *)src, hw, channel);
    }
#endif
    int cur_hw;
    int c;
    int idx = 0;
    for (cur_hw = 0; cur_hw < hw; ++cur_hw) {
        for (c = 0; c < channel; ++c) {
            int plane         = c / 4;
            const auto *src_c = plane * hw * 4 + src;
            int offset        = c % 4;
            dst[idx++] = src_c[4 * cur_hw + offset];
        }
    }
    return 0;
}
template <typename Tin, typename Tout>
int PackC4FromNHWC(Tout *dst, const Tin *src, size_t hw, size_t channel) {
#ifdef TNN_USE_NEON
    if (std::is_same<Tin, float>::value && std::is_same<Tout, float>::value) {
        return PackNeonNHWC((float *)dst, (const float *)src, hw, channel);
    }
#endif
    int c, cur_hw;
    int idx = 0;
    memset(dst, 0, hw * UP_DIV(channel, 4) * 4 * sizeof(Tout));
    for (cur_hw = 0; cur_hw < hw; ++cur_hw) {
        for (c = 0; c < channel; ++c) {
            int plane      = c / 4;
            auto *dstPlane = plane * hw * 4 + dst;
            int offset     = c % 4;
            dstPlane[4 * cur_hw + offset] = src[idx++];
        }
    }

    return 0;
}

/*
general conv micro kernel
*/
template <typename T>
void ConvCommonO4(T* dst, const T* src, const float* weight, long width, long src_w_setup, long src_depth_quad,
                  long src_depth_step, long fw, long fh, long dilate_x_step, long dilate_y_step) {
    long dx, sz, fx, fy;
    for (dx = 0; dx < width; ++dx) {
        T* dst_x             = dst + dx * 4;
        float dst_x_float[4] = {0};
        auto src_dx          = src + src_w_setup * dx;
        for (sz = 0; sz < src_depth_quad; ++sz) {
            auto src_z    = src_dx + sz * src_depth_step;
            auto weight_z = weight + sz * fh * fw * 16;
            for (fy = 0; fy < fh; ++fy) {
                auto src_y    = src_z + fy * dilate_y_step;
                auto weight_y = weight_z + fy * fw * 16;
                for (fx = 0; fx < fw; ++fx) {
                    auto weight_x = weight_y + 16 * fx;
                    auto src_x    = src_y + fx * dilate_x_step;
                    for (long i = 0; i < 4; ++i) {
                        for (long j = 0; j < 4; ++j) {
                            dst_x_float[j] += float(src_x[i]) * float(weight_x[4 * i + j]);
                        }
                    }
                }
            }
        }
        dst_x[0] = dst_x_float[0];
        dst_x[1] = dst_x_float[1];
        dst_x[2] = dst_x_float[2];
        dst_x[3] = dst_x_float[3];
    }
}
    template <typename T>
    void PostExec(const T* outputs) {
        const int batch = outputs[0]->GetBlobDesc().dims[0];
        auto dst_origin = reinterpret_cast<T *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
        if (post_func_) {
            OMP_PARALLEL_FOR_
            for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
                auto output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r4;
                for (int dz = 0; dz < k_param_->oc_r4; dz += 4) {
                    auto dst_z    = output_ptr + dz * k_param_->ow * k_param_->oh;
                    float *bias_z = reinterpret_cast<float *>(k_param_->bias) + dz;
                    post_func_(dst_z, bias_z, k_param_->ow * k_param_->oh, 1);
                }
            }
        }
    }
```