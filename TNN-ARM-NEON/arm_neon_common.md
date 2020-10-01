# ARM NEON COMMON

## Maths
```cpp
#define ROUND_UP(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y) * (int)(y))

#define UP_DIV(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y))

#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define MIN(x, y) ((x) < (y) ? (x) : (y))
```
## Brain floating-point format (bfloat16)
Brain floating-point format (bfloat16 or BF16) is a number encoding format occupying 16 bits representing a floating-point number. It is equivalent to a standard single-precision floating-point value with a truncated mantissa field. Bfloat16 is designed to be used in hardware accelerating machine learning algorithms. Bfloat was first proposed and implemented by Google with Intel supporting it in their FPGAs, Nervana neural processors, and CPUs.

Bfloat16 follows the same format as a standard IEEE 754 single-precision floating-point but truncates the mantissa field from 23 bits to just 7 bits. Preserving the exponent bits keeps the format to the same range as the 32-bit single precision FP (~1e-38 to ~3e38). This allows for relatively simpler conversion between the two data types. In other words, while some resolution is lost, numbers can still be represented. Microsoft developed a similar format for an 8-bit floating point based on the float16 range.

![float32: (range ~1e-38 to ~3e38)](./pictures/1600px-float32_encoding_format.svg.png)
![float16: (range ~5.96e-8 to 65,504)](./pictures/1600px-float16_encoding_format.svg.png)
![bfloat16: (range ~1e-38 to ~3e38)](./pictures/1600px-bfloat16_encoding_format.svg.png)

```cpp
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
```
## Neon Intrinsics
Neon intrinsics are function calls that the compiler replaces with an appropriate Neon instruction or sequence of Neon instructions. Intrinsics provide almost as much control as writing assembly language, but leave the allocation of registers to the compiler, so that developers can focus on the algorithms. It can also perform instruction scheduling to remove pipeline stalls for the specified target processor.
```cpp
struct Float4 {
    float32x4_t value;
    Float4() {}
    Float4(const float v) {
        value = vdupq_n_f32(v);
    }
    Float4(const float32x4_t& v) {
        value = v;
    }
    Float4(const float32x4_t&& v) {
        value = std::move(v);
    }
    Float4(const Float4& lr) {
        value = lr.value;
    }
    Float4(const Float4&& lr) {
        value = std::move(lr.value);
    }

    void set_lane(float v, int i) {
        value[i] = v;
    }

    const float operator[](const int i) const {
        return value[i];
    }

    static Float4 load(const float* addr) {
        Float4 v;
        v.value = vld1q_f32(addr);
        return v;
    }
    static void save(float* addr, const Float4& v) {
        vst1q_f32(addr, v.value);
    }
    static Float4 load(const bfp16_t* addr) {
        Float4 v;
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
    static void save(bfp16_t* addr, const Float4& v) {
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
    static void get_low(Float4& v1, Float2& v2) {
        v2.value = vget_low_f32(v1.value);
    }
    static void get_high(Float4& v1, Float2& v2) {
        v2.value = vget_high_f32(v1.value);
    }
    static Float4 combine(Float2& v1, Float2& v2) {
        return vcombine_f32(v1.value, v2.value);
    }
    static Float4 extract(const Float4& v1, const Float4& v2, const int n) {
        Float4 dst;
        if (n == 0) {
            dst.value = v1.value;
        } else if (n == 1) {
            dst.value = vextq_f32(v1.value, v2.value, 1);
        } else if (n == 2) {
            dst.value = vextq_f32(v1.value, v2.value, 2);
        } else if (n == 3) {
            dst.value = vextq_f32(v1.value, v2.value, 3);
        } else if (n == 4) {
            dst.value = v2.value;
        }
        return dst;
    }
    static Float4 pad(const Float4& v1, const Float4& v2, const int n) {
        static const uint32_t select  = uint32_t(-1);
        static const uint32x4_t mask1 = {select,select,select,0};
        static const uint32x4_t mask2 = {select,select,0,0};
        static const uint32x4_t mask3 = {select,0,0,0};
        Float4 dst;
        if (n == 0) {
            dst.value = v1.value;
        } else if (n == 1) {
            dst.value = vbslq_f32(mask1, v1.value, v2.value);
        } else if (n == 2) {
            dst.value = vbslq_f32(mask2, v1.value, v2.value);
        } else if (n == 3) {
            dst.value =  vbslq_f32(mask3, v1.value, v2.value);
        } else if (n == 4) {
            dst.value = v2.value;
        }
        return dst;
    }
    static void mla(Float4& v1, const Float4& v2, const Float4& v3) {
        v1.value = vmlaq_f32(v1.value, v2.value, v3.value);
    }
    static void mla_lane0(Float4& v1, const Float4& v2, const Float2& v3) {
        v1.value = vmlaq_lane_f32(v1.value, v2.value, v3.value, 0);
    }
    static void mla_lane1(Float4& v1, const Float4& v2, const Float2& v3) {
        v1.value = vmlaq_lane_f32(v1.value, v2.value, v3.value, 1);
    }
    static Float4 bsl_cle(const Float4& c1, const Float4& c2, const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = vbslq_f32(vcleq_f32(c1.value, c2.value), v1.value, v2.value);
        return dst;
    }
    static Float4 bsl_clt(const Float4& c1, const Float4& c2, const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = vbslq_f32(vcltq_f32(c1.value, c2.value), v1.value, v2.value);
        return dst;
    }
    static Float4 bsl_cge(const Float4& c1, const Float4& c2, const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = vbslq_f32(vcgeq_f32(c1.value, c2.value), v1.value, v2.value);
        return dst;
    }
    static Float4 bsl_cgt(const Float4& c1, const Float4& c2, const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = vbslq_f32(vcgtq_f32(c1.value, c2.value), v1.value, v2.value);
        return dst;
    }
    static Float4 neg(const Float4& v) {
        Float4 dst;
        dst.value = vnegq_f32(v.value);
        return dst;
    }
    static Float4 floor(const Float4& v) {
        Float4 dst;
#if __aarch64__
        dst.value = vcvtq_f32_s32(vcvtmq_s32_f32(v.value));
#else
        int32x4_t s32   = vcvtq_s32_f32(v.value);
        uint32x4_t mask = vcgtq_f32(vcvtq_f32_s32(s32), v.value);
        dst.value       = vcvtq_f32_s32(vaddq_s32(s32, vreinterpretq_s32_u32(mask)));
#endif
        return dst;
    }
    static Float4 ceil(const Float4& v) {
        Float4 dst;
#if __aarch64__
        dst.value = vcvtq_f32_s32(vcvtpq_s32_f32(v.value));
#else
        int32x4_t s32   = vcvtq_s32_f32(v.value);
        uint32x4_t mask = vcgtq_f32(v.value, vcvtq_f32_s32(s32));
        dst.value       = vcvtq_f32_s32(vsubq_s32(s32, vreinterpretq_s32_u32(mask)));
#endif
        return dst;
    }
    static Float4 max(const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = vmaxq_f32(v1.value, v2.value);
        return dst;
    }
    static Float4 min(const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = vminq_f32(v1.value, v2.value);
        return dst;
    }
    static Float4 div(const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = div_ps(v1.value, v2.value);
        return dst;
    }
    static Float4 exp(const Float4& v) {
        Float4 dst;
        dst.value = exp_ps(v.value);
        return dst;
    }
    static Float4 pow(const Float4& v, const Float4& e) {
        Float4 dst;
        dst.value = pow_ps(v.value, e.value);
        return dst;
    }
    static Float4 sqrt(const Float4& v) {
        Float4 dst;
        static float32x4_t zero = vdupq_n_f32(0.0f);
        dst.value = vbslq_f32(vceqq_f32(v.value, zero), zero, sqrt_ps(v.value));
        return dst;
    }
    static Float4 tanh(const Float4& v) {
        Float4 dst;
        dst.value = tanh_ps(v.value);
        return dst;
    }
    static Float4 tan(const Float4& v) {
        Float4 dst;

        float32x4_t ysin, ycos;
        sincos_ps(v.value, &ysin, &ycos);
        dst.value = div_ps(ysin, ycos);
        return dst;
    }
    static Float4 sin(const Float4& v) {
        Float4 dst;
        dst.value = sin_ps(v.value);
        return dst;
    }
    static Float4 cos(const Float4& v) {
        Float4 dst;
        dst.value = cos_ps(v.value);
        return dst;
    }
    static Float4 sigmoid(const Float4& v) {
        Float4 dst;
        dst.value = sigmoid_ps(v.value);
        return dst;
    }
    static Float4 log(const Float4& v) {
        Float4 dst;
        dst.value = log_ps(v.value);
        return dst;
    }
    static Float4 abs(const Float4& v) {
        Float4 dst;
        dst.value = vabsq_f32(v.value);
        return dst;
    }
    Float4 operator+(const Float4& lr) {
        Float4 dst;
        dst.value = value + lr.value;
        return dst;
    }
    Float4 operator-(const Float4& lr) {
        Float4 dst;
        dst.value = value - lr.value;
        return dst;
    }
    Float4 operator*(float lr) {
        Float4 dst;
        dst.value = vmulq_n_f32(value, lr);
        return dst;
    }
    Float4 operator*(const Float4& lr) {
        Float4 dst;
        dst.value = value * lr.value;
        return dst;
    }
    Float4& operator=(const Float4& lr) {
        value = lr.value;
        return *this;
    }
    Float4& operator=(const Float4&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Float4 operator-() {
        Float4 dst;
        dst.value = -value;
        return dst;
    }
};
```
## Pack and unpack
```cpp
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
```
