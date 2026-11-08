// Host-side FP8 helpers: bit-exact decode, LUT construction, RNE+saturating encode.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <limits>

// Step 1: the format is a runtime knob, never assumed.
//   E4M3FN: 1-4-3, bias 7,  no inf, NaN = S.1111.111, max finite 448
//   E5M2  : 1-5-2, bias 15, IEEE-like inf/NaN,        max finite 57344
struct Fp8Format {
    const char* name;
    int ebits, mbits, bias;
    bool has_inf;
    float max_finite;
};

static const Fp8Format FMT_E4M3FN = {"E4M3FN", 4, 3, 7, false, 448.0f};
static const Fp8Format FMT_E5M2   = {"E5M2",   5, 2, 15, true, 57344.0f};

inline bool fp8_is_nan(uint8_t c, const Fp8Format& f) {
    int e = (c >> f.mbits) & ((1 << f.ebits) - 1);
    int m = c & ((1 << f.mbits) - 1);
    if (f.has_inf) return e == (1 << f.ebits) - 1 && m != 0;
    return e == (1 << f.ebits) - 1 && m == (1 << f.mbits) - 1;
}

inline bool fp8_is_inf(uint8_t c, const Fp8Format& f) {
    int e = (c >> f.mbits) & ((1 << f.ebits) - 1);
    int m = c & ((1 << f.mbits) - 1);
    return f.has_inf && e == (1 << f.ebits) - 1 && m == 0;
}

// Decode from bit fields (never an integer cast). Preserves +0/-0 and +-inf.
inline float fp8_decode(uint8_t c, const Fp8Format& f) {
    int s = c >> 7;
    int e = (c >> f.mbits) & ((1 << f.ebits) - 1);
    int m = c & ((1 << f.mbits) - 1);
    if (fp8_is_nan(c, f)) return std::numeric_limits<float>::quiet_NaN();
    if (fp8_is_inf(c, f)) { float inf = std::numeric_limits<float>::infinity(); return s ? -inf : inf; }
    float v = (e == 0) ? std::ldexp((float)m, 1 - f.bias - f.mbits)
                       : std::ldexp((float)((1 << f.mbits) + m), e - f.bias - f.mbits);
    return s ? -v : v;
}

// uint8 -> fp16 bit patterns. Every FP8 value is exactly representable in fp16.
inline void fp8_build_lut(const Fp8Format& f, uint16_t out[256]) {
    for (int c = 0; c < 256; ++c) {
        if (fp8_is_nan(c, f)) { out[c] = (uint16_t)(((c >> 7) << 15) | 0x7E00); continue; }
        out[c] = __half_as_ushort(__float2half_rn(fp8_decode((uint8_t)c, f)));
    }
}

// Round-to-nearest-even, saturating to +-max_finite (no inf produced). NaN in -> NaN code.
inline uint8_t fp8_encode(float x, const Fp8Format& f) {
    if (std::isnan(x)) return 0x7F;
    int sign = std::signbit(x) ? 0x80 : 0;
    if (x == 0.0f) return (uint8_t)sign;
    double ax = std::fabs((double)x);
    // Saturate: +-inf and every |x| at or beyond max_finite map to the top finite code (E4M3FN 126,
    // E5M2 123). The search below only scans finite codes, so it cannot do this itself (inf gave
    // +-0, and E5M2 ties near 2^67 picked the even code 122 instead of 123).
    if (!(ax < (double)f.max_finite)) {
        int top = f.has_inf ? (((1 << f.ebits) - 2) << f.mbits) | ((1 << f.mbits) - 1)
                            : (((1 << f.ebits) - 1) << f.mbits) | ((1 << f.mbits) - 2);
        return (uint8_t)(sign | top);
    }
    int best = 0;
    double best_d = 1e300;
    for (int c = 0; c < 128; ++c) {  // positive codes only
        if (fp8_is_nan((uint8_t)c, f) || fp8_is_inf((uint8_t)c, f)) continue;
        double d = std::fabs(ax - (double)fp8_decode((uint8_t)c, f));
        if (d < best_d || (d == best_d && (c & 1) == 0)) { best_d = d; best = c; }
    }
    return (uint8_t)(sign | best);
}
