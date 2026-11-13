#pragma OPENCL FP_CONTRACT OFF
typedef struct {
    uint active_t, low_t;
    float raw_t, noise;
} Params;
typedef struct {
    uint active_t, low_t;
    float raw_t, before, after, offset;
    ulong active, quiet_sum;
    uint applied, seq;
} Meta;
__kernel void begin_frame(__global const Params* p, __global Meta* trace, uint k) {
    Meta m;
    m.active_t = p->active_t;
    m.low_t = p->low_t;
    m.raw_t = p->raw_t;
    m.before = p->noise;
    m.after = p->noise;
    m.offset = 0;
    m.active = 0;
    m.quiet_sum = 0;
    m.applied = 0;
    m.seq = k + 1;
    trace[k] = m;
}
__kernel void learn(__global const uchar* g, __global uchar* ref, __global uchar* age,
                    __global short* residual, __global uint2* partial, __global const Meta* trace,
                    uint k, uint n) {
    uint lane = get_local_id(0), gid = get_global_id(0), count = 0, sum = 0;
    int collect = (k + 1) % 5 == 0;
    Meta m = trace[k];
    for (uint i = gid; i < n; i += get_global_size(0)) {
        if (k == 0) {
            ref[i] = g[i];
            age[i] = 0;
            continue;
        }
        int signed_d = (int)g[i] - (int)ref[i];
        uint d = (uint)abs(signed_d);
        residual[i] = (short)signed_d;
        int active = d > m.active_t;
        uint next = active ? min(255u, (uint)age[i] + 1) : 0;
        uchar r = ref[i];
        if (d <= m.low_t) {
            r = g[i];
        } else if (!active) {
            r = (uchar)(((uint)r + (uint)g[i]) / 2);
        }
        if (next >= 50) {
            r = g[i];
            next = 0;
        }
        ref[i] = r;
        age[i] = (uchar)next;
        if (collect) {
            count += (uint)active;
            if (!active) {
                sum += d;
            }
        }
    }
    __local uint counts[256], sums[256];
    counts[lane] = count;
    sums[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = 128; step; step >>= 1) {
        if (lane < step) {
            counts[lane] += counts[lane + step];
            sums[lane] += sums[lane + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && collect) {
        partial[get_group_id(0)] = (uint2)(counts[0], sums[0]);
    }
}
__kernel void reduce_noise(__global const uint2* partial, uint groups, __global Params* p,
                           __global Meta* trace, uint k, uint n, uint main_area) {
    ulong active = 0, sum = 0;
    for (uint i = 0; i < groups; ++i) {
        uint2 v = partial[i];
        active += v.x;
        sum += v.y;
    }
    trace[k].active = active;
    trace[k].quiet_sum = sum;
    uint applied = active < (ulong)main_area && active < (ulong)n;
    trace[k].applied = applied;
    if (applied) {
        float mean = convert_float_rte(sum) / convert_float_rte((ulong)n - active);
        float old_part = .8f * p->noise;
        float target = 4.f + 2.f * mean;
        float new_part = .2f * target;
        float value = clamp(old_part + new_part, 12.f, 48.f);
        p->noise = value;
        p->active_t = convert_uint_rtz(value);
        p->low_t = max(1u, convert_uint_rtz(value * .2f));
        p->raw_t = value;
        trace[k].after = value;
    }
}
__kernel void histogram(__global const short* residual, __global uint* hist, uint w, uint h,
                        uint pitch) {
    uint nx = (w + 15) / 16, ny = (h + 15) / 16;
    for (uint i = get_global_id(0); i < nx * ny; i += get_global_size(0)) {
        atomic_inc(hist + ((int)residual[(i / nx) * 16 * pitch + (i % nx) * 16] + 255));
    }
}
__kernel void median_offset(__global const uint* hist, __global Meta* trace, uint k, uint samples) {
    uint a = (samples - 1) / 2, b = samples / 2, cumulative = 0;
    int first = -256, second = 0;
    for (int i = 0; i < 511; ++i) {
        cumulative += hist[i];
        if (first == -256 && cumulative > a) {
            first = i - 255;
        }
        if (cumulative > b) {
            second = i - 255;
            break;
        }
    }
    trace[k].offset = ((float)first + (float)second) * .5f;
}
__kernel void raw_mask(__global const short* residual, __global uchar* raw,
                       __global const Meta* trace, uint k, uint w, uint h, uint pitch, uint n,
                       int wx, int wy, int ww, int wh) {
    Meta m = trace[k];
    for (uint i = get_global_id(0); i < n; i += get_global_size(0)) {
        uint x = i % pitch, y = i / pitch;
        int excluded = x >= w || y >= h ||
                       ((int)x >= wx && (int)x < wx + ww && (int)y >= wy && (int)y < wy + wh);
        raw[i] = excluded ? 0 : (fabs((float)residual[i] - m.offset) > m.raw_t ? 255 : 0);
    }
}
