#pragma OPENCL FP_CONTRACT OFF
const sampler_t nearest =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__kernel void gray_from_y(read_only image2d_t yplane, __global uchar* gray,
                          __global const uchar* lut, int w, int h, int crop_x, int crop_y) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    uint v = convert_uint_sat_rte(read_imagef(yplane, nearest, (int2)(x + crop_x, y + crop_y)).x *
                                  255.0f);
    gray[y * w + x] = lut[v];
}
inline int reflect101(int p, int n) {
    if (n == 1) {
        return 0;
    }
    while (p < 0 || p >= n) {
        p = p < 0 ? -p : 2 * n - p - 2;
    }
    return p;
}
__kernel void horizontal9(__global const uchar* gray, __global ushort* temp, int w, int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    const uint c[9] = {4, 13, 30, 51, 60, 51, 30, 13, 4};
    uint sum = 0;
    for (int k = 0; k < 9; ++k) {
        sum += c[k] * gray[y * w + reflect101(x + k - 4, w)];
    }
    temp[y * w + x] = (ushort)sum;
}
__kernel void vertical9_mask(__global const ushort* temp, __global uchar* out, int w, int h, int pw,
                             int ph, int wx, int wy, int ww, int wh) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= pw || y >= ph) {
        return;
    }
    uchar v = 0;
    if (x < w && y < h && !(x >= wx && x < wx + ww && y >= wy && y < wy + wh)) {
        const uint c[9] = {4, 13, 30, 51, 60, 51, 30, 13, 4};
        uint sum = 0;
        for (int k = 0; k < 9; ++k) {
            sum += c[k] * temp[reflect101(y + k - 4, h) * w + x];
        }
        v = (uchar)((sum + 32768u) >> 16);
    }
    out[y * pw + x] = v;
}
