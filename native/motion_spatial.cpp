#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace py = pybind11;
using Json = nlohmann::json;
static void check_opencl(cl_int e) {
    if (e != CL_SUCCESS) {
        throw std::runtime_error("OpenCL error " + std::to_string(e));
    }
}
struct MotionParameters {
    cl_uint active_t, low_t;
    cl_float raw_t, noise;
};
struct MotionFrameMetrics {
    cl_uint active_t, low_t;
    cl_float raw_t, before, after, offset;
    cl_ulong active, quiet_sum;
    cl_uint applied, seq;
};
struct ComponentRegion {
    cl_int x, y, width, height, area;
};
static_assert(sizeof(MotionParameters) == 16 && sizeof(MotionFrameMetrics) == 48 &&
              offsetof(MotionFrameMetrics, active) == 24 && sizeof(ComponentRegion) == 20);

class MotionSpatialModel {
    cl_context cl_context_handle;
    cl_command_queue compute_queue;
    std::vector<cl_program> programs;
    std::map<std::string, cl_kernel> kernels;
    std::vector<cl_mem> device_buffers;
    cl_uint width, height, padded_width, padded_height, visible_pixels, padded_pixels, groups,
        trace_capacity, update_count = 0, region_capacity;
    int low_area, strict_area;
    std::array<int, 4> watermark_box;
    cl_mem reference_frame, reference_age, motion_residual, motion_mask, partial_statistics,
        residual_histogram, parameters, frame_metrics, packed_mask, morphology_buffer_a,
        morphology_buffer_b, low_keep_mask, strict_input_mask, component_parents,
        tile_component_ids, component_statistics, region_counts, region_buffer;
    uint64_t allocated = 0, compact_read = 0, trace_read = 0;
    cl_mem allocate_device_buffer(size_t size) {
        cl_int e;
        auto p = clCreateBuffer(cl_context_handle, CL_MEM_READ_WRITE, size, nullptr, &e);
        check_opencl(e);
        cl_uchar zero = 0;
        check_opencl(clEnqueueFillBuffer(compute_queue, p, &zero, 1, 0, size, 0, nullptr, nullptr));
        device_buffers.push_back(p);
        allocated += size;
        return p;
    }
    template <class T> void set_kernel_argument(cl_kernel k, int i, T value) {
        check_opencl(clSetKernelArg(k, i, sizeof(value), &value));
    }
    void enqueue_linear_kernel(const std::string& name, size_t size, int channels = 1,
                               size_t local_size = 256) {
        size_t global[2] = {(size + local_size - 1) / local_size * local_size, size_t(channels)},
               local[2] = {local_size, 1};
        check_opencl(clEnqueueNDRangeKernel(compute_queue, kernels.at(name), 2, nullptr, global,
                                            local, 0, nullptr, nullptr));
    }
    void enqueue_tile_kernel(const std::string& name, int channels = 1) {
        size_t global[3] = {(width + 15) / 16 * 16ul, (height + 15) / 16 * 16ul, size_t(channels)},
               local[3] = {16, 16, 1};
        check_opencl(clEnqueueNDRangeKernel(compute_queue, kernels.at(name), 3, nullptr, global,
                                            local, 0, nullptr, nullptr));
    }
    void label_components(cl_mem input, int channels) {
        auto k = kernels.at("tile_labels");
        set_kernel_argument(k, 0, input);
        set_kernel_argument(k, 1, component_parents);
        set_kernel_argument(k, 2, tile_component_ids);
        set_kernel_argument(k, 3, width);
        set_kernel_argument(k, 4, height);
        enqueue_tile_kernel("tile_labels", channels);
        k = kernels.at("join_tiles");
        set_kernel_argument(k, 0, component_parents);
        set_kernel_argument(k, 1, width);
        set_kernel_argument(k, 2, height);
        enqueue_tile_kernel("join_tiles", channels);
        k = kernels.at("flatten_init");
        set_kernel_argument(k, 0, component_parents);
        set_kernel_argument(k, 1, component_statistics);
        set_kernel_argument(k, 2, width);
        set_kernel_argument(k, 3, height);
        enqueue_linear_kernel("flatten_init", visible_pixels, channels);
        k = kernels.at("tile_stats");
        set_kernel_argument(k, 0, component_parents);
        set_kernel_argument(k, 1, tile_component_ids);
        set_kernel_argument(k, 2, component_statistics);
        set_kernel_argument(k, 3, width);
        set_kernel_argument(k, 4, height);
        enqueue_tile_kernel("tile_stats", channels);
    }
    void compile_kernels(const std::string& text, const std::vector<std::string>& names,
                         cl_device_id device) {
        const char* s = text.c_str();
        size_t length = text.size();
        cl_int e;
        auto p = clCreateProgramWithSource(cl_context_handle, 1, &s, &length, &e);
        check_opencl(e);
        e = clBuildProgram(p, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
        if (e != CL_SUCCESS) {
            size_t bytes;
            clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &bytes);
            std::string log(bytes, '\0');
            clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, bytes, log.data(), nullptr);
            throw std::runtime_error(log);
        }
        programs.push_back(p);
        for (auto& name : names) {
            kernels[name] = clCreateKernel(p, name.c_str(), &e);
            check_opencl(e);
        }
    }

  public:
    MotionSpatialModel(uintptr_t c, uintptr_t parent, int width, int height, int frames,
                       std::array<int, 4> watermark, const std::string& motion_source,
                       const std::string& spatial_source)
        : cl_context_handle((cl_context)c), width(width), height(height),
          padded_width((width + 7) / 8 * 8), padded_height((height + 7) / 8 * 8),
          visible_pixels(width * height), padded_pixels(padded_width * padded_height),
          trace_capacity(frames), watermark_box(watermark) {
        check_opencl(clRetainContext(cl_context_handle));
        cl_device_id device;
        check_opencl(clGetCommandQueueInfo((cl_command_queue)parent, CL_QUEUE_DEVICE,
                                           sizeof(device), &device, nullptr));
        cl_int e;
        compute_queue = clCreateCommandQueue(cl_context_handle, device, 0, &e);
        check_opencl(e);
        low_area = std::max(1, int(std::floor(double(width) * height / (768 * 432) * 50 + .5)));
        strict_area = std::max(1, int(std::floor(double(width) * height / (768 * 432) * 100 + .5)));
        region_capacity = visible_pixels / low_area + 1;
        compile_kernels(
            motion_source,
            {"begin_frame", "learn", "reduce_noise", "histogram", "median_offset", "raw_mask"},
            device);
        compile_kernels(spatial_source,
                        {"normalize_masks", "morph", "tile_labels", "join_tiles", "flatten_init",
                         "tile_stats", "collect_low", "collect_single"},
                        device);
        groups = (padded_pixels + 4095) / 4096;
        reference_frame = allocate_device_buffer(padded_pixels);
        reference_age = allocate_device_buffer(padded_pixels);
        motion_residual = allocate_device_buffer(size_t(padded_pixels) * 2);
        motion_mask = allocate_device_buffer(padded_pixels);
        partial_statistics = allocate_device_buffer(size_t(groups) * 8);
        residual_histogram = allocate_device_buffer(512 * 4);
        parameters = allocate_device_buffer(sizeof(MotionParameters));
        frame_metrics = allocate_device_buffer(size_t(trace_capacity) * sizeof(MotionFrameMetrics));
        MotionParameters initial = {32, 6, 32, 32};
        check_opencl(clEnqueueWriteBuffer(compute_queue, parameters, CL_TRUE, 0, sizeof(initial),
                                          &initial, 0, nullptr, nullptr));
        packed_mask = allocate_device_buffer(visible_pixels);
        morphology_buffer_a = allocate_device_buffer(visible_pixels);
        morphology_buffer_b = allocate_device_buffer(visible_pixels);
        low_keep_mask = allocate_device_buffer(visible_pixels);
        strict_input_mask = allocate_device_buffer(visible_pixels);
        component_parents = allocate_device_buffer(size_t(visible_pixels + 1) * 2 * 4);
        tile_component_ids = allocate_device_buffer(size_t(visible_pixels) * 2 * 2);
        component_statistics = allocate_device_buffer(size_t(visible_pixels + 1) * 2 * 20);
        region_counts = allocate_device_buffer(12);
        region_buffer = allocate_device_buffer(size_t(region_capacity) * 3 * 20);
    }
    ~MotionSpatialModel() {
        clFinish(compute_queue);
        for (auto p : device_buffers) {
            clReleaseMemObject(p);
        }
        for (auto& k : kernels) {
            clReleaseKernel(k.second);
        }
        for (auto p : programs) {
            clReleaseProgram(p);
        }
        clReleaseCommandQueue(compute_queue);
        clReleaseContext(cl_context_handle);
    }
    std::string process(uintptr_t input, uintptr_t foreground, uint64_t stride, uint64_t offset) {
        py::gil_scoped_release release;
        if (update_count >= trace_capacity) {
            throw std::runtime_error("model trace capacity exceeded");
        }
        cl_uint index = update_count++;
        auto k = kernels.at("begin_frame");
        set_kernel_argument(k, 0, parameters);
        set_kernel_argument(k, 1, frame_metrics);
        set_kernel_argument(k, 2, index);
        enqueue_linear_kernel("begin_frame", 1, 1, 1);
        k = kernels.at("learn");
        set_kernel_argument(k, 0, (cl_mem)input);
        set_kernel_argument(k, 1, reference_frame);
        set_kernel_argument(k, 2, reference_age);
        set_kernel_argument(k, 3, motion_residual);
        set_kernel_argument(k, 4, partial_statistics);
        set_kernel_argument(k, 5, frame_metrics);
        set_kernel_argument(k, 6, index);
        set_kernel_argument(k, 7, padded_pixels);
        enqueue_linear_kernel("learn", size_t(groups) * 256);
        if (update_count % 5 == 0) {
            k = kernels.at("reduce_noise");
            set_kernel_argument(k, 0, partial_statistics);
            set_kernel_argument(k, 1, groups);
            set_kernel_argument(k, 2, parameters);
            set_kernel_argument(k, 3, frame_metrics);
            set_kernel_argument(k, 4, index);
            set_kernel_argument(k, 5, padded_pixels);
            set_kernel_argument(k, 6, cl_uint(strict_area));
            enqueue_linear_kernel("reduce_noise", 1, 1, 1);
        }
        if (index) {
            cl_uint zero = 0;
            check_opencl(clEnqueueFillBuffer(compute_queue, residual_histogram, &zero, 4, 0,
                                             512 * 4, 0, nullptr, nullptr));
            k = kernels.at("histogram");
            set_kernel_argument(k, 0, motion_residual);
            set_kernel_argument(k, 1, residual_histogram);
            set_kernel_argument(k, 2, width);
            set_kernel_argument(k, 3, height);
            set_kernel_argument(k, 4, padded_width);
            cl_uint samples = ((width + 15) / 16) * ((height + 15) / 16);
            enqueue_linear_kernel("histogram", samples);
            k = kernels.at("median_offset");
            set_kernel_argument(k, 0, residual_histogram);
            set_kernel_argument(k, 1, frame_metrics);
            set_kernel_argument(k, 2, index);
            set_kernel_argument(k, 3, samples);
            enqueue_linear_kernel("median_offset", 1, 1, 1);
            k = kernels.at("raw_mask");
            set_kernel_argument(k, 0, motion_residual);
            set_kernel_argument(k, 1, motion_mask);
            set_kernel_argument(k, 2, frame_metrics);
            set_kernel_argument(k, 3, index);
            set_kernel_argument(k, 4, width);
            set_kernel_argument(k, 5, height);
            set_kernel_argument(k, 6, padded_width);
            set_kernel_argument(k, 7, padded_pixels);
            for (int i = 0; i < 4; ++i) {
                set_kernel_argument(k, 8 + i, watermark_box[i]);
            }
            enqueue_linear_kernel("raw_mask", size_t(groups) * 256);
        }
        cl_uint zero = 0;
        check_opencl(clEnqueueFillBuffer(compute_queue, region_counts, &zero, 4, 0, 12, 0, nullptr,
                                         nullptr));
        k = kernels.at("normalize_masks");
        set_kernel_argument(k, 0, (cl_mem)foreground);
        set_kernel_argument(k, 1, cl_ulong(offset));
        set_kernel_argument(k, 2, cl_uint(stride));
        set_kernel_argument(k, 3, motion_mask);
        set_kernel_argument(k, 4, padded_width);
        set_kernel_argument(k, 5, packed_mask);
        set_kernel_argument(k, 6, width);
        set_kernel_argument(k, 7, height);
        for (int i = 0; i < 4; ++i) {
            set_kernel_argument(k, 8 + i, watermark_box[i]);
        }
        set_kernel_argument(k, 12, int(index == 0));
        enqueue_linear_kernel("normalize_masks", visible_pixels);
        cl_mem from = packed_mask, to = morphology_buffer_a;
        for (int pass = 0; pass < 4; ++pass) {
            k = kernels.at("morph");
            set_kernel_argument(k, 0, from);
            set_kernel_argument(k, 1, to);
            set_kernel_argument(k, 2, width);
            set_kernel_argument(k, 3, height);
            set_kernel_argument(k, 4, int(pass == 1 || pass == 2));
            set_kernel_argument(k, 5, int(pass < 2));
            enqueue_tile_kernel("morph");
            from = to;
            to = to == morphology_buffer_a ? morphology_buffer_b : morphology_buffer_a;
        }
        label_components(morphology_buffer_b, 2);
        k = kernels.at("collect_low");
        set_kernel_argument(k, 0, component_parents);
        set_kernel_argument(k, 1, component_statistics);
        set_kernel_argument(k, 2, morphology_buffer_b);
        set_kernel_argument(k, 3, low_keep_mask);
        set_kernel_argument(k, 4, strict_input_mask);
        set_kernel_argument(k, 5, region_counts);
        set_kernel_argument(k, 6, region_buffer);
        set_kernel_argument(k, 7, width);
        set_kernel_argument(k, 8, height);
        set_kernel_argument(k, 9, low_area);
        set_kernel_argument(k, 10, region_capacity);
        enqueue_linear_kernel("collect_low", visible_pixels);
        label_components(strict_input_mask, 1);
        k = kernels.at("collect_single");
        set_kernel_argument(k, 0, component_parents);
        set_kernel_argument(k, 1, component_statistics);
        set_kernel_argument(k, 2, region_counts);
        set_kernel_argument(k, 3, region_buffer);
        set_kernel_argument(k, 4, visible_pixels);
        set_kernel_argument(k, 5, strict_area);
        set_kernel_argument(k, 6, region_capacity);
        enqueue_linear_kernel("collect_single", visible_pixels);
        std::array<cl_uint, 3> sizes;
        check_opencl(clEnqueueReadBuffer(compute_queue, region_counts, CL_TRUE, 0, 12, sizes.data(),
                                         0, nullptr, nullptr));
        compact_read += 12;
        std::array<Json, 3> branch_regions;
        for (int channel = 0; channel < 3; ++channel) {
            if (sizes[channel] > region_capacity) {
                throw std::runtime_error("region capacity invariant violated");
            }
            std::vector<ComponentRegion> values(sizes[channel]);
            size_t bytes = values.size() * sizeof(ComponentRegion);
            if (bytes) {
                check_opencl(
                    clEnqueueReadBuffer(compute_queue, region_buffer, CL_TRUE,
                                        size_t(channel) * region_capacity * sizeof(ComponentRegion),
                                        bytes, values.data(), 0, nullptr, nullptr));
                compact_read += bytes;
            }
            std::sort(values.begin(), values.end(), [](auto& x, auto& y) {
                return x.area != y.area ? x.area > y.area
                                        : std::array<int, 4>{x.x, x.y, x.width, x.height} <
                                              std::array<int, 4>{y.x, y.y, y.width, y.height};
            });
            branch_regions[channel] = Json::array();
            for (auto& r : values) {
                branch_regions[channel].push_back(
                    {{"box", {r.x, r.y, r.width, r.height}}, {"area", r.area}});
            }
        }
        return Json({{"mog_pass", sizes[0] > 0},
                     {"motion_pass", sizes[2] > 0},
                     {"has_candidate", sizes[0] > 0 || sizes[1] > 0},
                     {"warming", update_count <= 5},
                     {"regions",
                      {{"mog", branch_regions[0]},
                       {"motion_low", branch_regions[1]},
                       {"motion_high", branch_regions[2]}}}})
            .dump();
    }
    std::string model_trace() {
        std::vector<MotionFrameMetrics> data(update_count);
        check_opencl(clEnqueueReadBuffer(compute_queue, frame_metrics, CL_TRUE, 0,
                                         data.size() * sizeof(MotionFrameMetrics), data.data(), 0,
                                         nullptr, nullptr));
        trace_read += data.size() * sizeof(MotionFrameMetrics);
        Json out = Json::array();
        for (auto& m : data) {
            out.push_back({{"seq", m.seq},
                           {"active_threshold", m.active_t},
                           {"low_threshold", m.low_t},
                           {"raw_threshold", m.raw_t},
                           {"noise_used", m.before},
                           {"noise_after", m.after},
                           {"offset", m.seq == 1 ? Json(nullptr) : Json(m.offset)},
                           {"active_count", m.active},
                           {"quiet_sum", m.quiet_sum},
                           {"update_applied", bool(m.applied)}});
        }
        return out.dump();
    }
    std::string metrics() {
        return Json({{"updates", update_count},
                     {"precision", "FP32 noise; integer reference/age/residual"},
                     {"allocated_bytes", allocated},
                     {"compact_read_bytes", compact_read},
                     {"trace_read_bytes", trace_read},
                     {"raw_read_bytes", 0},
                     {"foreground_read_bytes", 0}})
            .dump();
    }
};
// Keep the Python binding name stable while naming the implementation by its role.
PYBIND11_MODULE(_motion_gpu_math, m) {
    py::class_<MotionSpatialModel>(m, "Math")
        .def(py::init<uintptr_t, uintptr_t, int, int, int, std::array<int, 4>, const std::string&,
                      const std::string&>())
        .def("process", &MotionSpatialModel::process)
        .def("model_trace", &MotionSpatialModel::model_trace)
        .def("metrics", &MotionSpatialModel::metrics);
}
