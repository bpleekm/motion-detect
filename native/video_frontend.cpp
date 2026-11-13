#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <CL/cl_va_api_media_sharing_intel.h>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/video/background_segm.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <nlohmann/json.hpp>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vaapi.h>
#include <libswscale/swscale.h>
}
#include <array>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Json = nlohmann::json;
static void check_opencl(cl_int e) {
    if (e != CL_SUCCESS) {
        throw std::runtime_error("OpenCL error " + std::to_string(e));
    }
}
static void check_ffmpeg(int e) {
    if (e < 0) {
        char text[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(e, text, sizeof(text));
        throw std::runtime_error(text);
    }
}
static int64_t pts_to_milliseconds(int64_t start, int64_t pts, int64_t first, AVRational tb) {
    __int128 num = (__int128)start * tb.den + (__int128)(pts - first) * tb.num * 1000000,
             den = (__int128)1000 * tb.den;
    bool negative = num < 0;
    if (negative) {
        num = -num;
    }
    __int128 q = num / den, r = num % den;
    if (r * 2 > den || (r * 2 == den && q % 2)) {
        ++q;
    }
    return negative ? -int64_t(q) : int64_t(q);
}
static AVPixelFormat select_vaapi_format(AVCodecContext*, const AVPixelFormat* formats) {
    for (auto p = formats; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == AV_PIX_FMT_VAAPI) {
            return *p;
        }
    }
    return AV_PIX_FMT_NONE;
}
struct DecoderGeneration {
    AVCodecContext* codec = nullptr;
    std::shared_ptr<std::recursive_mutex> mutex;
    ~DecoderGeneration() {
        std::lock_guard<std::recursive_mutex> lock(*mutex);
        avcodec_free_context(&codec);
    }
};
struct FrameLease {
    AVFrame* frame = nullptr;
    std::shared_ptr<DecoderGeneration> generation;
    Json metadata;
    int slot = 0;
    ~FrameLease() {
        av_frame_free(&frame);
    }
};

class VideoFrontend {
    Json configuration, files = Json::array();
    int width, height, padded_width, padded_height;
    std::array<int, 4> watermark_box;
    AVBufferRef* va_device = nullptr;
    VADisplay display = nullptr;
    cl_platform_id platform = nullptr;
    cl_device_id cl_device = nullptr;
    cl_context context = nullptr;
    cl_command_queue preprocess_queue = nullptr;
    cl_program preprocess_program = nullptr;
    std::map<std::string, cl_kernel> kernels;
    cl_mem blur_intermediate = nullptr, gray_lookup = nullptr;
    cv::UMat gray_frame, prepared_frame, foreground_mask;
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog_model;
    clCreateFromVA_APIMediaSurfaceINTEL_fn import_va = nullptr;
    clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn acquire_va = nullptr;
    clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn release_va = nullptr;
    std::array<VASurfaceID, 8> surface_ids = {};
    std::array<uint64_t, 8> slot_generations = {};
    std::shared_ptr<std::recursive_mutex> va_mutex = std::make_shared<std::recursive_mutex>();
    std::mutex mutex;
    std::condition_variable frame_available;
    std::deque<int> available_slots;
    std::deque<std::shared_ptr<FrameLease>> ready_frames;
    std::shared_ptr<FrameLease> current_frame;
    std::thread decode_thread;
    std::atomic<bool> stopped{false};
    bool done = false;
    std::string failure;
    std::atomic<uint64_t> decoded_count{0}, selected_count{0}, surface_sync_count{0};
    uint64_t model_update_count = 0, resident_count = 0, resident_peak = 0, imported_images = 0,
             acquires = 0, releases = 0;
    uint64_t person_bgr_bytes = 0, person_nv12_bytes = 0;
    template <class T> void set_kernel_argument(cl_kernel k, int i, T value) {
        check_opencl(clSetKernelArg(k, i, sizeof(value), &value));
    }
    void enqueue_kernel(const std::string& name, int width, int height) {
        size_t global[2] = {(width + 15) / 16 * 16ul, (height + 15) / 16 * 16ul},
               local[2] = {16, 16};
        check_opencl(clEnqueueNDRangeKernel(preprocess_queue, kernels.at(name), 2, nullptr, global,
                                            local, 0, nullptr, nullptr));
    }
    int acquire_slot() {
        std::unique_lock<std::mutex> lock(mutex);
        frame_available.wait(lock, [&] { return stopped || !available_slots.empty(); });
        if (stopped) {
            return -1;
        }
        int slot = available_slots.front();
        available_slots.pop_front();
        ++slot_generations[slot];
        return slot;
    }
    void release_slot(int slot) {
        std::lock_guard<std::mutex> lock(mutex);
        available_slots.push_back(slot);
        frame_available.notify_all();
    }
    void decode_frames() {
        try {
            std::ofstream index(configuration.at("output").get<std::string>() + "/frames.jsonl");
            int64_t last_selected_ms = -1;
            for (auto& source_file : configuration.at("files")) {
                if (stopped) {
                    break;
                }
                AVFormatContext* format = nullptr;
                check_ffmpeg(avformat_open_input(
                    &format, source_file.at("path").get<std::string>().c_str(), nullptr, nullptr));
                auto decoder_generation = std::make_shared<DecoderGeneration>();
                decoder_generation->mutex = va_mutex;
                int stream_index = source_file.at("stream_index");
                AVStream* stream = format->streams[stream_index];
                auto codec = avcodec_find_decoder(stream->codecpar->codec_id);
                decoder_generation->codec = avcodec_alloc_context3(codec);
                check_ffmpeg(
                    avcodec_parameters_to_context(decoder_generation->codec, stream->codecpar));
                decoder_generation->codec->pkt_timebase = stream->time_base;
                decoder_generation->codec->hw_device_ctx = av_buffer_ref(va_device);
                decoder_generation->codec->get_format = select_vaapi_format;
                decoder_generation->codec->extra_hw_frames = 8;
                decoder_generation->codec->thread_count = 1;
                decoder_generation->codec->thread_type = 0;
                {
                    std::lock_guard<std::recursive_mutex> lock(*va_mutex);
                    check_ffmpeg(avcodec_open2(decoder_generation->codec, codec, nullptr));
                }
                Json summary = {{"file_id", source_file.at("file_id")},
                                {"decoded_frames", 0},
                                {"selected_frames", 0},
                                {"decoder_eof", false},
                                {"demux_eof", false}};
                AVPacket* packet = av_packet_alloc();
                auto receive = [&]() {
                    while (!stopped) {
                        int slot = acquire_slot();
                        if (slot < 0) {
                            return AVERROR_EXIT;
                        }
                        AVFrame* frame = av_frame_alloc();
                        int error;
                        {
                            std::lock_guard<std::recursive_mutex> lock(*va_mutex);
                            error = avcodec_receive_frame(decoder_generation->codec, frame);
                            if (error == 0) {
                                auto s =
                                    vaSyncSurface(display, (VASurfaceID)(uintptr_t)frame->data[3]);
                                if (s != VA_STATUS_SUCCESS) {
                                    throw std::runtime_error("decode completion sync failed");
                                }
                                ++surface_sync_count;
                            }
                        }
                        if (error < 0) {
                            av_frame_free(&frame);
                            release_slot(slot);
                            return error;
                        }
                        if (frame->format != AV_PIX_FMT_VAAPI || frame->pts == AV_NOPTS_VALUE ||
                            frame->decode_error_flags) {
                            throw std::runtime_error("invalid hardware frame");
                        }
                        int visible_width =
                                frame->width - int(frame->crop_left + frame->crop_right),
                            visible_height =
                                frame->height - int(frame->crop_top + frame->crop_bottom);
                        if (visible_width != width || visible_height != height) {
                            throw std::runtime_error("visible geometry changed");
                        }
                        int64_t time_ms = pts_to_milliseconds(
                            source_file.at("start_us"), frame->pts,
                            source_file.at("source_start_pts"), stream->time_base);
                        bool chosen =
                            last_selected_ms < 0 ||
                            time_ms - last_selected_ms >= configuration.value("sample_ms", 200);
                        ++decoded_count;
                        summary["decoded_frames"] = summary["decoded_frames"].get<int>() + 1;
                        Json row = {{"frame_id", source_file.at("file_id").get<std::string>() +
                                                     ":s" + std::to_string(stream_index) + ":pts" +
                                                     std::to_string(frame->pts)},
                                    {"file_id", source_file.at("file_id")},
                                    {"stream_index", stream_index},
                                    {"pts", frame->pts},
                                    {"time_base", {stream->time_base.num, stream->time_base.den}},
                                    {"time_ms", time_ms},
                                    {"width", visible_width},
                                    {"height", visible_height},
                                    {"key_frame", bool(frame->key_frame)},
                                    {"slot_id", slot},
                                    {"generation", slot_generations[slot]}};
                        index << row.dump() << '\n';
                        if (chosen) {
                            auto p = std::make_shared<FrameLease>();
                            p->frame = frame;
                            p->generation = decoder_generation;
                            p->slot = slot;
                            p->metadata = row;
                            p->metadata["seq"] = ++selected_count;
                            last_selected_ms = time_ms;
                            auto* pool = (AVHWFramesContext*)frame->hw_frames_ctx->data;
                            summary["pool_size"] = pool->initial_pool_size;
                            summary["selected_frames"] = summary["selected_frames"].get<int>() + 1;
                            std::lock_guard<std::mutex> lock(mutex);
                            ready_frames.push_back(p);
                            ++resident_count;
                            resident_peak = std::max(resident_peak, resident_count);
                            frame_available.notify_all();
                        } else {
                            av_frame_free(&frame);
                            release_slot(slot);
                        }
                    }
                    return AVERROR_EXIT;
                };
                int error = 0;
                while (!stopped && (error = av_read_frame(format, packet)) >= 0) {
                    if (packet->stream_index == stream_index) {
                        int sent;
                        {
                            std::lock_guard<std::recursive_mutex> lock(*va_mutex);
                            sent = avcodec_send_packet(decoder_generation->codec, packet);
                        }
                        if (sent == AVERROR(EAGAIN)) {
                            int r = receive();
                            if (r != AVERROR(EAGAIN)) {
                                check_ffmpeg(r);
                            }
                            std::lock_guard<std::recursive_mutex> lock(*va_mutex);
                            sent = avcodec_send_packet(decoder_generation->codec, packet);
                        }
                        check_ffmpeg(sent);
                        int r = receive();
                        if (r != AVERROR(EAGAIN)) {
                            check_ffmpeg(r);
                        }
                    }
                    av_packet_unref(packet);
                }
                if (!stopped) {
                    if (error != AVERROR_EOF) {
                        check_ffmpeg(error);
                    }
                    summary["demux_eof"] = true;
                    {
                        std::lock_guard<std::recursive_mutex> lock(*va_mutex);
                        check_ffmpeg(avcodec_send_packet(decoder_generation->codec, nullptr));
                    }
                    int r = receive();
                    if (r != AVERROR_EOF) {
                        check_ffmpeg(r);
                    }
                    summary["decoder_eof"] = true;
                }
                av_packet_free(&packet);
                avformat_close_input(&format);
                files.push_back(summary);
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(mutex);
            failure = e.what();
        }
        std::lock_guard<std::mutex> lock(mutex);
        done = true;
        frame_available.notify_all();
    }

  public:
    VideoFrontend(const std::string& configuration_json, const std::string& shader)
        : configuration(Json::parse(configuration_json)), width(configuration.at("width")),
          height(configuration.at("height")), padded_width((width + 7) / 8 * 8),
          padded_height((height + 7) / 8 * 8),
          watermark_box(configuration.at("watermark").get<std::array<int, 4>>()) {
        cv::setNumThreads(2);
        auto render = configuration.value("render_device", std::string("/dev/dri/renderD128"));
        check_ffmpeg(
            av_hwdevice_ctx_create(&va_device, AV_HWDEVICE_TYPE_VAAPI, render.c_str(), nullptr, 0));
        auto* hw = (AVHWDeviceContext*)va_device->data;
        display = ((AVVAAPIDeviceContext*)hw->hwctx)->display;
        cl_uint count;
        check_opencl(clGetPlatformIDs(0, nullptr, &count));
        std::vector<cl_platform_id> platforms(count);
        check_opencl(clGetPlatformIDs(count, platforms.data(), nullptr));
        for (auto p : platforms) {
            auto get = (clGetDeviceIDsFromVA_APIMediaAdapterINTEL_fn)
                clGetExtensionFunctionAddressForPlatform(
                    p, "clGetDeviceIDsFromVA_APIMediaAdapterINTEL");
            cl_uint found = 0;
            if (get &&
                get(p, CL_VA_API_DISPLAY_INTEL, display, CL_PREFERRED_DEVICES_FOR_VA_API_INTEL, 1,
                    &cl_device, &found) == CL_SUCCESS &&
                found) {
                platform = p;
                break;
            }
        }
        if (!platform) {
            throw std::runtime_error("VA/OpenCL sharing unavailable");
        }
        cl_context_properties props[] = {CL_CONTEXT_PLATFORM,
                                         (cl_context_properties)platform,
                                         CL_CONTEXT_VA_API_DISPLAY_INTEL,
                                         (cl_context_properties)display,
                                         CL_CONTEXT_INTEROP_USER_SYNC,
                                         CL_FALSE,
                                         0};
        cl_int e;
        context = clCreateContext(props, 1, &cl_device, nullptr, nullptr, &e);
        check_opencl(e);
        char name[256];
        check_opencl(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), name, nullptr));
        cv::ocl::attachContext(name, platform, context, cl_device);
        cv::ocl::setUseOpenCL(true);
        if (!cv::ocl::useOpenCL()) {
            throw std::runtime_error("private OpenCL disabled");
        }
        preprocess_queue = clCreateCommandQueue(context, cl_device, 0, &e);
        check_opencl(e);
        import_va =
            (clCreateFromVA_APIMediaSurfaceINTEL_fn)clGetExtensionFunctionAddressForPlatform(
                platform, "clCreateFromVA_APIMediaSurfaceINTEL");
        acquire_va =
            (clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn)clGetExtensionFunctionAddressForPlatform(
                platform, "clEnqueueAcquireVA_APIMediaSurfacesINTEL");
        release_va =
            (clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn)clGetExtensionFunctionAddressForPlatform(
                platform, "clEnqueueReleaseVA_APIMediaSurfacesINTEL");
        if (!import_va || !acquire_va || !release_va) {
            throw std::runtime_error("VA sharing entrypoint missing");
        }
        gray_frame.create(height, width, CV_8U);
        prepared_frame.create(padded_height, padded_width, CV_8U);
        foreground_mask.create(padded_height, padded_width, CV_8U);
        if (!gray_frame.isContinuous() || !prepared_frame.isContinuous()) {
            throw std::runtime_error("non-linear private GPU buffers");
        }
        blur_intermediate =
            clCreateBuffer(context, CL_MEM_READ_WRITE, size_t(width) * height * 2, nullptr, &e);
        check_opencl(e);
        auto table = configuration.at("gray_lut").get<std::vector<uint8_t>>();
        if (table.size() != 256) {
            throw std::runtime_error("invalid gray LUT");
        }
        gray_lookup =
            clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 256, table.data(), &e);
        check_opencl(e);
        const char* text = shader.c_str();
        size_t size = shader.size();
        preprocess_program = clCreateProgramWithSource(context, 1, &text, &size, &e);
        check_opencl(e);
        e = clBuildProgram(preprocess_program, 1, &cl_device, "-cl-std=CL1.2", nullptr, nullptr);
        if (e != CL_SUCCESS) {
            size_t bytes;
            clGetProgramBuildInfo(preprocess_program, cl_device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                                  &bytes);
            std::string log(bytes, '\0');
            clGetProgramBuildInfo(preprocess_program, cl_device, CL_PROGRAM_BUILD_LOG, bytes,
                                  log.data(), nullptr);
            throw std::runtime_error(log);
        }
        for (auto key : {"gray_from_y", "horizontal9", "vertical9_mask"}) {
            kernels[key] = clCreateKernel(preprocess_program, key, &e);
            check_opencl(e);
        }
        mog_model = cv::createBackgroundSubtractorMOG2(150, 16, true);
        for (int i = 0; i < 8; ++i) {
            available_slots.push_back(i);
        }
        decode_thread = std::thread([this] { decode_frames(); });
    }
    ~VideoFrontend() {
        stopped = true;
        frame_available.notify_all();
        if (decode_thread.joinable()) {
            decode_thread.join();
        }
        current_frame.reset();
        ready_frames.clear();
        clFinish(preprocess_queue);
        mog_model.release();
        gray_frame.release();
        prepared_frame.release();
        foreground_mask.release();
        for (auto& k : kernels) {
            clReleaseKernel(k.second);
        }
        clReleaseMemObject(blur_intermediate);
        clReleaseMemObject(gray_lookup);
        clReleaseProgram(preprocess_program);
        clReleaseCommandQueue(preprocess_queue);
        clReleaseContext(context);
        av_buffer_unref(&va_device);
    }
    bool next() {
        py::gil_scoped_release release;
        if (current_frame) {
            throw std::runtime_error("lease not released");
        }
        std::unique_lock<std::mutex> lock(mutex);
        frame_available.wait(lock, [&] { return done || !ready_frames.empty(); });
        if (!failure.empty()) {
            throw std::runtime_error(failure);
        }
        if (ready_frames.empty()) {
            lock.unlock();
            decode_thread.join();
            return false;
        }
        current_frame = ready_frames.front();
        ready_frames.pop_front();
        return true;
    }
    void preprocess() {
        py::gil_scoped_release release;
        if (!current_frame) {
            throw std::runtime_error("missing lease");
        }
        AVFrame* frame = current_frame->frame;
        std::unique_lock<std::recursive_mutex> lock(*va_mutex);
        surface_ids[current_frame->slot] = (VASurfaceID)(uintptr_t)frame->data[3];
        if (vaSyncSurface(display, surface_ids[current_frame->slot]) != VA_STATUS_SUCCESS) {
            throw std::runtime_error("source sync failed");
        }
        cl_int e;
        cl_mem image =
            import_va(context, CL_MEM_READ_ONLY, &surface_ids[current_frame->slot], 0, &e);
        check_opencl(e);
        ++imported_images;
        check_opencl(acquire_va(preprocess_queue, 1, &image, 0, nullptr, nullptr));
        ++acquires;
        auto gray_mem = (cl_mem)gray_frame.handle(cv::ACCESS_WRITE);
        auto k = kernels.at("gray_from_y");
        set_kernel_argument(k, 0, image);
        set_kernel_argument(k, 1, gray_mem);
        set_kernel_argument(k, 2, gray_lookup);
        set_kernel_argument(k, 3, width);
        set_kernel_argument(k, 4, height);
        set_kernel_argument(k, 5, int(frame->crop_left));
        set_kernel_argument(k, 6, int(frame->crop_top));
        enqueue_kernel("gray_from_y", width, height);
        cl_event event;
        check_opencl(release_va(preprocess_queue, 1, &image, 0, nullptr, &event));
        check_opencl(clWaitForEvents(1, &event));
        clReleaseEvent(event);
        check_opencl(clReleaseMemObject(image));
        ++releases;
        lock.unlock();
        k = kernels.at("horizontal9");
        set_kernel_argument(k, 0, gray_mem);
        set_kernel_argument(k, 1, blur_intermediate);
        set_kernel_argument(k, 2, width);
        set_kernel_argument(k, 3, height);
        enqueue_kernel("horizontal9", width, height);
        auto target = (cl_mem)prepared_frame.handle(cv::ACCESS_WRITE);
        k = kernels.at("vertical9_mask");
        set_kernel_argument(k, 0, blur_intermediate);
        set_kernel_argument(k, 1, target);
        set_kernel_argument(k, 2, width);
        set_kernel_argument(k, 3, height);
        set_kernel_argument(k, 4, padded_width);
        set_kernel_argument(k, 5, padded_height);
        for (int i = 0; i < 4; ++i) {
            set_kernel_argument(k, 6 + i, watermark_box[i]);
        }
        enqueue_kernel("vertical9_mask", padded_width, padded_height);
        check_opencl(clFinish(preprocess_queue));
        ++model_update_count;
        mog_model->apply(prepared_frame, foreground_mask,
                         model_update_count == 1 ? 1. : .006644493744965563);
        cv::ocl::finish();
    }
    void release() {
        if (!current_frame) {
            return;
        }
        int slot = current_frame->slot;
        current_frame.reset();
        {
            std::lock_guard<std::mutex> lock(mutex);
            --resident_count;
        }
        release_slot(slot);
    }
    std::string metadata() {
        return current_frame->metadata.dump();
    }
    py::array_t<uint8_t> bgr() {
        if (!current_frame) {
            throw std::runtime_error("missing person frame lease");
        }
        AVFrame* downloaded = av_frame_alloc();
        AVFrame* planar = av_frame_alloc();
        {
            std::lock_guard<std::recursive_mutex> lock(*va_mutex);
            check_ffmpeg(av_hwframe_transfer_data(downloaded, current_frame->frame, 0));
            check_ffmpeg(av_frame_copy_props(downloaded, current_frame->frame));
        }
        if (downloaded->format != AV_PIX_FMT_NV12) {
            throw std::runtime_error("person download is not NV12");
        }
        person_nv12_bytes += uint64_t(width) * height * 3 / 2;
        planar->format = AV_PIX_FMT_YUV420P;
        planar->width = width;
        planar->height = height;
        check_ffmpeg(av_frame_get_buffer(planar, 32));
        for (int y = 0; y < height; ++y) {
            memcpy(planar->data[0] + y * planar->linesize[0],
                   downloaded->data[0] + (y + downloaded->crop_top) * downloaded->linesize[0] +
                       downloaded->crop_left,
                   width);
        }
        for (int y = 0; y < height / 2; ++y) {
            for (int x = 0; x < width / 2; ++x) {
                auto p = downloaded->data[1] +
                         (y + downloaded->crop_top / 2) * downloaded->linesize[1] +
                         downloaded->crop_left + x * 2;
                planar->data[1][y * planar->linesize[1] + x] = p[0];
                planar->data[2][y * planar->linesize[2] + x] = p[1];
            }
        }
        py::array_t<uint8_t> image({py::ssize_t(height), py::ssize_t(width), py::ssize_t(3)});
        uint8_t* target[] = {image.mutable_data(), nullptr, nullptr, nullptr};
        int pitch[] = {width * 3, 0, 0, 0};
        auto context = sws_getContext(width, height, AV_PIX_FMT_YUV420P, width, height,
                                      AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!context) {
            throw std::runtime_error("person color context failed");
        }
        int rows = sws_scale(context, planar->data, planar->linesize, 0, height, target, pitch);
        sws_freeContext(context);
        av_frame_free(&downloaded);
        av_frame_free(&planar);
        if (rows != height) {
            throw std::runtime_error("incomplete person color frame");
        }
        person_bgr_bytes += uint64_t(width) * height * 3;
        return image;
    }
    uintptr_t context_handle() {
        return (uintptr_t)context;
    }
    uintptr_t queue_handle() {
        return (uintptr_t)preprocess_queue;
    }
    uintptr_t prepared_handle() {
        return (uintptr_t)prepared_frame.handle(cv::ACCESS_READ);
    }
    uintptr_t foreground_handle() {
        return (uintptr_t)foreground_mask.handle(cv::ACCESS_READ);
    }
    uint64_t stride() {
        return foreground_mask.step[0];
    }
    uint64_t offset() {
        return foreground_mask.offset;
    }
    std::string metrics() {
        return Json({{"decoded_frames", decoded_count.load()},
                     {"selected_frames", selected_count.load()},
                     {"model_updates", model_update_count},
                     {"surface_syncs", surface_sync_count.load()},
                     {"resident_peak", resident_peak},
                     {"resident_at_end", resident_count},
                     {"imports", imported_images},
                     {"acquires", acquires},
                     {"releases", releases},
                     {"host_y_upload_bytes", 0},
                     {"person_bgr_bytes", person_bgr_bytes},
                     {"person_nv12_download_bytes", person_nv12_bytes},
                     {"files", files}})
            .dump();
    }
};
// Keep the Python binding name stable while naming the implementation by its role.
PYBIND11_MODULE(_motion_gpu_front, m) {
    py::class_<VideoFrontend>(m, "Engine")
        .def(py::init<const std::string&, const std::string&>())
        .def("next", &VideoFrontend::next)
        .def("preprocess", &VideoFrontend::preprocess)
        .def("release", &VideoFrontend::release)
        .def("metadata", &VideoFrontend::metadata)
        .def("context_handle", &VideoFrontend::context_handle)
        .def("queue_handle", &VideoFrontend::queue_handle)
        .def("prepared_handle", &VideoFrontend::prepared_handle)
        .def("foreground_handle", &VideoFrontend::foreground_handle)
        .def("stride", &VideoFrontend::stride)
        .def("offset", &VideoFrontend::offset)
        .def("bgr", &VideoFrontend::bgr)
        .def("metrics", &VideoFrontend::metrics);
}
