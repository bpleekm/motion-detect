#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <CL/cl_va_api_media_sharing_intel.h>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/video/background_segm.hpp>
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vaapi.h>
}
#include <array>
#include <algorithm>
#include <atomic>
#include <cstdint>
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

namespace py=pybind11;
using Json=nlohmann::json;
static void clcheck(cl_int e){if(e!=CL_SUCCESS)throw std::runtime_error("OpenCL error "+std::to_string(e));}
static void avcheck(int e){if(e<0){char text[AV_ERROR_MAX_STRING_SIZE];av_strerror(e,text,sizeof(text));throw std::runtime_error(text);}}
static int64_t milliseconds(int64_t start,int64_t pts,int64_t first,AVRational tb){
    __int128 num=(__int128)start*tb.den+(__int128)(pts-first)*tb.num*1000000,den=(__int128)1000*tb.den;
    bool negative=num<0;if(negative)num=-num;__int128 q=num/den,r=num%den;if(r*2>den || (r*2==den && q%2))++q;return negative?-int64_t(q):int64_t(q);
}
static AVPixelFormat hardware_format(AVCodecContext*,const AVPixelFormat* formats){for(auto p=formats;*p!=AV_PIX_FMT_NONE;++p)if(*p==AV_PIX_FMT_VAAPI)return *p;return AV_PIX_FMT_NONE;}
struct Generation{AVCodecContext* codec=nullptr;std::shared_ptr<std::recursive_mutex> mutex;
    ~Generation(){std::lock_guard<std::recursive_mutex> lock(*mutex);avcodec_free_context(&codec);}};
struct Lease{AVFrame* frame=nullptr;std::shared_ptr<Generation> generation;Json metadata;int slot=0;~Lease(){av_frame_free(&frame);}};

class Engine{
    Json cfg,files=Json::array();int w,h,pw,ph;std::array<int,4> wm;
    AVBufferRef* device=nullptr;VADisplay display=nullptr;cl_platform_id platform=nullptr;cl_device_id cldevice=nullptr;
    cl_context context=nullptr;cl_command_queue queue=nullptr;cl_program program=nullptr;std::map<std::string,cl_kernel> kernels;
    cl_mem temporary=nullptr,lut=nullptr;cv::UMat gray,prepared,foreground;cv::Ptr<cv::BackgroundSubtractorMOG2> mog;
    clCreateFromVA_APIMediaSurfaceINTEL_fn import_va=nullptr;clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn acquire_va=nullptr;clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn release_va=nullptr;
    std::array<VASurfaceID,8> surface_ids={};std::array<uint64_t,8> generations={};
    std::shared_ptr<std::recursive_mutex> va_mutex=std::make_shared<std::recursive_mutex>();std::mutex mutex;std::condition_variable condition;
    std::deque<int> free_slots;std::deque<std::shared_ptr<Lease>> ready;std::shared_ptr<Lease> current;std::thread producer;
    std::atomic<bool> stopped{false};bool done=false;std::string failure;
    std::atomic<uint64_t> decoded{0},selected{0},syncs{0};uint64_t updates=0,resident=0,peak=0,imports=0,acquires=0,releases=0;
    template<class T>void arg(cl_kernel k,int i,T value){clcheck(clSetKernelArg(k,i,sizeof(value),&value));}
    void launch(const std::string& name,int width,int height){size_t global[2]={(width+15)/16*16ul,(height+15)/16*16ul},local[2]={16,16};clcheck(clEnqueueNDRangeKernel(queue,kernels.at(name),2,nullptr,global,local,0,nullptr,nullptr));}
    int take(){std::unique_lock<std::mutex> lock(mutex);condition.wait(lock,[&]{return stopped || !free_slots.empty();});if(stopped)return -1;
        int slot=free_slots.front();free_slots.pop_front();++generations[slot];return slot;}
    void give(int slot){std::lock_guard<std::mutex> lock(mutex);free_slots.push_back(slot);condition.notify_all();}
    void produce(){
        try{
            std::ofstream index(cfg.at("output").get<std::string>()+"/frames.jsonl");int64_t last=-1;
            for(auto& item:cfg.at("files")){
                if(stopped)break;AVFormatContext* format=nullptr;avcheck(avformat_open_input(&format,item.at("path").get<std::string>().c_str(),nullptr,nullptr));
                auto gen=std::make_shared<Generation>();gen->mutex=va_mutex;int stream_index=item.at("stream_index");AVStream* stream=format->streams[stream_index];
                auto codec=avcodec_find_decoder(stream->codecpar->codec_id);gen->codec=avcodec_alloc_context3(codec);avcheck(avcodec_parameters_to_context(gen->codec,stream->codecpar));
                gen->codec->pkt_timebase=stream->time_base;gen->codec->hw_device_ctx=av_buffer_ref(device);gen->codec->get_format=hardware_format;
                gen->codec->extra_hw_frames=8;gen->codec->thread_count=1;gen->codec->thread_type=0;
                {std::lock_guard<std::recursive_mutex> lock(*va_mutex);avcheck(avcodec_open2(gen->codec,codec,nullptr));}
                Json summary={{"file_id",item.at("file_id")},{"decoded_frames",0},{"selected_frames",0},{"decoder_eof",false},{"demux_eof",false}};
                AVPacket* packet=av_packet_alloc();
                auto receive=[&](){while(!stopped){int slot=take();if(slot<0)return AVERROR_EXIT;AVFrame* frame=av_frame_alloc();int error;
                    {std::lock_guard<std::recursive_mutex> lock(*va_mutex);error=avcodec_receive_frame(gen->codec,frame);
                        if(error==0){auto s=vaSyncSurface(display,(VASurfaceID)(uintptr_t)frame->data[3]);if(s!=VA_STATUS_SUCCESS)throw std::runtime_error("decode completion sync failed");++syncs;}}
                    if(error<0){av_frame_free(&frame);give(slot);return error;}
                    if(frame->format!=AV_PIX_FMT_VAAPI || frame->pts==AV_NOPTS_VALUE || frame->decode_error_flags)throw std::runtime_error("invalid hardware frame");
                    int vw=frame->width-int(frame->crop_left+frame->crop_right),vh=frame->height-int(frame->crop_top+frame->crop_bottom);
                    if(vw!=w || vh!=h)throw std::runtime_error("visible geometry changed");
                    int64_t ms=milliseconds(item.at("start_us"),frame->pts,item.at("source_start_pts"),stream->time_base);
                    bool chosen=last<0 || ms-last>=cfg.value("sample_ms",200);++decoded;summary["decoded_frames"]=summary["decoded_frames"].get<int>()+1;
                    Json row={{"frame_id",item.at("file_id").get<std::string>()+":s"+std::to_string(stream_index)+":pts"+std::to_string(frame->pts)},
                        {"file_id",item.at("file_id")},{"stream_index",stream_index},{"pts",frame->pts},{"time_base",{stream->time_base.num,stream->time_base.den}},
                        {"time_ms",ms},{"width",vw},{"height",vh},{"key_frame",bool(frame->key_frame)},{"slot_id",slot},{"generation",generations[slot]}};
                    index<<row.dump()<<'\n';
                    if(chosen){auto p=std::make_shared<Lease>();p->frame=frame;p->generation=gen;p->slot=slot;p->metadata=row;p->metadata["seq"]=++selected;last=ms;
                        auto* pool=(AVHWFramesContext*)frame->hw_frames_ctx->data;summary["pool_size"]=pool->initial_pool_size;
                        summary["selected_frames"]=summary["selected_frames"].get<int>()+1;
                        std::lock_guard<std::mutex> lock(mutex);ready.push_back(p);++resident;peak=std::max(peak,resident);condition.notify_all();
                    }else{av_frame_free(&frame);give(slot);}}
                    return AVERROR_EXIT;};
                int error=0;
                while(!stopped && (error=av_read_frame(format,packet))>=0){if(packet->stream_index==stream_index){int sent;
                    {std::lock_guard<std::recursive_mutex> lock(*va_mutex);sent=avcodec_send_packet(gen->codec,packet);}
                    if(sent==AVERROR(EAGAIN)){int r=receive();if(r!=AVERROR(EAGAIN))avcheck(r);std::lock_guard<std::recursive_mutex> lock(*va_mutex);sent=avcodec_send_packet(gen->codec,packet);}
                    avcheck(sent);int r=receive();if(r!=AVERROR(EAGAIN))avcheck(r);}
                    av_packet_unref(packet);}
                if(!stopped){if(error!=AVERROR_EOF)avcheck(error);summary["demux_eof"]=true;
                    {std::lock_guard<std::recursive_mutex> lock(*va_mutex);avcheck(avcodec_send_packet(gen->codec,nullptr));}
                    int r=receive();if(r!=AVERROR_EOF)avcheck(r);summary["decoder_eof"]=true;}
                av_packet_free(&packet);avformat_close_input(&format);files.push_back(summary);
            }
        }catch(const std::exception& e){std::lock_guard<std::mutex> lock(mutex);failure=e.what();}
        std::lock_guard<std::mutex> lock(mutex);done=true;condition.notify_all();
    }
public:
    Engine(const std::string& configuration,const std::string& shader):cfg(Json::parse(configuration)),w(cfg.at("width")),h(cfg.at("height")),pw((w+7)/8*8),ph((h+7)/8*8),wm(cfg.at("watermark").get<std::array<int,4>>()){
        cv::setNumThreads(2);auto render=cfg.value("render_device",std::string("/dev/dri/renderD128"));avcheck(av_hwdevice_ctx_create(&device,AV_HWDEVICE_TYPE_VAAPI,render.c_str(),nullptr,0));
        auto* hw=(AVHWDeviceContext*)device->data;display=((AVVAAPIDeviceContext*)hw->hwctx)->display;
        cl_uint count;clcheck(clGetPlatformIDs(0,nullptr,&count));std::vector<cl_platform_id> platforms(count);clcheck(clGetPlatformIDs(count,platforms.data(),nullptr));
        for(auto p:platforms){auto get=(clGetDeviceIDsFromVA_APIMediaAdapterINTEL_fn)clGetExtensionFunctionAddressForPlatform(p,"clGetDeviceIDsFromVA_APIMediaAdapterINTEL");cl_uint found=0;
            if(get && get(p,CL_VA_API_DISPLAY_INTEL,display,CL_PREFERRED_DEVICES_FOR_VA_API_INTEL,1,&cldevice,&found)==CL_SUCCESS && found){platform=p;break;}}
        if(!platform)throw std::runtime_error("VA/OpenCL sharing unavailable");cl_context_properties props[]={CL_CONTEXT_PLATFORM,(cl_context_properties)platform,CL_CONTEXT_VA_API_DISPLAY_INTEL,(cl_context_properties)display,CL_CONTEXT_INTEROP_USER_SYNC,CL_FALSE,0};
        cl_int e;context=clCreateContext(props,1,&cldevice,nullptr,nullptr,&e);clcheck(e);char name[256];clcheck(clGetPlatformInfo(platform,CL_PLATFORM_NAME,sizeof(name),name,nullptr));
        cv::ocl::attachContext(name,platform,context,cldevice);cv::ocl::setUseOpenCL(true);if(!cv::ocl::useOpenCL())throw std::runtime_error("private OpenCL disabled");
        queue=clCreateCommandQueue(context,cldevice,0,&e);clcheck(e);
        import_va=(clCreateFromVA_APIMediaSurfaceINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform,"clCreateFromVA_APIMediaSurfaceINTEL");
        acquire_va=(clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform,"clEnqueueAcquireVA_APIMediaSurfacesINTEL");
        release_va=(clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform,"clEnqueueReleaseVA_APIMediaSurfacesINTEL");
        if(!import_va || !acquire_va || !release_va)throw std::runtime_error("VA sharing entrypoint missing");
        gray.create(h,w,CV_8U);prepared.create(ph,pw,CV_8U);foreground.create(ph,pw,CV_8U);
        if(!gray.isContinuous() || !prepared.isContinuous())throw std::runtime_error("non-linear private GPU buffers");
        temporary=clCreateBuffer(context,CL_MEM_READ_WRITE,size_t(w)*h*2,nullptr,&e);clcheck(e);auto table=cfg.at("gray_lut").get<std::vector<uint8_t>>();if(table.size()!=256)throw std::runtime_error("invalid gray LUT");
        lut=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,256,table.data(),&e);clcheck(e);
        const char* text=shader.c_str();size_t size=shader.size();program=clCreateProgramWithSource(context,1,&text,&size,&e);clcheck(e);e=clBuildProgram(program,1,&cldevice,"-cl-std=CL1.2",nullptr,nullptr);
        if(e!=CL_SUCCESS){size_t bytes;clGetProgramBuildInfo(program,cldevice,CL_PROGRAM_BUILD_LOG,0,nullptr,&bytes);std::string log(bytes,'\0');clGetProgramBuildInfo(program,cldevice,CL_PROGRAM_BUILD_LOG,bytes,log.data(),nullptr);throw std::runtime_error(log);}
        for(auto key:{"gray_from_y","horizontal9","vertical9_mask"}){kernels[key]=clCreateKernel(program,key,&e);clcheck(e);}
        mog=cv::createBackgroundSubtractorMOG2(150,16,true);for(int i=0;i<8;++i)free_slots.push_back(i);producer=std::thread([this]{produce();});
    }
    ~Engine(){stopped=true;condition.notify_all();if(producer.joinable())producer.join();current.reset();ready.clear();clFinish(queue);mog.release();gray.release();prepared.release();foreground.release();
        for(auto& k:kernels)clReleaseKernel(k.second);clReleaseMemObject(temporary);clReleaseMemObject(lut);clReleaseProgram(program);clReleaseCommandQueue(queue);clReleaseContext(context);av_buffer_unref(&device);}
    bool next(){py::gil_scoped_release release;if(current)throw std::runtime_error("lease not released");std::unique_lock<std::mutex> lock(mutex);condition.wait(lock,[&]{return done || !ready.empty();});
        if(!failure.empty())throw std::runtime_error(failure);if(ready.empty()){lock.unlock();producer.join();return false;}current=ready.front();ready.pop_front();return true;}
    void preprocess(){py::gil_scoped_release release;if(!current)throw std::runtime_error("missing lease");AVFrame* frame=current->frame;std::unique_lock<std::recursive_mutex> lock(*va_mutex);
        surface_ids[current->slot]=(VASurfaceID)(uintptr_t)frame->data[3];if(vaSyncSurface(display,surface_ids[current->slot])!=VA_STATUS_SUCCESS)throw std::runtime_error("source sync failed");
        cl_int e;cl_mem image=import_va(context,CL_MEM_READ_ONLY,&surface_ids[current->slot],0,&e);clcheck(e);++imports;clcheck(acquire_va(queue,1,&image,0,nullptr,nullptr));++acquires;
        auto gray_mem=(cl_mem)gray.handle(cv::ACCESS_WRITE);auto k=kernels.at("gray_from_y");arg(k,0,image);arg(k,1,gray_mem);arg(k,2,lut);arg(k,3,w);arg(k,4,h);arg(k,5,int(frame->crop_left));arg(k,6,int(frame->crop_top));launch("gray_from_y",w,h);
        cl_event event;clcheck(release_va(queue,1,&image,0,nullptr,&event));clcheck(clWaitForEvents(1,&event));clReleaseEvent(event);clcheck(clReleaseMemObject(image));++releases;lock.unlock();
        k=kernels.at("horizontal9");arg(k,0,gray_mem);arg(k,1,temporary);arg(k,2,w);arg(k,3,h);launch("horizontal9",w,h);
        auto target=(cl_mem)prepared.handle(cv::ACCESS_WRITE);k=kernels.at("vertical9_mask");arg(k,0,temporary);arg(k,1,target);arg(k,2,w);arg(k,3,h);arg(k,4,pw);arg(k,5,ph);for(int i=0;i<4;++i)arg(k,6+i,wm[i]);launch("vertical9_mask",pw,ph);clcheck(clFinish(queue));
        ++updates;mog->apply(prepared,foreground,updates==1?1.:.006644493744965563);cv::ocl::finish();}
    void release(){if(!current)return;int slot=current->slot;current.reset();{std::lock_guard<std::mutex> lock(mutex);--resident;}give(slot);}
    std::string metadata(){return current->metadata.dump();}
    uintptr_t context_handle(){return (uintptr_t)context;}uintptr_t queue_handle(){return (uintptr_t)queue;}
    uintptr_t prepared_handle(){return (uintptr_t)prepared.handle(cv::ACCESS_READ);}
    uintptr_t foreground_handle(){return (uintptr_t)foreground.handle(cv::ACCESS_READ);}uint64_t stride(){return foreground.step[0];}uint64_t offset(){return foreground.offset;}
    std::string metrics(){return Json({{"decoded_frames",decoded.load()},{"selected_frames",selected.load()},{"model_updates",updates},{"surface_syncs",syncs.load()},
        {"resident_peak",peak},{"resident_at_end",resident},{"imports",imports},{"acquires",acquires},{"releases",releases},{"host_y_upload_bytes",0},{"files",files}}).dump();}
};
PYBIND11_MODULE(_motion_gpu_front,m){py::class_<Engine>(m,"Engine").def(py::init<const std::string&,const std::string&>()).def("next",&Engine::next).def("preprocess",&Engine::preprocess)
    .def("release",&Engine::release).def("metadata",&Engine::metadata).def("context_handle",&Engine::context_handle).def("queue_handle",&Engine::queue_handle)
    .def("prepared_handle",&Engine::prepared_handle).def("foreground_handle",&Engine::foreground_handle).def("stride",&Engine::stride).def("offset",&Engine::offset).def("metrics",&Engine::metrics);}
