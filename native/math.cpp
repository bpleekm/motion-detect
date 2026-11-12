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

namespace py=pybind11;
using Json=nlohmann::json;
static void check(cl_int e){if(e!=CL_SUCCESS)throw std::runtime_error("OpenCL error "+std::to_string(e));}
struct Params{cl_uint active_t,low_t;cl_float raw_t,noise;};
struct Meta{cl_uint active_t,low_t;cl_float raw_t,before,after,offset;cl_ulong active,quiet_sum;cl_uint applied,seq;};
struct Region{cl_int x,y,w,h,area;};
static_assert(sizeof(Params)==16 && sizeof(Meta)==48 && offsetof(Meta,active)==24 && sizeof(Region)==20);

class Math {
    cl_context context;cl_command_queue queue;std::vector<cl_program> programs;
    std::map<std::string,cl_kernel> kernels;std::vector<cl_mem> memory;
    cl_uint w,h,pw,ph,n,padded,groups,capacity,updates=0,region_capacity;int low,high;std::array<int,4> wm;
    cl_mem ref,age,residual,raw,partial,hist,params,trace,normalized,a,b,kept,single,parents,local_components,stats,counts,regions;
    uint64_t allocated=0,compact_read=0,trace_read=0;
    cl_mem alloc(size_t size){cl_int e;auto p=clCreateBuffer(context,CL_MEM_READ_WRITE,size,nullptr,&e);check(e);
        cl_uchar zero=0;check(clEnqueueFillBuffer(queue,p,&zero,1,0,size,0,nullptr,nullptr));memory.push_back(p);allocated+=size;return p;}
    template<class T>void arg(cl_kernel k,int i,T value){check(clSetKernelArg(k,i,sizeof(value),&value));}
    void linear(const std::string& name,size_t size,int channels=1,size_t local_size=256){
        size_t global[2]={(size+local_size-1)/local_size*local_size,size_t(channels)},local[2]={local_size,1};
        check(clEnqueueNDRangeKernel(queue,kernels.at(name),2,nullptr,global,local,0,nullptr,nullptr));}
    void tile(const std::string& name,int channels=1){size_t global[3]={(w+15)/16*16ul,(h+15)/16*16ul,size_t(channels)},local[3]={16,16,1};
        check(clEnqueueNDRangeKernel(queue,kernels.at(name),3,nullptr,global,local,0,nullptr,nullptr));}
    void ccl(cl_mem input,int channels){
        auto k=kernels.at("tile_labels");arg(k,0,input);arg(k,1,parents);arg(k,2,local_components);arg(k,3,w);arg(k,4,h);tile("tile_labels",channels);
        k=kernels.at("join_tiles");arg(k,0,parents);arg(k,1,w);arg(k,2,h);tile("join_tiles",channels);
        k=kernels.at("flatten_init");arg(k,0,parents);arg(k,1,stats);arg(k,2,w);arg(k,3,h);linear("flatten_init",n,channels);
        k=kernels.at("tile_stats");arg(k,0,parents);arg(k,1,local_components);arg(k,2,stats);arg(k,3,w);arg(k,4,h);tile("tile_stats",channels);
    }
    void program(const std::string& text,const std::vector<std::string>& names,cl_device_id device){
        const char* s=text.c_str();size_t length=text.size();cl_int e;auto p=clCreateProgramWithSource(context,1,&s,&length,&e);check(e);
        e=clBuildProgram(p,1,&device,"-cl-std=CL1.2",nullptr,nullptr);
        if(e!=CL_SUCCESS){size_t bytes;clGetProgramBuildInfo(p,device,CL_PROGRAM_BUILD_LOG,0,nullptr,&bytes);std::string log(bytes,'\0');clGetProgramBuildInfo(p,device,CL_PROGRAM_BUILD_LOG,bytes,log.data(),nullptr);throw std::runtime_error(log);}
        programs.push_back(p);for(auto& name:names){kernels[name]=clCreateKernel(p,name.c_str(),&e);check(e);}
    }
public:
    Math(uintptr_t c,uintptr_t parent,int width,int height,int frames,std::array<int,4> watermark,const std::string& motion_source,const std::string& spatial_source):
        context((cl_context)c),w(width),h(height),pw((width+7)/8*8),ph((height+7)/8*8),n(w*h),padded(pw*ph),capacity(frames),wm(watermark){
        check(clRetainContext(context));cl_device_id device;check(clGetCommandQueueInfo((cl_command_queue)parent,CL_QUEUE_DEVICE,sizeof(device),&device,nullptr));
        cl_int e;queue=clCreateCommandQueue(context,device,0,&e);check(e);
        low=std::max(1,int(std::floor(double(w)*h/(768*432)*50+.5)));high=std::max(1,int(std::floor(double(w)*h/(768*432)*100+.5)));region_capacity=n/low+1;
        program(motion_source,{"begin_frame","learn","reduce_noise","histogram","median_offset","raw_mask"},device);
        program(spatial_source,{"normalize_masks","morph","tile_labels","join_tiles","flatten_init","tile_stats","collect_low","collect_single"},device);
        groups=(padded+4095)/4096;ref=alloc(padded);age=alloc(padded);residual=alloc(size_t(padded)*2);raw=alloc(padded);
        partial=alloc(size_t(groups)*8);hist=alloc(512*4);params=alloc(sizeof(Params));trace=alloc(size_t(capacity)*sizeof(Meta));
        Params initial={32,6,32,32};check(clEnqueueWriteBuffer(queue,params,CL_TRUE,0,sizeof(initial),&initial,0,nullptr,nullptr));
        normalized=alloc(n);a=alloc(n);b=alloc(n);kept=alloc(n);single=alloc(n);parents=alloc(size_t(n+1)*2*4);
        local_components=alloc(size_t(n)*2*2);stats=alloc(size_t(n+1)*2*20);counts=alloc(12);regions=alloc(size_t(region_capacity)*3*20);
    }
    ~Math(){clFinish(queue);for(auto p:memory)clReleaseMemObject(p);for(auto& k:kernels)clReleaseKernel(k.second);for(auto p:programs)clReleaseProgram(p);clReleaseCommandQueue(queue);clReleaseContext(context);}
    std::string process(uintptr_t input,uintptr_t foreground,uint64_t stride,uint64_t offset){
        py::gil_scoped_release release;if(updates>=capacity)throw std::runtime_error("model trace capacity exceeded");cl_uint index=updates++;
        auto k=kernels.at("begin_frame");arg(k,0,params);arg(k,1,trace);arg(k,2,index);linear("begin_frame",1,1,1);
        k=kernels.at("learn");arg(k,0,(cl_mem)input);arg(k,1,ref);arg(k,2,age);arg(k,3,residual);arg(k,4,partial);arg(k,5,trace);arg(k,6,index);arg(k,7,padded);linear("learn",size_t(groups)*256);
        if(updates%5==0){k=kernels.at("reduce_noise");arg(k,0,partial);arg(k,1,groups);arg(k,2,params);arg(k,3,trace);arg(k,4,index);arg(k,5,padded);arg(k,6,cl_uint(high));linear("reduce_noise",1,1,1);}
        if(index){cl_uint zero=0;check(clEnqueueFillBuffer(queue,hist,&zero,4,0,512*4,0,nullptr,nullptr));
            k=kernels.at("histogram");arg(k,0,residual);arg(k,1,hist);arg(k,2,w);arg(k,3,h);arg(k,4,pw);cl_uint samples=((w+15)/16)*((h+15)/16);linear("histogram",samples);
            k=kernels.at("median_offset");arg(k,0,hist);arg(k,1,trace);arg(k,2,index);arg(k,3,samples);linear("median_offset",1,1,1);
            k=kernels.at("raw_mask");arg(k,0,residual);arg(k,1,raw);arg(k,2,trace);arg(k,3,index);arg(k,4,w);arg(k,5,h);arg(k,6,pw);arg(k,7,padded);
            for(int i=0;i<4;++i)arg(k,8+i,wm[i]);linear("raw_mask",size_t(groups)*256);}
        cl_uint zero=0;check(clEnqueueFillBuffer(queue,counts,&zero,4,0,12,0,nullptr,nullptr));
        k=kernels.at("normalize_masks");arg(k,0,(cl_mem)foreground);arg(k,1,cl_ulong(offset));arg(k,2,cl_uint(stride));arg(k,3,raw);arg(k,4,pw);arg(k,5,normalized);arg(k,6,w);arg(k,7,h);
        for(int i=0;i<4;++i)arg(k,8+i,wm[i]);arg(k,12,int(index==0));linear("normalize_masks",n);
        cl_mem from=normalized,to=a;for(int pass=0;pass<4;++pass){k=kernels.at("morph");arg(k,0,from);arg(k,1,to);arg(k,2,w);arg(k,3,h);arg(k,4,int(pass==1 || pass==2));arg(k,5,int(pass<2));tile("morph");from=to;to=to==a?b:a;}
        ccl(b,2);k=kernels.at("collect_low");arg(k,0,parents);arg(k,1,stats);arg(k,2,b);arg(k,3,kept);arg(k,4,single);arg(k,5,counts);arg(k,6,regions);arg(k,7,w);arg(k,8,h);arg(k,9,low);arg(k,10,region_capacity);linear("collect_low",n);
        ccl(single,1);k=kernels.at("collect_single");arg(k,0,parents);arg(k,1,stats);arg(k,2,counts);arg(k,3,regions);arg(k,4,n);arg(k,5,high);arg(k,6,region_capacity);linear("collect_single",n);
        std::array<cl_uint,3> sizes;check(clEnqueueReadBuffer(queue,counts,CL_TRUE,0,12,sizes.data(),0,nullptr,nullptr));compact_read+=12;
        std::array<Json,3> lists;for(int channel=0;channel<3;++channel){if(sizes[channel]>region_capacity)throw std::runtime_error("region capacity invariant violated");
            std::vector<Region> values(sizes[channel]);size_t bytes=values.size()*sizeof(Region);
            if(bytes){check(clEnqueueReadBuffer(queue,regions,CL_TRUE,size_t(channel)*region_capacity*sizeof(Region),bytes,values.data(),0,nullptr,nullptr));compact_read+=bytes;}
            std::sort(values.begin(),values.end(),[](auto& x,auto& y){return x.area!=y.area?x.area>y.area:std::array<int,4>{x.x,x.y,x.w,x.h}<std::array<int,4>{y.x,y.y,y.w,y.h};});
            lists[channel]=Json::array();for(auto& r:values)lists[channel].push_back({{"box",{r.x,r.y,r.w,r.h}},{"area",r.area}});}
        return Json({{"mog_pass",sizes[0]>0},{"motion_pass",sizes[2]>0},{"has_candidate",sizes[0]>0 || sizes[1]>0},{"warming",updates<=5},
            {"regions",{{"mog",lists[0]},{"motion_low",lists[1]},{"motion_high",lists[2]}}}}).dump();
    }
    std::string model_trace(){std::vector<Meta> data(updates);check(clEnqueueReadBuffer(queue,trace,CL_TRUE,0,data.size()*sizeof(Meta),data.data(),0,nullptr,nullptr));trace_read+=data.size()*sizeof(Meta);
        Json out=Json::array();for(auto& m:data)out.push_back({{"seq",m.seq},{"active_threshold",m.active_t},{"low_threshold",m.low_t},{"raw_threshold",m.raw_t},
            {"noise_used",m.before},{"noise_after",m.after},{"offset",m.seq==1?Json(nullptr):Json(m.offset)},{"active_count",m.active},{"quiet_sum",m.quiet_sum},{"update_applied",bool(m.applied)}});return out.dump();}
    std::string metrics(){return Json({{"updates",updates},{"precision","FP32 noise; integer reference/age/residual"},{"allocated_bytes",allocated},
        {"compact_read_bytes",compact_read},{"trace_read_bytes",trace_read},{"raw_read_bytes",0},{"foreground_read_bytes",0}}).dump();}
};
PYBIND11_MODULE(_motion_gpu_math,m){py::class_<Math>(m,"Math").def(py::init<uintptr_t,uintptr_t,int,int,int,std::array<int,4>,const std::string&,const std::string&>())
    .def("process",&Math::process).def("model_trace",&Math::model_trace).def("metrics",&Math::metrics);}
