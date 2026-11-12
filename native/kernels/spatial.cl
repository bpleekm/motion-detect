// Packed bits: MOG low, Motion low, independent Motion strength-2.
typedef struct{int area,x0,y0,x1,y1;}Stat;
typedef struct{int x,y,w,h,area;}Region;
__kernel void normalize_masks(__global const uchar* fg,ulong offset,uint stride,__global const uchar* raw,uint pitch,
                             __global uchar* packed,uint w,uint h,int wx,int wy,int ww,int wh,int first){
    uint i=get_global_id(0);if(i>=w*h)return;uint x=i%w,y=i/w;
    int excluded=first || ((int)x>=wx && (int)x<wx+ww && (int)y>=wy && (int)y<wy+wh);
    packed[i]=excluded?0:((fg[offset+y*stride+x]==255?1:0)|(raw[y*pitch+x]?6:0));
}
__kernel void morph(__global const uchar* input,__global uchar* output,uint w,uint h,int cross,int erode){
    __local uchar tile[400];uint lane=get_local_id(1)*16+get_local_id(0);int bx=(int)get_group_id(0)*16-2,by=(int)get_group_id(1)*16-2;
    for(uint t=lane;t<400;t+=256){int x=bx+(int)(t%20),y=by+(int)(t/20);tile[t]=(x>=0 && y>=0 && x<(int)w && y<(int)h)?input[y*w+x]:0;}
    barrier(CLK_LOCAL_MEM_FENCE);uint x=get_global_id(0),y=get_global_id(1);if(x>=w || y>=h)return;
    uint low=erode?3:0,high=erode?4:0;int lx=get_local_id(0)+2,ly=get_local_id(1)+2;
    #pragma unroll
    for(int dy=-2;dy<=2;++dy){
        #pragma unroll
        for(int dx=-2;dx<=2;++dx){uint v=tile[(ly+dy)*20+lx+dx];
            if(abs(dx)<=1 && abs(dy)<=1 && (!cross || abs(dx)+abs(dy)<=1))low=erode?(low&(v&3)):(low|(v&3));
            if(!cross || abs(dx)+abs(dy)<=2)high=erode?(high&(v&4)):(high|(v&4));}}
    output[y*w+x]=(uchar)(low|high);
}
inline uint local_root(volatile __local uint* p,uint a){for(;;){uint b=atomic_add(p+a-1,0u);if(a==b)return a;a=b;}}
inline void local_join(volatile __local uint* p,uint a,uint b){for(;;){a=local_root(p,a);b=local_root(p,b);if(a==b)return;
    uint hi=max(a,b),lo=min(a,b),old=atomic_cmpxchg(p+hi-1,hi,lo);if(old==hi)return;}}
inline uint global_root(volatile __global uint* p,uint a){for(;;){uint b=atomic_add(p+a,0u);if(a==b)return a;a=b;}}
inline void global_join(volatile __global uint* p,uint a,uint b){for(;;){a=global_root(p,a);b=global_root(p,b);if(a==b)return;
    uint hi=max(a,b),lo=min(a,b),old=atomic_cmpxchg(p+hi,hi,lo);if(old==hi)return;}}
__kernel void tile_labels(__global const uchar* mask,__global uint* parents,__global ushort* local_components,uint w,uint h){
    __local uint p[256];uint x=get_global_id(0),y=get_global_id(1),c=get_group_id(2),lx=get_local_id(0),ly=get_local_id(1),lane=ly*16+lx,n=w*h;
    int active=x<w && y<h && (mask[y*w+x]&(1u<<c));p[lane]=active?lane+1:0;barrier(CLK_LOCAL_MEM_FENCE);
    if(active){if(lx && atomic_add(p+lane-1,0u))local_join(p,lane+1,lane);if(ly && atomic_add(p+lane-16,0u))local_join(p,lane+1,lane-15);}
    barrier(CLK_LOCAL_MEM_FENCE);if(x<w && y<h){uint root=active?local_root(p,lane+1):0;local_components[c*n+y*w+x]=(ushort)root;
        uint id=root?(y-ly+(root-1)/16)*w+x-lx+(root-1)%16+1:0;parents[c*(n+1)+y*w+x+1]=id;if(x==0 && y==0)parents[c*(n+1)]=0;}
}
__kernel void join_tiles(__global uint* parents,uint w,uint h){
    uint x=get_global_id(0),y=get_global_id(1),c=get_group_id(2),n=w*h;if(x>=w || y>=h)return;volatile __global uint* p=parents+c*(n+1);uint id=y*w+x+1;
    if(!p[id])return;if(x && x%16==0 && p[id-1])global_join(p,id,id-1);if(y && y%16==0 && p[id-w])global_join(p,id,id-w);
}
__kernel void flatten_init(__global uint* parents,__global Stat* stats,uint w,uint h){
    uint i=get_global_id(0),c=get_global_id(1),n=w*h;if(i>=n)return;volatile __global uint* p=parents+c*(n+1);if(!p[i+1])return;
    uint r=global_root(p,i+1);atomic_xchg(p+i+1,r);if(r==i+1)stats[c*(n+1)+r]=(Stat){0,(int)w,(int)h,-1,-1};
}
__kernel void tile_stats(__global const uint* parents,__global const ushort* local_components,__global Stat* stats,uint w,uint h){
    __local int area[256],x0[256],y0[256],x1[256],y1[256];uint x=get_global_id(0),y=get_global_id(1),c=get_group_id(2),n=w*h,lane=get_local_id(1)*16+get_local_id(0);
    area[lane]=0;x0[lane]=w;y0[lane]=h;x1[lane]=-1;y1[lane]=-1;barrier(CLK_LOCAL_MEM_FENCE);
    if(x<w && y<h){uint a=local_components[c*n+y*w+x];if(a){--a;atomic_inc(area+a);atomic_min(x0+a,(int)x);atomic_min(y0+a,(int)y);atomic_max(x1+a,(int)x);atomic_max(y1+a,(int)y);}}
    barrier(CLK_LOCAL_MEM_FENCE);if(area[lane]){uint r=parents[c*(n+1)+y*w+x+1];__global Stat* s=stats+c*(n+1)+r;
        atomic_add(&s->area,area[lane]);atomic_min(&s->x0,x0[lane]);atomic_min(&s->y0,y0[lane]);atomic_max(&s->x1,x1[lane]);atomic_max(&s->y1,y1[lane]);}
}
inline void emit(__global uint* counts,__global Region* regions,uint c,uint capacity,Stat s){uint k=atomic_inc(counts+c);regions[c*capacity+k]=(Region){s.x0,s.y0,s.x1-s.x0+1,s.y1-s.y0+1,s.area};}
__kernel void collect_low(__global const uint* parents,__global const Stat* stats,__global const uchar* morph_mask,__global uchar* kept,
                          __global uchar* strict_mask,__global uint* counts,__global Region* regions,uint w,uint h,int minimum,uint capacity){
    uint i=get_global_id(0),n=w*h;if(i>=n)return;uint bits=0;
    for(uint c=0;c<2;++c){uint r=parents[c*(n+1)+i+1];if(r){Stat s=stats[c*(n+1)+r];if(s.area>=minimum){bits|=1u<<c;if(r==i+1)emit(counts,regions,c,capacity,s);}}}
    kept[i]=(uchar)bits;strict_mask[i]=((bits&2) && (morph_mask[i]&4))?1:0;
}
__kernel void collect_single(__global const uint* parents,__global const Stat* stats,__global uint* counts,__global Region* regions,uint n,int minimum,uint capacity){
    uint i=get_global_id(0);if(i>=n)return;uint r=parents[i+1];if(r){Stat s=stats[r];if(s.area>minimum && r==i+1)emit(counts,regions,2,capacity,s);}
}
