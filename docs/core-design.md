# 核心方案

## 已确认的数学语义

- 分析选帧按真实PTS至少间隔200ms，跨源文件不重置单份模型。
- 在可见内容上先Gaussian9/reflect101，再清配置的水印，再补8对齐算法零边；硬件pitch/coded padding不属于内容。
- MOG2：history150、varThreshold16、shadows开启、首次学习率1，后续`.006644493744965563`；只取255为前景。
- Motion先计算int16有符号残差，再更新整数reference和age。噪声初值32，clip12–48；active严格`>`，low `<=max(1,int(noise*.2))`，平均向下取整，age饱和255、50次吸收。
- 每5次更新按精确quiet统计更新noise；offset为内容区残差每16像素采样的中位数。补偿后以float32产生raw。
- 低档从raw独立去噪1；严格Motion从raw独立去噪2，再与Motion低档keep相交。矩形/十字腐蚀→十字/矩形膨胀，独立零边界。
- 4连通，面积按768×432缩放、round-half-up；低档`>=50ref`、严格高档`>100ref`。区域按面积降序、box升序排序。

## 后端状态

`ReferenceCore`是独立CPU double反馈对照，禁用OpenCL并固定OpenCV2线程。其CPU MOG不被宣称为GPU MOG逐像素等价，也不替代GPU FP32闭环验证。

GPU主线：VAAPI完整解码→每帧VA acquire→GPU gray→release/销毁CL导入→GPU prepared/MOG/FP32 Motion→三bit独立形态学→tile内/跨tile精确4conn→area/box/keep。CPU只读紧凑区域，按需取原图。

VA/CL共享期间AVFrame租约必须存活；所有解码输出含未选帧回收前完成同步。8应用租约与DPB独立，模型有序，源文件切换不改变学习顺序。

迁移只读来源时，仅提取通用源码和参数语义。私有场景配置、源清单、机器路径及旧业务保留标签不进入仓库。原生后端完成独立构建和合成验证前，状态必须明确保留为待验收。

## 按需GPU人物特征

触发条件保持为`!warming && has_candidate && !(mog_pass || motion_pass)`。模型前处理保持BGR→RGB、居中416画布、INTER_AREA缩放、原float32归一化；推理指定DNN_BACKEND_OPENCV/DNN_TARGET_OPENCL（FP32），score/NMS及输出映射显式配置。无模型先hold人物特征，主证据仍可输出；不把“尚未检测”记成“无人”。

CPU预处理/后处理和必要人物图像传输如实计量。需核对网络实际GPU执行及CPU层回退；不可用或不能满足GPU前向要求时记录hold，禁止静默选择CPU网络。
