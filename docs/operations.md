# 运行与维护

## 环境

Python>=3.11，依赖见pyproject。建议在外部工作根建立venv；设置`TMPDIR`、`XDG_CACHE_HOME`、`PIP_CACHE_DIR`和`PYTHONDONTWRITEBYTECODE=1`。测试指定外部basetemp/cache，GPU内核缓存也定向外部根。

当前CLI可通过`PYTHONPATH="$DETECT_REPO/src"`及`python -B -m motion_detect.cli`使用。输入单个视频或显式source manifest，所有真实路径在本地传入。输出目录必须不存在，避免覆盖旧记录。

```sh
"$PYTHON" -B -m motion_detect.cli run --input "$INPUT_VIDEO" --out "$RUN_ROOT/detect"
```

`status: hold`表示缺少数据。先集中登记，补齐时统一创建新运行。指纹变化或非法身份应修复来源，不以放宽校验绕过。

## 原生GPU

需要系统FFmpeg/VA/OpenCL开发库，以及单独的OpenCL-enabled OpenCV4.12 core/video SDK。只读SDK前缀显式传入，不替换Python wheel。构建始终写外部目录：

```sh
"$PYTHON" -B -m motion_detect.cli build-native --out "$NATIVE_BUILD" --opencv-prefix "$PRIVATE_OPENCV_PREFIX"
"$PYTHON" -B -m motion_detect.cli run --input "$INPUT_VIDEO" --out "$RUN_ROOT/gpu" --backend native-gpu --native-build "$NATIVE_BUILD"
```

原生进程不导入Python cv2/av；MOG和Motion/空间共享VA context，输入Y/前景/raw热路径无CPU整图往返。所有构建和内核缓存由外部工作根管理；缺少SDK、模块或设备时状态hold，不选择CPU代跑。

## 按需GPU人检

`run --person-model "$PERSON_MODEL" --person-score 0.25`配置NanoDet权重。只有原候选触发条件满足才启动独立OpenCL DNN进程、按需获取BGR。前处理和NMS在CPU，网络前向OpenCL FP32；逐调用观测本进程GPU render增量并检查网络图没有CPU层回退。

无模型或设备/执行位置不满足时，`person.status=hold`、`detections=null`，不会写成无人；像素主证据可继续完整输出，`status.person`汇总待补特征。模型指纹、原图下载/传输量及特征状态只进入私有运行目录。

## 提交

真实视频/截图/清单/日志/权重/编译产物不提交。提交前检查status、diff、历史、暂存文件及内容；文档与实现同步，用指定作者/提交者身份commit并push。不要修改全局Git设置。

## 原方案来源

旧工作区只读。禁止在那里运行会写产物的脚本；迁移副本和构建前缀均属于新工作区。私有迁移映射仅写外部工作根，公开文档不包含用户主机或数据标识。
