# motion-detect

独立的帧级运动证据生产器，与 [motion-evidence](https://github.com/bpleekm/motion-evidence) 通过版本化文件协议协作。

输入原片，输出真实PTS帧索引、MOG/Motion空间证据和完整解码状态。检测结果是测量，不直接代表“值得原速保留”。

交付阶段：CPU reference、独立原生GPU和按需GPU人检主流程均已接入。合成/执行位置验证已完成，真实素材与新口径语义质量分开验收。

## 主流程

```text
原片/源清单 → 顺序解码与帧身份 → 像素模型 → 空间复核 → detection-v1 → AI证据项目
```

```sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DETECT_REPO/src"
"$PYTHON" -B -m motion_detect.cli run \
  --input "$INPUT_VIDEO" --out "$RUN_ROOT/detect"
```

`RUN_ROOT`应位于工作区外部的私有临时根，输出目录必须新建。缺少输入或时长等必需数据时生成`status: hold`；补齐后用新的运行目录重跑。

默认`cpu-reference`是明确标记的CPU double反馈对照。原生GPU入口通过`--backend native-gpu --native-build "$NATIVE_BUILD"`使用；两者精度/性能分别记录，不静默回退。原生VA/OpenCL主线见[核心方案](docs/core-design.md)和[路线](docs/roadmap.md)。

`--person-model "$PERSON_MODEL"`启用原触发条件下的GPU FP32人检；无模型先hold特征，静止人物仍不自动保留。输入BGR按需传输，预处理/NMS在CPU；网络前向的GPU执行和层级位置被验证。

## 文档

- [产品](docs/product.md) · [架构](docs/architecture.md) · [核心方案](docs/core-design.md)
- [协议](docs/contracts.md) · [验证](docs/validation.md) · [运行](docs/operations.md)
- [路线](docs/roadmap.md) · [决策记录](docs/adr/0001-boundaries.md)

真实视频、截图、源清单、机器路径和运行报告不属于Git提交物。源码示例和测试均使用合成数据。
