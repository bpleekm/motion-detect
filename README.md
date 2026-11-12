# motion-detect

独立的帧级运动证据生产器，与 [motion-evidence](https://github.com/bpleekm/motion-evidence) 通过版本化文件协议协作。

输入原片，输出真实PTS帧索引、MOG/Motion空间证据和完整解码状态。检测结果是测量，不直接代表“值得原速保留”。

交付阶段：CPU reference与完整索引主流程已通过合成验证；原生GPU与按需GPU人检分批接入。

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

当前可运行入口是明确标记的`cpu-reference`：迁移MOG/Motion/形态学/4conn语义，用于跑通协议和流程。它不冒充原生GPU方案的精度或性能。原生VA/OpenCL主线整理见[核心方案](docs/core-design.md)和[路线](docs/roadmap.md)。

## 文档

- [产品](docs/product.md) · [架构](docs/architecture.md) · [核心方案](docs/core-design.md)
- [协议](docs/contracts.md) · [验证](docs/validation.md) · [运行](docs/operations.md)
- [路线](docs/roadmap.md) · [决策记录](docs/adr/0001-boundaries.md)

真实视频、截图、源清单、机器路径和运行报告不属于Git提交物。源码示例和测试均使用合成数据。
