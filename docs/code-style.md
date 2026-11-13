# 代码风格

## Python

- Python 3.11语法基线，4空格缩进、UTF-8/LF，Ruff统一100列格式和导入顺序。
- 模块/函数/变量用snake_case，类型用PascalCase，常量用UPPER_SNAKE_CASE。
- 命名表达职责：source_manifest、output_dir、frame_metadata、gpu_metrics。图像局部坐标x/y及明确范围内的数学索引可用短名。
- CLI入口只解析与调度；媒体处理、构建、进程通信分别归对应模块。源路径只在运行参数中传入。
- 对外入口补简短职责说明，异常/hold保留明确语义；不因格式整理改变检测数学或输出协议。

```sh
ruff check --no-cache src tests tools
ruff format --check --no-cache src tests tools
```

工具版本固定在dev依赖。临时/cache目录仍由外部工作根提供，格式整理后运行已有单元和集成检查。

## C++ / OpenCL

- 使用仓库.clang-format：LLVM基础、4空格、100列，多条操作拆行，控制流显式分块。
- C++类型用PascalCase，函数/变量用snake_case。跨像素域状态使用reference、residual、padded_width等领域名。
- OpenCL中的x/y、w/h等坐标局部变量允许保留；buffer、component和统计参数应表达角色。
- 保持FP_CONTRACT OFF、整数宽度、运算次序、内存布局和ABI静态断言。格式化不重排算式，也不默认启用fast-math。
- Python原生绑定名属于兼容接口，与实现类型名可以不同；改变实现命名时同时核对构建入口和worker。

```sh
clang-format --dry-run --Werror native/*.cpp native/kernels/*.cl
```

原生变更需重新构建到外部目录，再运行GPU主流程烟测；不能用格式化前的二进制作验证。
