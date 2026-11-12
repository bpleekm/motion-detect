# detection-v1文件协议

协议标识`motion-detect/v1`。JSON为UTF-8；逐帧文件为JSONL。

| 文件 | 内容 |
|---|---|
| source.json | case_id、total_ms；files含file_id/full SHA/path/stream_index/time_base/start_us/source_start_pts/width/height |
| frames.jsonl | 全部解码帧：frame_id、file_id、stream_index、pts、time_base、time_ms、width、height、key_frame |
| observations.jsonl | 选帧：frame_id/time_ms/seq，mog_pass/motion_pass/has_candidate/warming，独立区域与可选model信息 |
| status.json | complete或hold；完整计数、文件EOF、后端、产物SHA |

帧ID采用`file_id:s<stream>:pts<pts>`，file_id来自源SHA。真实SHA/路径只存在私有运行目录。时间由源偏移和PTS有理数计算，round-half-even到ms；原PTS/time_base始终保留。

消费者必须验证complete、文件指纹、帧ID与时间顺序。`mog_pass`是MOG低档通过，`motion_pass`是严格Motion通过。布尔值不是语义正确性的置信分。

[`schemas/detection-v1.json`](../schemas/detection-v1.json)定义观测格式。新增可选字段可保持v1兼容；改变时间、身份或字段语义必须升级协议。motion-evidence使用合成fixture检验兼容性。
