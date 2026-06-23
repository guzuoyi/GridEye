# 开发任务拆解

> 对应 `design-doc.md`（最终决策版），按模块/依赖关系分 7 个 Phase。每个子任务标注验证方式。

---

## Phase 1：基础框架

**目标**：搭好项目骨架，视频抽帧跑通，配置系统就位

### 1.1 项目初始化
- [ ] 1.1.1 创建 Python 项目结构（`src/`, `config/`, `tests/`, `data/`, `archive/`, `notebooks/`）
- [ ] 1.1.2 编写 `requirements.txt`（OpenCV, PyTorch, numpy, openai, ultralytics）
- [ ] 1.1.3 配置 logging，统一日志格式（时间戳 + 模块名 + 级别）

### 1.2 配置系统
- [ ] 1.2.1 创建 `config/default.yaml`，包含所有可配置参数：
  - 视频源路径、抽帧 FPS
  - 分辨率（原始 1920×1080）、YOLO 输入尺寸（640）
  - 尺寸过滤阈值表（person/car/truck/two-wheeler 的高度+面积）
  - 防闪烁参数（窗口 10 帧，阈值 7，尺寸阈值系数 0.8）
  - 位置匹配距离阈值（40px）
  - 状态机参数（驻留 5 帧→WARNING，消失 3s→CLEARED，刷新间隔 30s，MANUAL_REQUIRED 超时 300s）
  - 查找表维护时间
  - LM Studio 配置：`base_url`（默认 `http://localhost:1234/v1`）、`api_key`（本地可为占位值 `lm-studio`）
  - 端到端延迟预算（≤ 5s）
- [ ] 1.2.2 实现 `ConfigLoader` 类：YAML 加载 + 环境变量覆盖 + 参数校验
- [ ] 1.2.3 单元测试：配置加载、缺失字段报错、类型校验

### 1.3 视频抽帧
- [ ] 1.3.1 实现 `VideoSampler` 类：
  - 输入：视频文件路径 / RTSP 流 / 摄像头 index
  - 输出：按配置 FPS（1~2）抽取的帧，格式 `(frame_idx, timestamp, np.ndarray)`
  - 帧序号连续，记录实际抽帧时间戳
- [ ] 1.3.2 单元测试：用一段 30s 测试视频验证抽帧数 ≈ FPS×时长
- [ ] 1.3.3 可视化：抽帧结果写入临时目录，人工确认画面正常

### 1.4 坐标约定
- [ ] 1.4.1 在代码中统一坐标表示：`(x, y) = (col, row)`，原点左上角
- [ ] 1.4.2 实现 `rescale_bbox_stretch(bbox_640) → bbox_orig` 工具函数
  - 等比缩放（Stretch）：`x_orig = x_640 × 3`, `y_orig = y_640 × 1.6875`
- [ ] 1.4.3 单元测试：已知输入输出验证映射正确

**Phase 1 完成标准**：抽帧程序可运行，配置文件可加载，坐标映射函数有测试

---

## Phase 2：YOLOv11 目标检测

**目标**：YOLOv11n 跑通推理，尺寸过滤生效，防闪烁逻辑正确

### 2.1 YOLOv11n 模型加载与推理
- [ ] 2.1.1 下载 YOLOv11n 权重（ultralytics 官方源），放到 `models/` 目录
- [ ] 2.1.2 实现 `YOLODetector` 类：
  - `detect(frame) → List[Detection]`
  - `Detection` 包含：`bbox_orig`, `class`, `confidence`, `center_orig`
  - 内部处理：原图 → resize 640×640（Stretch）→ 推理 → 坐标映射回 1920×1080
- [ ] 2.1.3 单元测试：用一张已知内容的测试图片，验证检测框数量和类别

### 2.2 尺寸过滤
- [ ] 2.2.1 实现 `SizeFilter` 类：
  - 对每个 Detection 计算 `h = y2-y1`, `area = w×h`
  - 对照配置中的阈值表判定（person: 35px/1000px², car: 22/1500, truck: 28/2000, two-wheeler: 22/1000）
  - 低于阈值的直接丢弃，返回 `filtered: List[Detection]`
- [ ] 2.2.2 单元测试：构造已知 bbox 的 Detection（低于/高于阈值），验证过滤正确
- [ ] 2.2.3 可视化：在原图上画出「通过」和「被过滤」的框（不同颜色），人工抽检

### 2.3 防闪烁
- [ ] 2.3.1 实现 `FlickerSuppressor` 类：
  - 维护每个目标最近 N=10 帧的尺寸有效性布尔队列
  - 新增检测帧时 push + pop
  - `is_stable(pseudo_id) → bool`（最近 10 帧 ≥ 7 帧有效）
  - **尺寸有效判定**：使用 §3.1.2 阈值的 **0.8×**（放宽系数）
- [ ] 2.3.2 与位置缓存联动：防闪烁判定为 unstable → 该目标驻留帧数重置为 0
- [ ] 2.3.3 单元测试：构造稳定性波动序列，验证判定阈值

**Phase 2 完成标准**：YOLO 可逐帧检测，小目标和抖动目标被正确过滤，可视化确认

---

## Phase 3：车道线检测与查找表

**目标**：首日标定流程走通，查找表生成并验证，日常维护脚本就位

### 3.1 相机内参标定（棋盘格）
- [ ] 3.1.1 准备棋盘格图像采集脚本：提示用户移动标定板到不同位置，自动抓取 ≥ 15 张
- [ ] 3.1.2 实现 `CameraCalibrator` 类：
  - 调用 `cv2.findChessboardCorners` + `cv2.calibrateCamera`
  - 输出 `K (3×3)` 和 `dist_coeffs`
  - 保存到 `config/camera_params.yaml`
- [ ] 3.1.3 验证：输出重投影误差，要求 < 0.5 px；可视化去畸变效果

### 3.2 IPM 矩阵与外参标定（车道线几何约束）
- [ ] 3.2.1 实现 `IPMCalibrator` 类：
  - 输入：内参 K、消失点（自动检测 or 手动标注）、车道宽度 3.75m
  - 利用消失点 + 车道线端点 + 车道宽度约束，求解俯仰角/偏航角/离地高度
  - 计算 `cv2.getPerspectiveTransform` 或基于外参的 IPM 矩阵 H
- [ ] 3.2.2 验证：在鸟瞰图中车道线近似平行，车道宽度像素比 ≈ 实际比例
- [ ] 3.2.3 保存 H 到 `config/ipm_params.yaml`

### 3.3 查找表（LUT）生成
- [ ] 3.3.1 实现 `LUTGenerator` 类：
  - 输入：H 矩阵 + 车道线方程
  - 车道号约定：硬路肩=0，从图像左侧向右递增（1,2,3...）
  - 为每个像素 (u,v) 计算：反投影到地面平面 → 求车道号 → 求纵向距离
  - 输出：`lut.npy`（1080×1920, dtype=`[('lane','i4'),('dist','f4')]`）
- [ ] 3.3.2 实现 `LUTLookup` 函数：`lookup(u, v) → (lane, dist)`，O(1)
- [ ] 3.3.3 单元测试：构造简单 H 矩阵，验证查表结果；在真实图上选若干点人工估算验证

### 3.4 车道线自动检测（日常维护用）
- [ ] 3.4.1 实现 `LaneDetector` 类：
  - IPM 变换 → 灰度 → 高斯模糊 → Canny → HoughLinesP
  - 聚类拟合：合并相近线段，RANSAC 拟合多项式车道线
- [ ] 3.4.2 可视化：在 IPM 鸟瞰图上叠加检测到的车道线
- [ ] 3.4.3 集成到 `DailyMaintainer`：抓图 → 检测 → 生成新 LUT → 原子替换

### 3.5 每日自动维护调度
- [ ] 3.5.1 实现 `LUTScheduler`：基于配置的每日时间（如 03:00）触发维护
- [ ] 3.5.2 实现原子替换：**Immutable Snapshot**——生成完整新 LUT 后，`active_lut = new_lut` 引用赋值
- [ ] 3.5.3 回退机制：新 LUT 验证失败 → 保留旧 LUT，记录告警日志

**Phase 3 完成标准**：完整标定流程跑通，LUT 生成并保存，查表 O(1) 正确返回车道号+距离

---

## Phase 4：位置缓存与状态机

**目标**：实现无 ID 依赖的位置匹配，状态机 5 个状态 9 条迁移规则全部正确

### 4.1 位置缓存数据结构
- [ ] 4.1.1 实现 `PositionRecord` dataclass（字段见设计文档 §3.3.2，含 MANUAL_REQUIRED 状态）
- [ ] 4.1.2 实现 `PositionCache` 类：
  - `match_or_create(detections) → List[PositionRecord]`
  - `get_active() → List[PositionRecord]`
  - `cleanup_expired(current_time)`
- [ ] 4.1.3 单元测试：构造检测序列，验证 match（同位置同类别）和 create（新位置/new class）

### 4.2 位置匹配
- [ ] 4.2.1 实现匹配函数：类别相同 + 中心点欧氏距离 < 40 px
- [ ] 4.2.2 实现 `update_position(record, detection)`：更新坐标、驻留帧数+1、最后时间戳
- [ ] 4.2.3 处理边界：多个缓存记录匹配同一检测 → 选距离最近的
- [ ] 4.2.4 处理边界：一个检测匹配多个缓存 → 选距离最近，其余标记 unmatched
- [ ] 4.2.5 单元测试：构造多目标靠近场景，验证匹配不串扰

### 4.3 状态机
- [ ] 4.3.1 实现 `StateMachine` 类，**五个**状态常量：`IDLE, WARNING, CONFIRMED, CLEARED, MANUAL_REQUIRED`
- [ ] 4.3.2 实现全部 9 条状态迁移规则：

  | # | 当前状态 | 条件 | 下一状态 |
  |---|---------|------|---------|
  | 1 | IDLE | 连续驻留 ≥ 5 帧 | WARNING |
  | 2 | WARNING | Qwen 返回异常 | CONFIRMED |
  | 3 | WARNING | Qwen 返回正常 / 目标消失 ≥3s | IDLE |
  | 4 | CONFIRMED | 目标消失 ≥ 3s | CLEARED |
  | 5 | CONFIRMED | 持续 > 30s（周期性） | CONFIRMED |
  | 6 | CLEARED | Qwen 确认风险消除 | IDLE |
  | 7 | CLEARED | Qwen 判定风险未消除 | MANUAL_REQUIRED |
  | 8 | MANUAL_REQUIRED | 人工确认风险消除 | IDLE |
  | 9 | MANUAL_REQUIRED | 人工确认仍存在 | CONFIRMED |

- [ ] 4.3.3 单元测试：对每种状态迁移写独立测试用例，验证时间戳更新、驻留计数

### 4.4 过期清理
- [ ] 4.4.1 实现 `cleanup(current_time)`：
  - IDLE/WARNING：3s 未命中 → 删除
  - CONFIRMED：3s 未命中 → 转 CLEARED（不删除）
  - CLEARED：Qwen 处理完毕 → 删除或回 IDLE
  - MANUAL_REQUIRED：300s（5min）未获人工确认 → 记录告警日志并保持状态
- [ ] 4.4.2 单元测试：构造超时场景，验证清理/状态转换

**Phase 4 完成标准**：位置缓存匹配逻辑有完备测试，状态机 9 条路径可单测验证

---

## Phase 5：Qwen 大模型调用（OpenAI 兼容 API）

**目标**：三种调用场景全部实现，Prompt 模板定义清晰，LM Studio API 对接稳定

### 5.1 API 对接（OpenAI SDK）
- [ ] 5.1.1 实现 `QwenAPIClient` 类：
  - 使用 `openai` Python SDK，配置 `base_url=http://localhost:1234/v1`，`api_key=lm-studio`
  - 调用 `client.chat.completions.create(model="qwen3.5-9b", messages=[...])`
  - 视觉输入：图片转 base64 → 嵌入 `content` 为 `[{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]`
  - 超时 15s，失败重试 3 次（指数退避）
- [ ] 5.1.2 与 Phase 4 状态机解耦：Qwen 模块通过回调/队列接收推理请求
- [ ] 5.1.3 连通性测试：确认 LM Studio 运行中 → 用一张任意图片验证 API 可达 + 响应正确

### 5.2 事件确认 Prompt（场景一）
- [ ] 5.2.1 编写 `prompt_event_confirm` 模板：
  - System: 你是高速公路交通事件分析专家...
  - User: 目标裁剪图（448×448，base64）+ 上下文（类别、车道号、距离、驻留时长）
  - 要求输出 JSON：`{"is_anomaly": bool, "event_type": "违停|抛锚|事故|遗撒|正常", "confidence": float, "reasoning": str}`
- [ ] 5.2.2 实现响应解析器：JSON 提取 + 容错（非 JSON 时正则回退）
- [ ] 5.2.3 测试：用 3~5 张真实交通场景图验证输出合理

### 5.3 状态刷新 Prompt（场景二）
- [ ] 5.3.1 编写 `prompt_state_refresh` 模板
- [ ] 5.3.2 实现响应解析
- [ ] 5.3.3 测试

### 5.4 风险消除 Prompt（场景三）
- [ ] 5.4.1 编写 `prompt_risk_clearance` 模板（纯文本推理，无图片）
- [ ] 5.4.2 实现响应解析
- [ ] 5.4.3 测试

### 5.5 Qwen 调用调度
- [ ] 5.5.1 实现 `QwenCallScheduler`：
  - 管理推理队列（先进先出）
  - 控制并发（≤ 2 并发，避免占满 LM Studio）
  - 超时处理（单个请求超时 15s 则跳过）
  - 延迟监控：记录从入队到结果返回的时间，确保 ≤ 5s 预算
- [ ] 5.5.2 与状态机集成：状态机产生事件→入队→Qwen 推理→回调更新状态

**Phase 5 完成标准**：三种 Prompt 均可正常推理，响应解析无崩溃，调度队列不丢请求

---

## Phase 6：数据闭环

**目标**：错误样本自动归档，评估指标计算

### 6.1 错误样本归档
- [ ] 6.1.1 实现 `Archiver` 类：
  - 目录结构：`archive/{event_confirm,risk_clearance}/{TP,FP,FN}/`
  - 每个样本：裁剪图（PNG）+ 元数据（JSON）
  - 归档时机：Qwen 判定后 + 人工复核标签到达后
- [ ] 6.1.2 归档元数据 schema（见设计文档 §4.2）
- [ ] 6.1.3 单元测试：构造 TP/FP/FN 各一条，验证写入路径和内容正确

### 6.2 评估指标
- [ ] 6.2.1 实现 `MetricsCalculator`：
  - 事件确认：Precision, Recall, F1, Accuracy
  - 风险消除：Precision, Recall, F1, Accuracy
  - Qwen 调用次数 / 总帧数 比值（H1 验证）
- [ ] 6.2.2 实现 `MetricsReport`：Markdown 格式报告，含混淆矩阵
- [ ] 6.2.3 单元测试：用已知标注数据验证计算正确

**Phase 6 完成标准**：归档目录结构正确，指标计算准确

---

## Phase 7：系统集成与实验验证

**目标**：端到端跑通，验证 6 项核心假设

### 7.1 主循环集成
- [ ] 7.1.1 实现 `MainPipeline` 类：
  ```python
  while True:
      frame = video_sampler.next()
      detections = yolo.detect(frame)          # 路径 A
      detections = size_filter.filter(detections)
      detections = flicker_suppressor.process(detections)
      records = position_cache.match_or_create(detections)
      for rec in records:
          rec.lane, rec.dist = lut.lookup(rec.center_x, rec.center_y)
          new_state = state_machine.transition(rec)
          if new_state.requires_qwen:
              qwen_scheduler.enqueue(rec)
      cleanup_expired()
  ```
- [ ] 7.1.2 添加实时可视化（可选）：OpenCV imshow 叠加检测框 + 状态标签
- [ ] 7.1.3 添加性能日志：每帧耗时（YOLO / LUT / 匹配 / Qwen 各段延迟，总延迟 ≤ 5s）

### 7.2 实验准备
- [ ] 7.2.1 收集 ≥ 3 段测试视频（正常交通 / 含违停 / 含事故场景），每段 ≥ 5 分钟
- [ ] 7.2.2 准备棋盘格标定板（A3）×1
- [ ] 7.2.3 搭建测试环境：PC + 摄像头 + LM Studio 已加载 Qwen3.5-9B

### 7.3 假设验证（H1~H6）

| # | 假设 | 验证方法 | 通过标准 |
|---|------|---------|---------|
| H1 | YOLO 初筛减少 Qwen 调用 | 统计 Qwen 调用次数 vs baseline「逐帧调用」 | 减少 ≥ 90% |
| H2 | 位置缓存低帧率下稳定 | 人工标注 100 帧 GT ID，对比缓存匹配 | 匹配成功率 ≥ 95% |
| H3 | 查找表定位准确 | 实测地面标记点，对比查表结果 | 车道号正确，距离误差 < 10% |
| H4 | Qwen 事件确认准确 | 人工标注事件真值，计算 precision/recall | 准确率 ≥ 85% |
| H5 | 风险消除逻辑正确（含 MANUAL_REQUIRED） | 构造「出现→消失→未消除→人工介入」全链路 | 正确触发全链路 |
| H6 | 首日自动标定可用 | 完整执行标定流程 | 标定成功，LUT 可用 |

### 7.4 实验报告
- [ ] 7.4.1 记录 H1~H6 结果
- [ ] 7.4.2 记录失败案例并分析原因
- [ ] 7.4.3 输出最终结论：技术路线是否有效，下一步改进方向

**Phase 7 完成标准**：全系统端到端跑通，6 项假设验证完毕，实验报告产出

---

## 依赖关系图

```
Phase 1 (基础框架)
    ├──→ Phase 2 (YOLO 检测 + 过滤) ──┐
    ├──→ Phase 3 (车道线 + 查找表) ──┤
                                     ├──→ Phase 4 (位置缓存 + 状态机)
                                     │         │
                                     │         ▼
                                     │    Phase 5 (Qwen API 调用)
                                     │         │
                                     ├─────────┤
                                     │         ▼
                                     ├──→ Phase 6 (数据闭环)
                                     │
                                     └──→ Phase 7 (集成 + 验证)
```

- **Phase 2 和 Phase 3 可并行开发**
- **Phase 5 可与 Phase 4 后半段并行**（Prompt 模板/API 对接部分）
- **Phase 6 可提前开发**（独立的归档/指标模块）

---

## 决策变更记录

| 决策项 | 变更内容 | 影响 Phase |
|--------|---------|:---:|
| Q1 新增 MANUAL_REQUIRED | 状态机 4→5 状态，迁移规则 7→9 条 | P4, P7 |
| Q2 防闪烁阈值 0.8× | 配置增加第二套阈值 | P1, P2 |
| Q3 Qwen API 格式 | LM Studio + OpenAI 兼容 API，`openai` SDK 调用 | P1, P5 |
| Q4 车道号 硬路肩=0 | LUT 生成逻辑明确编号规则 | P3 |
| Q5 Stretch 映射 | 坐标映射公式确定：x×3, y×1.6875 | P1, P2 |
| Q6 延迟 ≤ 5s | 增加延迟预算约束 | P5, P7 |
| Q7 dataclass + JSON | 模块接口格式确定 | P1, P4 |
| Q8 Immutable Snapshot | LUT 并发策略确定 | P3 |
| Q9 baseline 逐帧调用 | H1 验证方法明确 | P7 |
| Q10 保持 40px 单判据 | 暂不增加辅助匹配条件 | P4 |

---

## 总预估

| Phase | 预估人天 | 关键路径 |
|-------|:---:|:---:|
| Phase 1 | 1~2 天 | 是 |
| Phase 2 | 2~3 天 | 是 |
| Phase 3 | 2~3 天 | 是 |
| Phase 4 | 2~3 天 | 是 |
| Phase 5 | 2~3 天 | 是 |
| Phase 6 | 1~2 天 | 否 |
| Phase 7 | 2~3 天 | 是 |
| **合计** | **12~19 天** | |

> 单人全职开发。Phase 2/3 并行可压缩到 10~16 天。
