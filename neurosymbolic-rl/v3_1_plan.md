# v3.1 重新训练方案

> 基于 v3 实验（19.45%）的分析，三个核心改进同时执行后重新训练。
> 代码库已有 v3 的所有新算子/终端/grammar 修复，在此基础上改。

---

## 改进 1：Learnable Kernel 预训练

**问题**：v3 的 conv3x3_0..7 和 conv5x5_0..3 冻结在随机值，等效于随机投影，浪费了 action space。

**修复**：Phase 1 之前先训练好 kernel，让 RL 从一开始就有有意义的滤波器。

```
实现：
1. 写一个 pretrain_kernels.py
2. 模型：对 10 个 terminal channel 各用 12 个 kernel → adaptive_avg_pool → Linear(120, 1000)
3. 数据：20K 图（20/class），112×112
4. 训练 10 个 epoch，提取 conv3x3 和 conv5x5 的权重
5. 保存为 kernel_bank_pretrained.pt
6. Phase 1 的 SymbolicKernelBank 从这个文件加载初始值（仍然冻结，但不再是随机值）
```

预计耗时：10 分钟。

---

## 改进 2：放宽 Phase 1 入库门槛

**问题**：v3 用 min_accuracy=0.08 + correlation=0.85，每个 bank 只收了 ~2,500 条。大量弱但独特的公式被扔掉。

**修复**：改配置文件 `tensor_vsr_imagenet_v3.yaml`：

```yaml
strategy:
  min_accuracy_threshold: 0.004    # 从 0.08 改为 0.002（和 v2 一致）
  correlation_threshold: 0.92      # 从 0.85 改为 0.95（放宽，让更多公式进入）
  feature_bank_size: 5000          # 保持每 bank 5000
```

目标：4 banks × ~4,000–5,000 = 16,000–20,000 条公式。

---

## 改进 3：Layer 2 层级公式

**问题**：当前公式只能表达单通道的统计量，无法表达跨公式的语义组合（如"上蓝下绿=风景"）。

**实现**：Phase 1A 完成后，增加 Phase 1B。

### Phase 1B 具体流程：

**Step 1：Forward Selection 选 top-100 Layer 1 bodies。**

不按 accuracy 排序——按**互补性**贪心选择：

```
1. 准备一个小的 val set（5K 图，5/class）
2. 对所有 Layer 1 formula bodies，去掉 final pooling，提取 feature map
3. 每条 body 用 8 SPP pools 编码成 8 个标量

贪心循环（100 轮）：
  Round 1: 选单条 val accuracy 最高的 body → 加入 selected
  Round 2: 对每条候选 body，临时加入 selected，用当前 selected 的所有特征训练一个快速 Linear 分类器（几步 SGD），看 val accuracy 提升多少 → 选提升最大的
  Round 3-100: 重复 Round 2
  
  如果某轮最大提升 < 0.1%，提前停止。
  
  每轮测试时不用遍历所有候选，随机抽 500 条测试即可（加速）。
```

输出：100 条最互补的 Layer 1 formula bodies，保存为 `layer1_bases.json`。

**Step 2：Run Phase 1B。**

```
- 110 个终端：10 个原始通道 (I_R..I_BY) + 100 个 Layer 1 feature maps (L1_0..L1_99)
- build_data_batch_L2() 需要新增：对每个 batch 先执行 100 条 Layer 1 body 得到 feature maps，加入 data_batch
- 4 banks，max_depth=5，max_seq_len=12（L1 已做空间变换，L2 公式可以更短）
- min_accuracy=0.002, correlation_threshold=0.95
- early stopping 和 Phase 1A 一样
```

---

## 完整执行流程

```
Step 0: 预训练 kernel（10 分钟）
  → kernel_bank_pretrained.pt

Step 1: Phase 1A — Layer 1 RL（~30 分钟）
  → 4 banks, 112×112, 预训练 kernel 冻结
  → min_acc=0.002, corr=0.95
  → 目标 16,000-20,000 条公式

Step 2: Forward Selection（~1 小时）
  → 从 Layer 1 中选出 100 条最互补的 bodies
  → layer1_bases.json

Step 3: Phase 1B — Layer 2 RL（~30 分钟）
  → 110 terminals, 4 banks, max_depth=5
  → 目标 ~8,000-10,000 条 Layer 2 公式

Step 4: Phase 3 — 特征提取 + 分类（~8 小时）
  → (L1 + L2) bodies × 12 encodings (8 SPP + 4 histogram) × 2 resolutions (112 + 224)
  → L1 feature selection → ~50K
  → Feature interactions (top-500 pairwise products)
  → L1 selection again → final ~50K
  → 500K 训练图, nn.Linear, sweep wd=[10, 20, 50, 100], 30 epochs

Step 5: 报告结果
```

---

## Phase 2 说明

**跳过 Phase 2。** Phase 1 的 correlation_threshold=0.92 已做去重。Phase 3 的 L1 feature selection 会自动筛掉冗余特征。v2/v3 的经验表明 Phase 2 耗时长、收益小、还容易出 bug（terminal 不匹配问题）。

---

## 关键配置变化 vs v3

| 参数 | v3 | v3.1 |
|------|-----|------|
| min_accuracy_threshold | 0.08 | **0.004** |
| correlation_threshold | 0.85 | **0.92** |
| learnable kernel init | 随机 | **预训练** |
| Layer 2 | 无 | **有（100 个 L1 base）** |
| Phase 2 | 有（出了 bug） | **跳过** |
| 训练图数量 | 200K | **500K** |

---

## 注意事项

1. **FP32 throughout.** 不要用 FP16。
2. **Grammar 规则保持 v3 的修复**：pooling 只在末尾，禁止连续重复一元算子。
3. **build_data_batch 必须包含全部 10 个终端**（v3 的 Phase 2 bug 不能重犯）。
4. **Kernel bank 文件管理**：预训练权重保存一份，Phase 1A/1B 都用同一份，整个流程 kernel 值不变（端到端微调留到后续实验）。
5. **Forward selection 的 quick eval 不需要完整训练**——每轮用 3-5 步 SGD 在 5K val set 上估算就够了，精确度不重要，排序正确就行。
