# 基于 Transformer 的中文 NER 课程作业

本项目完成了课程要求的中文命名实体识别任务，统一使用中文预训练 Transformer 作为编码器，并在同一套数据处理、训练和预测框架下完成了四组可复查的实验结果：

- `Baseline`：`RoBERTa + Linear`
- `Baseline + LowFreqDrop`：在 `Baseline` 基础上加入低频字输入增强
- `Stage2 CRF`：`RoBERTa + Linear + CRF`
- `Stage3 Refine+CRF`：`RoBERTa + Refinement + CRF`

项目已经包含数据分析、训练、恢复训练、预测导出、格式校验和项目级图表汇总的完整流程。当前最终提交版本为 `Baseline` 路线生成的测试集预测文件。

## 1. 项目概述

任务目标是对中文序列进行逐字标注，标签体系采用 `BIO`，实体类别包括：

- `LOC`
- `ORG`
- `PER`
- `T`

整体流程如下：

1. 读取并校验 `train / dev / test` 数据格式
2. 使用 fast tokenizer 完成字符到 token 的对齐
3. 按不同实验路线训练模型
4. 基于最佳 checkpoint 在 `dev / test` 上做预测
5. 自动校验输出文件格式并生成汇总图表

## 2. 实现概览

### 2.1 三条模型结构路线

- `baseline`：使用 `hfl/chinese-roberta-wwm-ext` 编码后直接接线性分类层。
- `final_stage2`：在 backbone 输出之后接手写线性链 `CRF`。
- `final_stage3`：在 backbone 与分类头之间加入自写轻量 `RefinementEncoder`，再接 `CRF`。

说明：

- `Baseline + LowFreqDrop` 不是单独的第四个 `variant`，而是在 `baseline` 训练时额外打开低频字输入增强开关形成的最小对照实验。
- 当前训练数据并不严格满足全部 `BIO` 约束，因此正式实验统一使用 `bio_constraint_mode="none"`。

### 2.2 工程实现要点

- 使用 fast tokenizer 的 `word_ids()` 做严格字符级对齐。
- 使用 `valid_mask` 只在真实字符对应位置计算损失和指标。
- `CRF` 路线先通过 `compress_for_crf` 把稀疏有效位压缩成连续序列，再送入 `CRF`。
- 长句预测采用 token-aware 滑窗切分，并用 `center_priority` 合并窗口结果。
- 训练阶段支持 `BucketBatchSampler`，减少 padding 浪费。
- `Baseline + LowFreqDrop` 只在训练输入侧对低频字按概率替换为 `UNK`。
- 预测后自动检查行数、空行位置与每行标签数，确保提交文件格式合法。
- `Stage3` 额外加入了更保守的新增层学习率、attention 数值稳定处理和 `Non-finite loss` 快速失败检查。

## 3. 代码结构

```text
.
├── config.py              # 统一配置，管理 profile 与 variant
├── data.py                # 数据读取、对齐、统计与 DataLoader 构造
├── crf.py                 # 手写线性链 CRF
├── model_refine.py        # 手写轻量 refinement 模块
├── model_backbone.py      # 统一 NER 模型封装
├── evaluate.py            # token accuracy / entity-F1 评估
├── train.py               # 训练、checkpoint 保存与 resume
├── predict.py             # dev/test 预测与格式校验
├── plot_curves.py         # 单实验图与项目级汇总图
├── main.py                # 命令行入口
├── data/                  # 课程提供的数据与统计结果
└── outputs/               # 正式实验目录与项目级汇总结果
```

## 4. 运行环境

- Python `3.10` 或 `3.11`
- `torch>=2.1.0`
- `transformers>=4.39.0`
- `tokenizers>=0.15.0`
- 正式实验推荐在 Linux + CUDA 环境运行

当前仓库中已落盘实验的 `experiment_summary.csv` 显示，正式实验运行在 `NVIDIA A100-SXM4-80GB` 环境。

安装依赖：

```bash
pip install -r requirements.txt
```

项目提供两个 profile：

- `local_debug`：本地联调用，默认 `max_len=128`
- `cloud_train`：正式实验用，默认 `max_len=384`

## 5. 常用命令

### 5.1 数据分析

```bash
python main.py --mode analyze --profile cloud_train --variant baseline
```

会生成：

- `data/data_report.json`
- `data/label_stats.json`
- 对应实验目录下的数据统计图

### 5.2 训练

训练 `Baseline`：

```bash
python main.py --mode train_baseline --profile cloud_train
```

训练 `Stage2 CRF`：

```bash
python main.py --mode train_final --profile cloud_train --variant final_stage2
```

训练 `Stage3 Refine+CRF`：

```bash
python main.py --mode train_final --profile cloud_train --variant final_stage3
```

训练 `Baseline + LowFreqDrop` 最小对照实验：

```bash
python main.py --mode train_baseline --profile cloud_train --use_low_freq_char_dropout --low_freq_char_threshold 1 --low_freq_char_dropout_prob 0.5
```

恢复训练：

```bash
python main.py --mode resume --profile cloud_train --resume_from outputs/某个实验目录
```

### 5.3 预测并导出提交文件

以最终推荐的 `Baseline` 为例：

```bash
python main.py --mode predict --profile cloud_train --variant baseline --experiment_dir outputs/baseline_roberta_linear_20260408_103539 --output_filename 2023211751.txt
```

预测结束后会生成：

- `predictions/dev_pred.txt`
- `predictions/2023211751.txt`
- `results/output_validation.json`
- `results/dev_prediction_metrics.json`

### 5.4 重画图表

```bash
python main.py --mode plot --profile cloud_train --variant baseline --experiment_dir outputs/baseline_roberta_linear_20260408_103539
```

## 6. 实验结果

当前项目级汇总结果包含三条模型结构路线，以及一条最小输入增强对照实验。

### 6.1 最佳 epoch 与开发集指标

| 路线 | 实验目录 | 最佳 epoch | 最佳 dev accuracy | 最佳 checkpoint dev F1 | 训练时间 |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline | `baseline_roberta_linear_20260408_103539` | 6 | 0.996749 | 0.968576 | 03:34:08 |
| Baseline + LowFreqDrop | `baseline_roberta_linear_lowfreqdrop_20260411_150237` | 5 | 0.996654 | 0.967328 | 02:33:33 |
| Stage2 CRF | `final_stage2_roberta_crf_20260408_105834` | 1 | 0.994697 | 0.953846 | 04:11:31 |
| Stage3 Refine+CRF | `final_stage3_roberta_refine_crf_20260408_160054` | 3 | 0.995740 | 0.962612 | 04:16:10 |

说明：

- `best_dev_accuracy` 来自训练日志中的最佳 epoch。
- `best_checkpoint_dev_f1` 来自最佳 checkpoint 在开发集上的重新评估结果。
- 四条已完成实验线中，`Baseline` 的开发集 `dev_accuracy` 和 best checkpoint `entity-F1` 都是最高值。

### 6.2 开发集预测指标

| 路线 | token accuracy | entity precision | entity recall | entity F1 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.996747 | 0.965525 | 0.971647 | 0.968576 |
| Baseline + LowFreqDrop | 0.996654 | 0.963059 | 0.971635 | 0.967328 |
| Stage2 CRF | 0.994703 | 0.951614 | 0.956089 | 0.953846 |
| Stage3 Refine+CRF | 0.995732 | 0.962756 | 0.962467 | 0.962612 |

四条实验线当前的 `output_validation.json` 都通过了内部格式检查，说明导出的预测文件在行数、空行位置和标签数量上与原始数据保持一致。

## 7. 输出文件说明

每个实验目录下主要包含以下内容：

- `checkpoints/`：最佳模型、最后一轮模型、优化器与调度器状态
- `predictions/`：开发集预测文件与最终提交文件
- `results/`：训练日志导出、预测指标、格式校验结果
- `figures/`：该实验对应的训练曲线与统计图

`outputs/project_summary/` 下保存的是项目级汇总结果，包括：

- 四条实验线的训练曲线对比图
- 最佳 checkpoint 的开发集 F1 对比图
- 最佳 epoch 汇总表
- 数据统计摘要

## 8. 最终提交

若依据当前开发集结果选择最终提交版本，推荐使用 `Baseline` 路线导出的提交文件：

`outputs/baseline_roberta_linear_20260408_103539/predictions/2023211751.txt`

仓库中同时保留了四条实验线各自的提交文件，便于后续复查和对比：

- `outputs/baseline_roberta_linear_20260408_103539/predictions/2023211751.txt`
- `outputs/baseline_roberta_linear_lowfreqdrop_20260411_150237/predictions/2023211751.txt`
- `outputs/final_stage2_roberta_crf_20260408_105834/predictions/2023211751.txt`
- `outputs/final_stage3_roberta_refine_crf_20260408_160054/predictions/2023211751.txt`

## 9. 文档与说明

- 课程报告见 [实验报告.md](实验报告.md)
- 本项目在开发和写作过程中使用了 `ChatGPT` 作为辅助工具，其参与范围主要限于表达润色、文档整理和参考文献格式辅助。
- 任务拆解、实验路线设计、代码实现、结果核对和最终提交判断均由本人完成。
