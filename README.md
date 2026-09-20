# Demo3: Qwen2.5-7B 生物医学命名实体识别微调

本项目使用 Qwen2.5-7B-Instruct，在 BC2GM 数据集上进行基因/蛋白质命名实体识别微调，支持：

- LoRA 微调
- QLoRA 4-bit NF4 微调
- BF16 混合精度
- 梯度累积与梯度检查点
- AdamW 与 8-bit Paged AdamW 优化器
- 验证集和测试集生成式评估
- checkpoint、optimizer、scheduler 和训练状态保存
- 断点续训和 checkpoint
- 训练过程中的显存峰值统计
- SwanLab 实验日志

## 一、项目结构

```text
demo333
├── args/
│   ├── lora.json
│   └── qlora.json
├── data/
│   ├── train.json
│   ├── dev.json
│   ├── test.json
│   └── labels.json
├── Dataset.py       # 数据读取、prompt 构造和 batch padding
├── model.py         # Qwen2.5、LoRA、QLoRA 和优化器
├── prompt_utils.py  # 对话模板和目标输出格式
├── trainer.py       # 自定义训练、评估、保存和恢复
├── predict.py       # 单条文本预测
└── utils.py         # 参数、指标、日志和最佳模型保存
```

## 二、环境

```text

torch==2.8.0+cu128
transformers==5.8.0
peft==0.18.1
accelerate==1.11.0
tokenizers==0.22.2
swanlab==0.10.0
tqdm==4.70.0
bitsandbytes==0.46.1

```

## 三、准备模型

默认配置读取本地模型：

```text
../autodl-tmp/Qwen2.5-7B-Instruct
```

## 四、数据格式

训练、验证和测试文件均为 JSON 数组
```
[
  {
    "sentence": "The expression of FSH was measured.",
    "entities": [
      {
        "name": "FSH",
        "type": "GENE",
        "pos": [21, 24]
      }
    ]
  }
]
```

## 五、训练

### LoRA

```bash
python trainer.py --arg ./args/lora.json
```
主要参数：

```
{
  "num_epochs": 3,
  "micro_batch_size": 4,
  "gradient_accumulation_steps": 1,
  "max_length": 400,
  "eval_batch_size": 4,
  "bf16": true,
  "gradient_checkpointing": false,
  "optim": "adamw_torch",
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.1,
  "lora_target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
  ]
}
```

### QLoRA

```bash
python trainer.py --arg ./args/qlora.json
```

主要参数：

```
{
  "micro_batch_size": 4,
  "gradient_accumulation_steps": 1,
  "max_length": 400,
  "eval_batch_size": 4,
  "bf16": true,
  "gradient_checkpointing": true,
  "optim": "paged_adamw_8bit",
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05
}
```

QLoRA 基座模型使用 4-bit NF4 量化和 double quantization，计算精度为 BF16。普通 LoRA 使用完整 BF16 基座模型。


## 六、Checkpoint 和断点续训

每次训练会自动创建新的实验目录，例如：

```text
checkpoint/exp1/
├── checkpoint-200/
├── checkpoint-400/
├── best_lora/
├── best_lora_final/
└── log.jsonl
```

checkpoint 中包含：

- LoRA adapter 权重
- optimizer.pt
- scheduler.pt
- trainer_state.json



## 七、实验结果


| 方法 | 测试集 Precision | 测试集 Recall | 测试集 Micro-F1 | 验证集最高 Micro-F1 |
|---|---:|---:|---:|---:|
| LoRA | 0.8317 | 0.8346 | 0.8331 |  0.8324 | 
| QLoRA | 0.8317 | 0.8230 | 0.8273| 0.8316 |



不同阶段的显存峰值如下：

| 方法 | 训练 Allocated | 验证 Allocated | 测试 Allocated |
|---|---:|---:|---:|
| LoRA | 19,093 M | 15,853 M | 15,001 M |
| QLoRA | 13,054 M | 9,331 M | 5,797 M |


