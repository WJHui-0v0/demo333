

## LoRA 与 QLoRA

普通 LoRA 使用 bf16 base model：

```json
"method": "lora"
```

QLoRA 使用 4-bit NF4 base model：

```json
"method": "qlora"
```

QLoRA 不建议直接对 4-bit 模型执行 `merge_and_unload()`。项目默认保存 adapter，推理时重新加载干净 base model，再挂载 adapter。

## 保存、加载和预测

LoRA adapter 目录包含 `adapter_config.json` 和 `adapter_model.safetensors`，必须使用：

```python
model = model_config.load_adapter("path/to/best_lora")
```

只有已经合并后的完整模型才使用：

```python
model = model_config.load_model("path/to/merged_model")
```

单条预测：

```bash
python predictor.py --arg ./args/arg1.json \
  --weight ../autodl-tmp/checkpoint/exp1/best_lora_final \
  --text "Sp1 binds to CBF."
```

训练和预测必须共用 `build_prompt_text()`，生成后只解析 prompt 之后的新 token。

## 评估

实体同时匹配 `type` 和在原句中的起止位置。项目输出每类实体以及 micro/macro 平均 Precision、Recall 和 F1。实体名称先在原句定位，再与标注位置比较。

## SwanLab 和显存

训练器记录：

```text
train/peak_allocated_mib
train/peak_reserved_mib
eval/peak_allocated_mib
eval/peak_reserved_mib
test/peak_allocated_mib
test/peak_reserved_mib
```

`allocated` 是模型实际使用过的最大显存；`reserved` 是 PyTorch 缓存分配器保留过的最大显存。比较实验时优先关注 `peak_allocated_mib`，并用 `nvidia-smi` 确认进程是否接近显卡总容量。

建议起始配置：

```json
"batch_size": 4,
"eval_batch_size": 1,
"max_length": 512,
"max_new_tokens": 128
```

## 常见问题

### `ValueError: max_length 太小`

完整训练序列超过最大长度。把 `max_length` 从 `256` 增加到 `512` 或 `768`，不要直接删除检查，否则可能得到全部为 `-100` 的 labels。

### `right-padding was detected`

确认 tokenizer 和手动 collator 都使用左填充。

### `Expected all tensors to be on the same device`

在训练和评估循环中显式执行：

```python
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
labels = batch["labels"].to(device)
```

### `Already found a peft_config attribute`

不要在已有 PEFT 模型上再次加载 adapter。释放旧模型后重新加载干净 base model，再调用 `PeftModel.from_pretrained()`。

### `libgomp: Invalid value for environment variable OMP_NUM_THREADS`

```bash
export OMP_NUM_THREADS=8
```

或：

```bash
unset OMP_NUM_THREADS
```

## 实验记录

每次实验建议记录训练方法、batch size、评估 batch size、序列长度、生成长度、learning rate、LoRA rank/alpha/dropout、train/eval/test F1、峰值显存、JSON parse error rate 和最佳 adapter 路径。比较 LoRA 与 QLoRA 时，应保持数据、prompt、序列长度、生成长度和评估代码一致。

## 参考

- [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- [BC2GM Corpus](https://github.com/spyysalo/bc2gm-corpus)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [SwanLab](https://www.swanlab.cn/)



# QwenNER — 基于 Qwen 大模型的命名实体识别

使用 Qwen2.5-7B-Instruct和Qwen2.5-7B模型，通过 QLoRA / LoRA 在 BC2GM数据集上进行命名实体识别（NER）。

## 项目结构

```
QwenNER/
├── data/                  # 数据集
│   ├── train.json         # 训练集 
│   ├── dev.json           # 验证集 
│   ├── test.json          # 测试集 
│   └── labels.json        # 实体类别
├── args/                  # 训练配置文件
│   ├── arg1.json          
├── model.py               # 模型定义 (Qwen4NER 类)
├── trainer.py             # 训练主入口 + Trainer 类
├── MyDataset.py           # 数据集与 collate 函数
├── utils.py               # 工具: Arguments, Metrics, EarlyStop
├── predict.py             # 推理脚本
├── analysis.py            # 序列长度分布分析
├── template.py            # Qwen 对话模板定义
├── requirements.txt       # 依赖列表

```



### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 下载基础模型

```bash
# 下载 模型
hf download Qwen/Qwen2.5-7B --local-dir ./Qwen2.5-7B
```

### 3. 配置训练参数

编辑 `args/arg1.json`，完整参数：

```json
{
    "num_epochs": 5,
    "batch_size": 4,
    "lr":2e-05,
    "weight_decay": 0.01,
    "device": "cuda:0",
    "model_name": "Qwen2.5-7B",
    "model_dir": "../autodl-tmp/model/Qwen2.5-7B",
    "dropout_rate": 0.2,
    "data_path": "./data/",
    "max_length":400 ,
    "max_new_tokens": 300,
    "patience":10,
    "monitor": "val_f1",
    "delta":0.0001,
    "save_dir": "../autodl-tmp/checkpoint/",
    "warmup_steps": 100,
    "eps":1e-8,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "method": "lora",
    "lora_target_modules": [ 
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"],
    "prompt": "You are an expert in biomedical named entity recognition. Your task is to identify gene and protein entities from the given English biomedical text. The entity type is defined as follows: GENE includes gene or protein names, such as gene products, enzymes, receptors, antibodies, cytokines, and similar molecules. Please strictly output the results in the following JSON format: {\"entities\": [{\"name\": \"entity name\", \"type\": \"entity type\"}]}. You must output only a valid JSON string with no additional content. Only use the predefined entity type \"GENE\"; do not recognize or include any other entity types. Only extract entities that clearly belong to the GENE category as defined, and do not include any reasoning or explanations—just the JSON output. The input sentence is provided below.",
    "template_name": "qwen"
}
```

### 4. 训练

```bash
python trainer.py --arg ./args/arg1.json
```


### 5. 推理

```bash
python predict.py \
    --arg ./args/arg1.json \
    --weight ./checkpoint/exp1/ \
    --text "Using the same approach we have shown that hFIRE binds the stimulatory proteins Sp1 and Sp3 in addition to CBF"
```

输出示例：

```json
[{"entities": [{"name": "hFIRE", "type": "GENE"}, {"name": "Sp1", "type": "GENE"}, {"name": "Sp3", "type": "GENE"}, {"name": "CBF", "type": "GENE"}]}]
```

## 数据集


数据格式：

```json
{
    "sentence": "Comparison with alkaline phosphatases and 5 - nucleotidase",
    "entities": [
        {"name": "alkaline phosphatases", "type": "GENE", "pos": [16, 37]}
    ]
}
```


## 📈 评估指标

在实体级别进行精确匹配评估：

- **Precision**：预测正确的实体数 / 预测实体总数
- **Recall**：预测正确的实体数 / 真实实体总数
- **F1**：精确率和召回率的调和平均
- 支持按实体类型单独计算和 micro/macro 平均

### 实验结果
#### 使用了LoRA，注入 "q_proj","k_proj","v_proj","o_proj" 
**Qwen2.5-7B**
|  | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| GENE | 0.8254 | 0.8203 | 0.8229 | 6323 |
| macro_avg | 0.8254 | 0.8203 | 0.8229 | - |
| micro_avg | 0.8254 | 0.8203 | 0.8229 | - |
**Qwen2.5-7B-Instruct**
| | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| GENE | 0.821344 | 0.821604 | 0.821474 | 6323.0 |
| macro_avg | 0.821344 | 0.821604 | 0.821474 | NaN |
| micro_avg | 0.821344 | 0.821604 | 0.821474 | NaN |

<figure>
  <img src="img/lora2.png" alt="LoRA 训练结果">
  <figcaption>Qwen2.5-7B-Instruct使用了LoRA，注入"q_proj","k_proj","v_proj","o_proj"，显存占用</figcaption>
</figure>

#### 使用了LoRA，注入"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj" 
**Qwen2.5-7B**
|  | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| GENE | 0.8365 | 0.8331 | 0.8348 | 6323 |
| macro_avg | 0.8365 | 0.8331 | 0.8348 | - |
| micro_avg | 0.8365 | 0.8331 | 0.8348 | - |
**Qwen2.5-7B-Instruct**
|               | Precision | Recall | F1 | Support |
|---------------|-----------|---------|----------|---------|
| GENE | 0.837331 | 0.833623 | 0.835473 | 6323.0 |
| macro_avg | 0.837331 | 0.833623 | 0.835473 | NaN |
| micro_avg | 0.837331 | 0.833623 | 0.835473 | NaN |

<figure>
  <img src="img/lora1.png" alt="LoRA 训练结果">
  <figcaption>Qwen2.5-7B-Instruct使用了LoRA，注入"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"，显存占用</figcaption>
</figure>

#### 使用了QLoRA，注入"q_proj","k_proj","v_proj","o_proj" 
**Qwen2.5-7B**
|  | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| GENE | 0.8251 | 0.8172 | 0.8211 | 6323 |
| macro_avg | 0.8251 | 0.8172 | 0.8211 | - |
| micro_avg | 0.8251 | 0.8172 | 0.8211 | - |
**Qwen2.5-7B-Instruct**
|  | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| GENE | 0.825468 | 0.816068 | 0.820741 | 6323.0 |
| macro_avg | 0.825468 | 0.816068 | 0.820741 | NaN |
| micro_avg | 0.825468 | 0.816068 | 0.820741 | NaN |

<figure>
  <img src="img/qlora1.png" alt="QLoRA 训练结果">
  <figcaption>Qwen2.5-7B-Instruct使用了QLoRA，注入"q_proj","k_proj","v_proj","o_proj"，显存占用</figcaption>
</figure>

#### 使用了QLoRA，注入"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj" r=8，α=32
**Qwen2.5-7B**
|  | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| GENE | 0.8333 | 0.8238 | 0.8285 | 6323 |
| macro_avg | 0.8333 | 0.8238 | 0.8285 | - |
| micro_avg | 0.8333 | 0.8238 | 0.8285 | - |
**Qwen2.5-7B-Instruct**
|  | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| GENE | 0.832987 | 0.824292 | 0.828617 | 6323.0 |
| macro_avg | 0.832987 | 0.824292 | 0.828617 | NaN |
| micro_avg | 0.832987 | 0.824292 | 0.828617 | NaN |


<figure>
  <img src="img/qlora2.png" alt="alt text">
  <figcaption>Qwen2.5-7B-Instruct使用了QLoRA，注入"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"，显存占用</figcaption>
</figure>


# Qwen2.5-7B BC2GM NER

基于 Qwen2.5-7B微调的生物医学实体识别项目。本项目使用BC2GM数据集，比较 **LoRA (BF16)** 与 **QLoRA (4-bit NF4)** 两种微调方案的效果与显存消耗。

---

## 1. 项目结构

```text
demo3/
├── args/                       # 配置文件目录
│   ├── lora.json               # LoRA配置文件
│   └── qlora.json              # QLoRA配置文件
├── data/                       # BC2GM 数据集
│   ├── dev.json
│   ├── test.json
│   └── train.json 
├── swanlog/               # swanlab 本地日志  
├── model.py               # 底座加载、LoRA/QLoRA 封装、优化器、adapter 读写  
├── Dataset.py             # 数据集与 DataLoader 构建  
├── prompt_utils.py        # prompt 构造, 生成 eos_token_id 等工具  
├── trainer.py             # 训练、评估、显存监控、swanlab 日志、主入口  
├── utils.py               # Arguments、Metrics（实体级 P/R/F1）、日志与保存工具  
└── predict.py             # 推理脚本：加载 LoRA adapter 对新文本预测实体



```

---

## 2. 核心设计与评估逻辑

直接让生成式模型输出字符位置（如 `"start": 16, "end": 37`）难度大，数字错一点就会被判错。
因此本项目将任务拆分：

1. **模型输出**：只负责识别实体名称和类型。
```json
{
  "entities": [
    {
      "name": "alkaline phosphatases",
      "type": "GENE"
    }
  ]
}

```


2. **处理检索（`metrics.py`）**：输出的 `name`，到原始句子中检索对应的位置，得到 `(start, end, type)` 元组。

### 判定正确的条件

必须同时满足以下三个条件才算预测正确（TP）：

* 实体开始位置 `start` 相同
* 实体结束位置 `end` 相同
* 实体类型 `type` 相同

> **注意（重复实体）**：如果一个实体名在句中出现多次，程序会按模型输出的顺序依次在原句中寻找未被使用的位置。

---

## 3. 快速开始

```bash
cd /root/demo333

# 运行 LoRA 实验
python train.py --config args/lora.json

# 运行 QLoRA 实验
python train.py --config args/qlora.json

```

---

## 4. 实验对比设置与 SwanLab 结果

### 实验控制变量

为了保证公平对比，两组实验保持以下配置完全一致：

* **基座模型**：Qwen2.5-7B
* **LoRA 目标模块**：`q_proj, k_proj, v_proj, o_proj, ` (`r=16, alpha=32`)
* **训练超参**：`lr=2e-5`，Cosine 调度，Batch Size 与 Epochs 完全一致（共 15.6k steps）
* **唯一变量**：是否开启 4-bit NF4 量化 (`use_qlora`)

### 结果汇总

| 方法 | 量化 | Best Dev F1 | Precision | Recall | 合法 JSON 率 | 峰值显存 |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **LoRA** | 否 (BF16) | **0.839** (Step 6k) | 0.843 | 0.836 | ~100% | 19.13 GB |  |
| **QLoRA** | 4-bit NF4 | **0.827** (Step 9.8k) | 0.827 | **0.827** | 100% | **~13.55 GB** | |


---
1. **收敛与过拟合**：
* **LoRA** 在 Step 6,000 左右迅速达到最高点，随后 F1 出现小幅回落，说明训练后期有轻微过拟合趋势。
* **QLoRA** 受4-bit量化影响，前期收敛速度略慢于 LoRA，在Step 9,800达到最高点，之后曲线保持平稳。


2. **JSON 输出稳定性**：
* QLoRA 全程错误率线平直停留在 `0`，LoRA 的解析成功率也基本达到100%，证明 Qwen2.5-7B 微调后输出结构化 JSON 非常稳定。


