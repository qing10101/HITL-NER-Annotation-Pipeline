# IMPRINT Threat 实验设计：Span 重建攻击与维度存在性攻击

日期：2026-10-29
状态：设计稿，待实现

## 1. 目标与原则

实验要回答一个问题：把 IMPRINT 标出的 span 遮掉之后，攻击者还能从剩下的文本推回多少隐私。两个方案都是纯 ML，不调用 LLM，不需要任何新标注，真值直接来自已有的 span 标注。

论文 3.1 节已经承诺了三个量，实验按这三个量对齐：

| 论文承诺 | 方案 A 对应指标 | 方案 B 对应指标 |
| --- | --- | --- |
| 攻击者推断精度下降 | span 恢复率（exact、category） | 三个维度的 AUROC、F1 |
| 攻击者 certainty 下降 | gold span 的概率质量 | sigmoid 概率、Brier |
| 证据来源转移 | occlusion 归因的 top 上下文词 | integrated gradients 的 top token |

两个方案回答的问题不同，互为印证：

- 方案 A（span 重建攻击）回答"被遮掉的词能不能被猜回来"，借用 SANTEXT（Yue et al. 2021）和 InferDPT 的 mask token inference attack 协议。
- 方案 B（维度存在性分类器）回答"遮掉之后还能不能判断这条评论有没有披露"，借用 Coavoux et al. 2018 和 Elazar & Goldberg 2018 的 attacker classifier 协议。

不需要新标注的原因：方案 A 的真值就是 span 原文本身；方案 B 的标签只有三个，直接从 span 类型读出（MINOR_AGE 或 MINOR_EDU 存在即 minor 为 1），没有任何语义推导。

每个方案都跑两个攻击者：

- 非自适应攻击者：没见过任何隐私 span 标注的对手。方案 A 里是只在负类行的随机 span 上微调过的 infiller，方案 B 里是在原文上训练的分类器。
- 自适应攻击者：在 gold 遮盖后的训练集上训练，专门学残留线索，是真正的压力测试。

两个攻击者的差距精确等于"见没见过隐私标注"的贡献，不依赖预训练目标。

两者差距本身就是一个结果。

## 2. 共用数据准备

一个 `mask.py` 把每条评论生成五个版本，两个方案共用。所有遮盖按字符偏移从后往前替换，避免替换后偏移错位。

| 版本 | 遮什么 | 遮盖依据 | 用途 |
| --- | --- | --- | --- |
| raw | 不遮 | 无 | 攻击上限 |
| gold_mask | 全部 gold span | `entities_json` 的 start/end | 防御上限，主结果 |
| detector_mask | SpanMarker 预测的 span | 预测 CSV 的 `predicted_entities_json` | 可部署的真实效果。**暂缓**：SpanMarker checkpoint 和预测文件当前不在仓库，先不跑，后续重训后补一行 |
| random_mask | 随机选等量的非 span 词 | 同一行内随机采样 | 对照：普通词是不是同样好猜 |
| loo_mask | 每次只遮一个 span，其余保留 | `entities_json` | 仅方案 A，测 span 之间互相泄露 |

替换符两个方案不同：方案 A 用 T5 哨兵 token `<extra_id_k>`；方案 B 主条件直接删除 span 并合并多余空格，原因见第 4 节。

**训练和测试划分**沿用 benchmark 现有切分。测试集是 `test_balanced_400.csv`，400 行里 300 行有 span，共 501 个 span。训练池是 20k 语料去掉这 400 行后的 19,600 行，用 pipeline 标签做 gold_mask。训练池再按 90/10 切 train/dev，dev 用于选 checkpoint 和调阈值，测试集只碰一次。

**偏移校验是第一步。** 测试集打印出来的文本里弯引号显示为乱码（"didn��t"），说明 CSV 存储或读取时编码可能变过。遮盖前对每个 span 校验 `raw_text[start:end] == text`，不相等的行先统计数量，能修的修（用正确编码重读），修不了的剔除并在论文里报剔除数。

**类型词表**用于方案 A 的 category 指标：对训练池的全部 span 按五个类型分别统计归一化后的词频，每个类型取出现至少 2 次的词条。这是纯频率统计，不涉及语义判断。

**归一化规则**两个方案的比对都用：

- 小写，去首尾标点和所有格
- 英文数词转数字（two 转 2）
- 年龄表达规整："5 year old"、"5yo"、"5-year-old"、"5 yr old" 都归为 `age:5`
- 月龄规整："10 month old"、"10 months" 归为 `months:10`
- 其余词做简单 lemmatize（daughters 归 daughter）

## 3. 方案 A：Span 重建攻击

### 3.1 任务

给攻击者遮盖后的评论，让它直接猜被遮掉的内容。真值是 span 原文，不需要任何新标签。

### 3.2 模型选择：T5 而不是 RoBERTa MLM

两个原因：

1. span 多为多词（"5 year old"、"third trimester"）。RoBERTa 要放 k 个 `<mask>`，k 本身泄露长度，Truthful Text Sanitization（arXiv 2412.12928）指出过这个漏洞。T5 用一个哨兵替整个 span，长度不泄露，和真实部署里 `[REDACTED]` 的信息量一致。
2. T5 的预训练目标就是 span infilling，预训练模型零训练就是一个非自适应攻击者。

主模型用 T5Gemma（2025 年）：首选 `google/t5gemma-2-270m-270m`（T5Gemma 2，2025 年 12 月，基于 Gemma 3，文本部分约 370M），备选 T5Gemma 一代的 T5-Base 尺寸 UL2 变体（`google/t5gemma-b-b-ul2`）。非自适应和自适应共用同一个 checkpoint。

**先做 smoke test。** T5Gemma 是从 decoder-only 模型适配来的，零训练时是否按哨兵格式输出 span 内容未经验证。取 20 条测试样本试填空：输出是 span 内容则零训练可用；输出整句重写则非自适应攻击者按第 1 节的新定义，先在负类行的随机 span 上微调一轮再用。两种情况下自适应攻击者的训练都不受影响。

Fallback：原版 `t5-base`（2019 年，220M），预训练目标就是哨兵 span corruption，零训练填空可靠，fp16 和 bf16 都稳。

### 3.3 训练时的输入和输出

每条有 span 的评论变成一对序列。遮盖位置放哨兵 `<extra_id_k>`，按出现顺序编号；目标是"哨兵 + 被遮内容"交替拼接，最后补一个哨兵表示结束。以下三行是测试集里的真实数据：

```
RAW   : Bought this for my 1 year old nephew.  He likes it and it's super cute.  Good quality
INPUT : Bought this for my <extra_id_0> <extra_id_1>.  He likes it and it's super cute.  Good quality
TARGET: <extra_id_0> 1 year old <extra_id_1> nephew <extra_id_2>
LABELS: MINOR_AGE, FAM_KIN
```

```
RAW   : They didn't work for my baby, they were way too big for my 17 lb 10 month old. So they kept sliding down her legs and rolling up. ...
INPUT : They didn't work for my <extra_id_0>, they were way too big for my 17 lb <extra_id_1>. So they kept sliding down her legs and rolling up. ...
TARGET: <extra_id_0> baby <extra_id_1> 10 month old <extra_id_2>
LABELS: MINOR_AGE, MINOR_AGE
```

```
RAW   : So cute! Perfect for my girls' room! ... I don't have to worry about the little ones poking a hole in it! ...
INPUT : So cute! Perfect for my <extra_id_0> room! ... I don't have to worry about the <extra_id_1> poking a hole in it! ...
TARGET: <extra_id_0> girls' <extra_id_1> little ones <extra_id_2>
LABELS: FAM_KIN, MINOR_AGE
```

模型是 encoder-decoder：encoder 读 INPUT，decoder 逐 token 生成 TARGET。训练用 teacher forcing，每步给 decoder 真实前缀，预测下一个 token，损失是 TARGET 上每个 token 的交叉熵。INPUT 里的词不算损失，模型只在"被遮的内容"上受训。

这一对样本教会模型的是：看到 "Bought this for my ___ ___. He likes it"，第一个空大概率是年龄，第二个空大概率是男性晚辈亲属。

`<extra_id_0>` 到 `<extra_id_99>` 是 T5 词表里预留的 token，T5 预训练就是用它们做 span 填空。

**混入的对照样本。** 训练集再加约 800 行没有 span 的评论，随机遮 1 到 3 个词，格式完全相同。目的是防止微调后模型只会吐亲属词，保留通用填空能力，输出更诚实。

### 3.4 两个攻击者

| 攻击者 | 训练数据 | 训练步骤 |
| --- | --- | --- |
| 非自适应 | 零训练，或仅约 800 行负类行的随机 span 遮盖（smoke test 决定） | 直接推理，或 1 个 epoch 的通用填空微调 |
| 自适应 | 训练池里有 span 的约 4,000 行（gold_mask 版本）+ 约 800 行随机遮盖的负类行 | 3 个 epoch 的 seq2seq 微调 |

### 3.5 训练配置

| 项 | 值 |
| --- | --- |
| 模型 | google/t5gemma-2-270m-270m（fallback：t5-base） |
| 优化器 | AdamW，lr 1e-4，weight decay 0.01；显存紧张时换 Adafactor |
| batch | 8，梯度累积 2（等效 16）；T5Gemma 2 显存不够时 batch 4、累积 4 并开 gradient checkpointing |
| epoch | 3 |
| 输入截断 | 384 token |
| 目标截断 | 64 token |
| 精度 | bf16（T5 系列在 fp16 下会溢出，Gemma 系列也只认 bf16） |
| 显存 | RTX 3060 6 GB 可跑；不够时开 gradient checkpointing |
| checkpoint 选择 | dev 上每个 epoch 做一次 generate，按 span 级 exact match |
| 种子 | 3 个，报均值和标准差 |

```python
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

MODEL = "google/t5gemma-2-270m-270m"   # fallback: "t5-base"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

def encode(masked_text, target):
    x = tok(masked_text, max_length=384, truncation=True)
    y = tok(target, max_length=64, truncation=True)["input_ids"]
    x["labels"] = [t if t != tok.pad_token_id else -100 for t in y]
    return x

args = Seq2SeqTrainingArguments(
    output_dir="runs/recon_adapt", learning_rate=1e-4,
    per_device_train_batch_size=8, gradient_accumulation_steps=2,
    num_train_epochs=3, bf16=True,
    eval_strategy="epoch", save_strategy="epoch",
    predict_with_generate=True, generation_max_length=64,
    generation_num_beams=5, load_best_model_at_end=True, seed=42)
```

4,800 行三个 epoch 在单张消费级 GPU 上几分钟。

### 3.6 测试流程

对测试集 300 行有 span 的评论：

1. 做成同样的 INPUT，不给 TARGET，beam search（beam 5）generate，保留 top-5 候选。
2. 按哨兵拆开，得到每个 span 位置的预测。
3. 归一化后和 gold 逐个 span 比对。
4. 把 gold TARGET 喂给 decoder 做 teacher forcing，取各 token log prob 的平均，exp 后作为模型对正确答案的概率质量。

举例，模型输出 `<extra_id_0> 1 year old <extra_id_1> son <extra_id_2>`：

| span | gold | 预测 | exact | category |
| --- | --- | --- | --- | --- |
| 0 | 1 year old | 1 year old | 命中 | 命中 |
| 1 | nephew | son | 未命中 | 命中（son 在 FAM_KIN 词表里） |

category 这一列是关键。"nephew" 被猜成 "son" 虽然词错了，但攻击者已经知道这家有个男性晚辈。"17 lb ___" 几乎一定被猜成 "X month old"，具体数字错不要紧，年龄段泄露了。

### 3.7 指标

| 指标 | 定义 | 对应论文承诺 |
| --- | --- | --- |
| exact@1 / exact@5 | 归一化后命中原 span | 推断精度 |
| category@1 | 预测落在同类型 span 词表里 | 推断精度 |
| gold 概率质量 | exp(teacher forcing 平均 log prob) | certainty |
| exposure rate | 仅 detector_mask：检测器漏掉的 span 算直接暴露，加上被重建的 span，除以总 span 数 | 端到端效果（随 detector_mask 暂缓） |
| 证据归因 | 对每个 span 做 occlusion：逐个删上下文词，看 gold log prob 掉多少，按类型汇总 top-20 词 | evidence shift |

每个类型分开报，FAM_KIN 大概率比 MINOR_AGE 好猜。

### 3.8 要报的对比

- 非自适应 vs 自适应
- gold_mask（主）vs loo_mask（span 互泄）vs random_mask（普通词可猜性的基线）
- detector_mask 的 exposure rate（暂缓）

所有条件用同一个模型，只换 INPUT。

## 4. 方案 B：维度存在性分类器

### 4.1 任务和标签

给攻击者一条评论，判断它有没有 minor、gender、kinship 三类披露。输入纯文本，输出 3 维 0/1：

| 标签 | 规则 |
| --- | --- |
| minor | 行内存在 MINOR_AGE 或 MINOR_EDU span |
| gender | 行内存在 GEN_NOUN 或 GEN_PHYS span |
| kinship | 行内存在 FAM_KIN span |

训练池约 19,600 行，大致 21% 的行有任意披露，gender 行约 3%，要用 pos_weight。

### 4.2 遮盖方式的坑

方案 B 的主条件必须用删除而不是 `[REDACTED]` 占位符。占位符出现本身就告诉模型"这行有披露"，测试集 100 行负类一个占位符都没有，AUROC 会被虚抬。占位符版本可以报成变体，解读为"攻击者知道哪里被删过"的更强设定。

### 4.3 两个攻击者

| 攻击者 | 训练数据 | 测什么 |
| --- | --- | --- |
| A_raw | 原文 | raw（上限）、gold_mask、random_mask |
| A_adapt | gold_mask（删除版） | gold_mask |

detector_mask 条件暂缓，见第 2 节。

### 4.4 训练配置

| 项 | 值 |
| --- | --- |
| 模型 | chandar-lab/NeoBERT（2025 年，250M，`trust_remote_code=True`），`problem_type="multi_label_classification"`；fallback roberta-base（与 SpanMarker 同骨干） |
| 损失 | BCEWithLogitsLoss，pos_weight = neg/pos，clip 到 20 |
| 优化器 | AdamW，lr 2e-5，weight decay 0.01，warmup 6% |
| batch | 16，梯度累积 2（等效 32） |
| epoch | 4 |
| max_len | 256 |
| 精度 | bf16 |
| checkpoint 选择 | dev macro AUROC |
| 阈值 | dev 上按标签调 |
| 种子 | 3 个 |

```python
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)

LABELS = ["minor", "gender", "kinship"]
MODEL = "chandar-lab/NeoBERT"   # fallback: "roberta-base"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(LABELS), trust_remote_code=True,
    problem_type="multi_label_classification")

pos_weight = torch.tensor(neg_counts / pos_counts).clamp(max=20.0)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        logits = model(**inputs).logits
        loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(logits.device))(logits, labels.float())
        return (loss, {"logits": logits}) if return_outputs else loss

args = TrainingArguments(
    output_dir="runs/presence_raw", learning_rate=2e-5,
    per_device_train_batch_size=16, gradient_accumulation_steps=2,
    num_train_epochs=4, warmup_ratio=0.06, weight_decay=0.01, bf16=True,
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="macro_auc", seed=42)
```

数据集每条是 `tok(text, truncation=True, max_length=256)` 加一个 3 维 float 标签向量。19,600 行四轮在 RTX 3060 上约 30 分钟。调试阶段可用 `distilroberta-base`。

**必加一个非神经基线**：TF-IDF（词 1 到 2 gram 加字符 3 到 5 gram）加 logistic regression，同样的两个攻击者和测试条件。它几秒训完，系数直接给出遮盖后残留线索的词表，比 IG 热图更直观，也是 RoBERTa 结果的 sanity check。结果表里作为单独一行。

### 4.5 指标

| 指标 | 定义 | 对应论文承诺 |
| --- | --- | --- |
| AUROC per label | 不依赖阈值，负类多也稳 | 推断精度 |
| F1 at dev threshold | 可读性 | 推断精度 |
| 正类上的平均 p、p 在 0.4 到 0.6 的比例、Brier | sigmoid 输出 | certainty |
| 攻击成功率 | 猜对且 p 不低于 0.8 | 一句话结论 |
| integrated gradients top-20 token | 对遮盖文本做归因（Captum） | evidence shift |

### 4.6 预期和解读

A_raw 在原文上会接近满分，因为标签和 span 词同源，论文里要明说这是泄露上限。真正要看的数字是 A_adapt 在 gold_mask 上还能拿到多少。比如 minor 的 AUROC 从 0.99 掉到 0.75，说明 stroller、diaper 这类行为词还在泄露，这和方案 A 的 category@1 可以互相印证。

## 5. 实验矩阵与结果表格式

### 5.1 方案 A 主表

行是条件，列是指标，每个类型一组列，加 micro 平均：

| 攻击者 | 条件 | exact@1 | exact@5 | category@1 | gold 概率 |
| --- | --- | --- | --- | --- | --- |
| 非自适应 | gold_mask | | | | |
| 非自适应 | loo_mask | | | | |
| 非自适应 | random_mask | | | | |
| 自适应 | gold_mask | | | | |
| 自适应 | loo_mask | | | | |
| 自适应 | random_mask | | | | |

附一张 residual cue 表：每个类型 occlusion 归因 top-20 上下文词。

### 5.2 方案 B 主表

| 攻击者 | 测试条件 | minor AUROC | gender AUROC | kinship AUROC | 平均 p（正类） | 攻击成功率 |
| --- | --- | --- | --- | --- | --- | --- |
| A_raw | raw | | | | | |
| A_raw | gold_mask | | | | | |
| A_raw | random_mask | | | | | |
| A_adapt | gold_mask | | | | | |

### 5.3 解读框架

- raw 到 gold_mask 的下降 = 遮盖带来的保护。
- gold_mask 到 detector_mask 的差距 = 检测器漏检的代价（detector_mask 补跑后再填）。
- random_mask 接近 raw 而 gold_mask 明显下降 = 保护来自遮对了词，不是删字本身。
- A_adapt 在 gold_mask 上仍高 = 残留线索泄露，用归因表指出是哪些词。

## 6. 实现计划

目录放在 `benchmark/threat/`。**单独建 `.venv-threat`**：NeoBERT 和 T5Gemma 都需要比 4.45.2 新的 transformers，而 span_marker 要求低于 4.46，两者不能共存。新环境装最新 transformers、torch cu121、datasets、accelerate、scikit-learn、Captum；NeoBERT 的 remote code 可能额外依赖 xformers，smoke test 时确认。T5Gemma 是 Gemma 许可，HF 上要先接受条款并 `huggingface-cli login`。

| 脚本 | 职责 |
| --- | --- |
| `mask.py` | 偏移校验；生成五个遮盖版本；输出 train/dev/test 的各版本 CSV |
| `lexicon.py` | 从训练池 span 生成五类型词表；归一化函数 |
| `recon_t5.py` | 方案 A：构造哨兵输入输出；非自适应推理；自适应微调；generate 和 teacher forcing 打分 |
| `presence_roberta.py` | 方案 B：标签构造；A_raw 和 A_adapt 训练；推理输出概率 |
| `metrics.py` | 两个方案的全部指标；输出主表 |
| `attribution.py` | 方案 A 的 occlusion；方案 B 的 integrated gradients；输出 residual cue 表 |

执行顺序：

0. 建 `.venv-threat`（已建；torch 需 2.6 以上才能被 transformers 5.x 识别，固定 2.7.1 cu126）；接受 Gemma 条款并登录 HF；smoke test：20 条样本上 T5Gemma 零训练填空看输出格式，NeoBERT 跑一个 batch 前向确认显存和依赖。
1. `mask.py` 跑偏移校验，先看有多少行编码出问题。
2. 方案 A 非自适应：零训练，直接在 400 行上跑，看数字量级。
3. 方案 B A_raw 训练和测试。
4. 方案 A 自适应微调，方案 B A_adapt 训练。
5. 归因分析。
6. 三个种子复跑，汇总表。

算力估计：目标机器是 RTX 3060 Laptop 6 GB。T5Gemma 2 270M-270M 微调约 15 到 20 分钟一轮，NeoBERT 约 30 分钟一轮，三个种子全部跑完一天以内。

**模型选型汇总**

| 实验 | 角色 | 模型 | 发布 | 参数量 | 6 GB 训练 |
| --- | --- | --- | --- | --- | --- |
| A | 主模型，两个攻击者共用 | google/t5gemma-2-270m-270m | 2025-12 | 约 370M | 可以，batch 4 到 8 加累积 |
| A | 备选 | google/t5gemma-b-b-ul2 | 2025-07 | 约 T5-base 量级 | 可以 |
| A | fallback | t5-base | 2019-10 | 220M | 可以 |
| B | 主模型 | chandar-lab/NeoBERT | 2025-02 | 250M | 可以 |
| B | fallback | roberta-base | 2019-07 | 125M | 可以 |
| B | 基线和可解释性 | TF-IDF + logistic regression | 无 | 无 | 秒级 |

决定：先只跑 2025 年的模型，老模型作为 fallback 暂不跑；TF-IDF 基线保留，因为它零成本且直接产出 residual cue 词表。

两个预训练模型都在大规模网页文本上预训练，对应论文 3.1 节"具备广泛世界知识的被动观察者"；在本语料上微调对应"针对该 schema 做过准备的攻击者"。选 2025 年的模型是为了让攻击者尽量强，避免低估风险；两者都能在一张笔记本 GPU 上复现，支撑"威胁是现实的"这一论点。

## 7. 风险与注意事项

- **标签同源。** 方案 B 的标签从 span 类型读出，A_raw 在原文上的高分有循环成分。论文里明说它是泄露上限，不是独立估计；真正干净的数字是 A_adapt。
- **占位符泄露。** 方案 B 必须用删除做主条件；方案 A 的哨兵不泄露长度和类型，但泄露"这里删过"，这对应真实部署，是保守的、对防御不利的设定，适合当主结果。
- **编码问题。** 偏移校验不通过的行要么修要么剔除，剔除数写进论文。
- **稀疏类型。** MINOR_EDU 和 GEN_PHYS 在测试集里分别只有 10 和 22 个 span，方案 A 按类型报时标注 indicative only，和论文 Table 9 的处理方式一致。
- **测试集是富集的。** 400 行里 75% 有实体，AUROC 不受影响，但 precision 类指标不能当真实世界估计，沿用论文已有的 limitation 声明。
- **论文写法。** 两个方案放在 6.2 节，开头一句话说明攻击者是在本语料上训练的 ML 模型而非 LLM，理由是可复现、零 API 成本、可解释；再用一句话说明 LLM 攻击者（Staab et al. 协议）可作为补充，不是替代。

## 8. 实现备注与结果状态（2026-11-02 更新）

单种子（42）的全部实验已跑完，汇总在 `benchmark/threat/results/summary.md`，按运行目录保存的逐 span 预测可直接做错误分析。

**方案 B 结果有效。** NeoBERT 原文训练在原文 / gold 删除 / 随机等量删除上的 macro AUROC 为 0.940 / 0.719 / 0.939，三类正例攻击成功率从 0.80 / 0.61 / 0.89 降到 0.20 / 0.09 / 0.30；NeoBERT 自适应在 gold 删除文本上为 0.759，低于 TF-IDF 自适应的 0.782（训练日志显示第 1 轮后过拟合，残留线索是 my / our / she / he 这类功能词，n-gram 模型更占优）。

**方案 A 第一轮 T5Gemma 2 结果无效，已移到 `results/invalid_bf16/`。** 原因是模型以 bf16 存储权重直接训练，Adafactor 在 lr 2e-4 下的更新量落在 bf16 分辨率附近，模型只学到输出先验：不同评论的 5 条 beam 完全一样，换上下文后 gold span 概率几乎不变（0.193 对 0.210）。权重加载、编码器双向掩码、解码起始 token 均已核对无误。修复方式是 `recon_t5.py --fp32`（可训练权重 fp32，计算仍 bf16 autocast），由 `run_recon_fp32.sh` 重跑。

**fp32 重跑后 T5Gemma 2 仍只比先验好一点。** 自适应 gold 条件 exact@1 0.108、exact@5 0.307、cat@1 0.497，而"永远猜训练集最高频跨度"的无上下文先验（`prior_baseline.py`）为 0.100 / 0.299 / 0.413；模型只学会了代词性别（he → son，she → daughter），"married for 15 years" 仍答 daughter。非自适应版连随机词填空都只有 exact@1 0.026。判断：在 `<unusedN>` 哨兵格式和 4.7k 训练对的条件下，T5Gemma 2 270M 不是合格的重建攻击者，不宜用它的数字支撑 6.2 节结论。

**t5-base 对照是可用的攻击者**（`run_recon_t5base.sh`，fp32，lr 3e-4，单次 15 分钟）。自适应 gold 条件 exact@1 0.263、exact@5 0.469、cat@1 0.673（FAM_KIN 0.391，GEN_NOUN 0.219，MINOR_AGE 0.157），留一条件 exact@1 0.281，随机词遮盖 exact@1 0.417；非自适应 gold 0.078、留一 0.170。换上下文检验的对数比为 3.29（T5Gemma 2 为 0.20），手工提示 "married … She" → wife、"45 year old ___" → woman / man。建议论文 6.2 节的方案 A 以 t5-base（或更大的 T5 / FLAN-T5）为主攻击者，T5Gemma 2 作为负面结果一句带过或省略；若坚持 Gemma 系列，需先改任务格式（自然语言遮盖标记）并用条件化检验确认后再跑全量。

**新增文件。** `recon_t5.py` 增加 `--fp32 / --optim / --tag`；`run_recon_fp32.sh`、`run_recon_t5base.sh`（可排队等待前一链结束）；`prior_baseline.py` 产出 `results/prior_baseline.json`，`summarize.py` 自动附上先验参照行。条件化检验脚本在会话 scratchpad（`recon_diag.py`）：64 个留一 span 的真实上下文对换上下文 gold 概率，外加 7 条手工提示。

**尚未做。** detector_mask 条件（等 SpanMarker 预测）、多种子、方案 B 的 NeoBERT 归因分析（目前只有 TF-IDF 系数词表）。

## 参考

- SANTEXT: Differential Privacy for Text Analytics via Natural Text Sanitization, arXiv 2106.01221
- InferDPT, arXiv 2310.12214
- Truthful Text Sanitization Guided by Inference Attacks, arXiv 2412.12928
- Coavoux, Narayan & Cohen 2018, Privacy-preserving Neural Representations of Text, EMNLP
- Elazar & Goldberg 2018, Adversarial Removal of Demographic Attributes from Text Data, EMNLP
- Manzanares-Salor, Sánchez & Lison 2024, Evaluating the disclosure risk of anonymized documents via a machine learning-based re-identification attack, DMKD
- Staab et al. 2025, Language Models are Advanced Anonymizers, ICLR
