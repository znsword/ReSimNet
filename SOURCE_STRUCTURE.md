# ReSimNet 源码结构分析

- 项目目标：实现基于 Siamese 神经网络的药物转录响应相似度预测（CMap 分数），学习药物在响应空间中的嵌入，并支持单模型与集成预测。
- 技术栈：`Python 3`、`PyTorch 0.3.0`、`NumPy`，并包含少量可视化与脚本工具。

## 目录结构

- `models/`
  - `drug_model.py`：核心模型定义（Siamese/MLP/LSTM/图卷积分支）。
  - `root/`：在当前仓库中为占位项；`main.py` 依赖 `models.root.utils`，但缺少实际 `utils.py` 文件，参见“潜在问题”。
- `tasks/`
  - `drug_task.py`：数据集与数据加载、表示类型管理、批采样器与 `collate_fn`。
  - `drug_run.py`：训练/验证/测试循环（回归与二分类）、评估与结果保存、对新对儿的打分。
  - `plot.py`、`run_plot.py`：TSNE 降维与分类/聚类绘图工具。
  - `tasks/.ipynb_checkpoints/`：Jupyter 运行缓存。
- 顶层脚本与文件
  - `main.py`：程序入口与 CLI 参数、模型与数据装配、完整运行流程。
  - `utils.py`：通用工具（性能分析、进度条），但入口引用指向 `models.root.utils`。
  - `train.sh`、`train_ensemble.sh`、`predict*.sh`、`test.sh`：常用运行脚本。
  - `images/`：流程图等图片资源。
  - `README.md`、`LICENSE`、`.gitignore`：文档与许可。

## 核心模块

- 模型定义（`models/drug_model.py:18`）

  - 结构选择：
    - 图卷积（GraphConv）分支：`graph_conv`（`models/drug_model.py:156`），4 层线性+邻接传播，`MaxPool1d` 聚合，输出药物嵌入。
    - 序列分支：SMILES/InChI 的 LSTM 编码，`siamese_sequence`（`models/drug_model.py:121`）。
    - MLP 分支：ECFP/Mol2vec 等定长向量的前馈编码，`siamese_basic`（`models/drug_model.py:191`）。
  - 距离层：`distance_layer`（`models/drug_model.py:194`）支持 `cos/l1/l2`，二分类时应用 `sigmoid`。
  - 前向逻辑：根据输入表示自动选择分支，返回 `(similarity, embed1, embed2)`（`models/drug_model.py:210`）。
  - 损失与优化：MSE 或二分类自定义对数似然，Adam 优化器（`models/drug_model.py:82`、`models/drug_model.py:226`）。
  - 断点：保存/加载检查点（`models/drug_model.py:255`、`models/drug_model.py:260`）。
- 数据集与数据加载（`tasks/drug_task.py:18`）

  - 原始字典构建与字符词表：`process_drug_id`（SMILES/InChI；`tasks/drug_task.py:68`），记录最大长度与字符映射。
  - 药物子表示拼接：指纹、Mol2vec、图特征等（`append_drug_sub`，`tasks/drug_task.py:136`）。
  - 细胞系拆分：按 `MCF7/PC3` 等细胞线划分 `tr/va/te`（`process_cell_lines`，`tasks/drug_task.py:108`）。
  - 表示类型选择：`rep_idx` → `0:SMILES`，`1:InChIKey`，`2:ECFP`，`3:Mol2vec`，`4:图表示`（`tasks/drug_task.py:616`）。
  - DataLoader：
    - 文本/向量批处理 `collate_fn`（`tasks/drug_task.py:298`、同名函数在 `tasks/drug_task.py:447`）。
    - 图批处理 `collate_fn_graph`（`tasks/drug_task.py:501`）。
    - 排序批采样器：`SortedBatchSampler`（`tasks/drug_task.py:677`）。
    - 细胞线专用 Loader：`get_cellloader`（`tasks/drug_task.py:244`）。
  - 样本类：`Representation`（定长/序列；`tasks/drug_task.py:631`）、`Rep_graph`（图；`tasks/drug_task.py:699`）。
- 训练与评估（`tasks/drug_run.py`）

  - 二分类循环：`run_bi`（`tasks/drug_run.py:28`），按 KK/KU/UU 三类拆分评估，`precision_recall_fscore_support` 计算 F1。
  - 回归循环：`run_reg`（`tasks/drug_run.py:180`），计算整体与 KK/KU/UU 的皮尔逊相关、MSE@K、P@K/ROC（`evaluation`，`tasks/drug_run.py:375`）。
  - 结果产出：
    - 导出药物嵌入：`save_embed`（`tasks/drug_run.py:398`）。
    - 测试集预测 CSV：`save_prediction`（`tasks/drug_run.py:457`）。
    - 新对儿打分：`save_pair_score`（`tasks/drug_run.py:539`）与 ZINC 特例（`tasks/drug_run.py:607`）。
    - 集成推断：`perform_ensemble`（`tasks/drug_run.py:482`）。
- 程序入口与运行（`main.py`）

  - CLI 参数：数据/模型/训练配置/图模型参数（`main.py:47`–`main.py:131`）。
  - 模型构造：依据 `rep_idx` 切换图/序列/MLP（`get_model`，`main.py:278`）。
  - 运行流程：
    - `run_experiment` 装配 DataLoader、选择回归或二分类指标、训练-验证-保存最优-测试（`main.py:134`）。
    - 支持保存嵌入/预测、对 ZINC/自定义对儿评分、集成评估（`main.py:154`、`main.py:170`、`main.py:183`、`main.py:386`）。
  - `main`：入口，区分验证、集成、保存分数等路径（`main.py:362`）。
- 工具与可视化

  - 通用工具：进度条与性能分析装饰器（`utils.py:11`、`utils.py:41`、`utils.py:63`）。
  - 可视化：TSNE 降维与分类/聚类绘图（`tasks/plot.py:134`、`tasks/plot.py:65`、`tasks/plot.py:213`、`tasks/plot.py:164`）；任务驱动入口（`tasks/run_plot.py:15`）。

## 运行脚本

- 训练单模型：`train.sh` 调用 `python main.py --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'trained_model.mdl' --rep-idx 2`（`train.sh:4`）。
- 集成训练：`train_ensemble.sh` 触发 `--perform-ensemble`（`train_ensemble.sh:4`）。
- 预测示例/通用/对 ZINC：`predict_example.sh`、`predict.sh`、`predict_zinc.sh`（分别见 `predict_example.sh:5`、`predict.sh:5`、`predict_zinc.sh:5`）。
- 保存测试集预测：`test.sh`（`test.sh:5`）。

## 数据与依赖

- 关键数据与模型
  - 训练数据：`ReSimNet-Dataset.pkl`（`README.md:40`–`README.md:43`）。
  - 指纹映射：`pertid2fingerprint.pkl`（`README.md:56`–`README.md:58`）。
  - 预训练/集成模型 zip：见 `README.md:44`–`README.md:51`。
  - 示例对儿与 ZINC 文件：`README.md:52`–`README.md:118`。
- 依赖版本
  - `PyTorch 0.3.0`、`CUDA 8.0`、`cuDNN v5.1`、`NumPy`（`README.md:14`–`README.md:19`）。

## 参数要点

- 表示选择 `--rep-idx`（`main.py:121`；详见 `tasks/drug_task.py:616`）：
  - `0` SMILES（序列/LSTM），`1` InChIKey（序列/LSTM），`2` ECFP（MLP），`3` Mol2vec（MLP），`4` 图（GraphConv）。
- 距离函数 `--dist-fn`：`cos/l1/l2`（`main.py:122`；实现于 `models/drug_model.py:194`）。
- 二分类开关 `--binary`：影响损失与输出激活（`main.py:111`、`models/drug_model.py:88`）。
- 学习率、嵌入维度、图层参数等：`main.py:103`–`main.py:130`。

## 典型工作流

- 训练→验证→模型保存→测试：`main.py:210`–`main.py:265`。
- 导出药物嵌入：`--save-embed`（`main.py:154`，`tasks/drug_run.py:398`）。
- 对新对儿打分：`--save-pair-score`（`main.py:183`，`tasks/drug_run.py:539`）；ZINC 特例 `--save-pair-score-zinc`（`tasks/drug_run.py:607`）。
- 集成评估：`--perform-ensemble`（`main.py:386`、`tasks/drug_run.py:482`）。

## 潜在问题与改进建议

- `models.root.utils` 导入问题：入口处引用 `from models.root.utils import *`（`main.py:20`），但当前仓库不含 `models/root/utils.py`。建议：
  - 若实用工具仅来自顶层 `utils.py`，将引用改为 `from utils import *`；或将 `utils.py` 移入 `models/root/` 并设为包。
- 旧版 PyTorch API：广泛使用 `Variable` 与 `F.sigmoid`（如 `models/drug_model.py:206`），在新版中已弃用；若迁移到新版本，需用 `tensor.requires_grad` 与 `torch.sigmoid` 替换，并检查 `pack_padded_sequence` 的 `enforce_sorted` 参数。
- 图分支的归一化与填充：`collate_fn_graph` 对邻接矩阵直接加自环并零填充（`tasks/drug_task.py:524`–`tasks/drug_task.py:547`），可考虑标准化与批内对齐策略以提升稳定性。

---

以上为当前源码结构与职责的系统性分析，包含关键函数/文件位置以便快速定位与理解。若需要，我可以补充调用序列图或将潜在问题的修正以补丁形式提交。

# 模型定义详解（`models/drug_model.py`）

## 设计总览

- 单模型以“共享编码 + 距离层”实现 Siamese 结构，支持 3 种输入表示：序列（SMILES/InChI）、定长向量（ECFP/Mol2vec）、图（节点特征+邻接）。
- 统一输出为药物嵌入向量（维度 `drug_embed_dim`）与两向量间的相似度/距离分数。

## 构造参数与持久化配置（`models/drug_model.py:18`）

- 主要参数：
  - `input_dim`、`hidden_dim`、`drug_embed_dim`：输入维度、隐藏层维度、药物嵌入维度。
  - 序列相关：`lstm_layer`、`lstm_dropout`、`char_vocab_size`、`char_embed_dim`、`char_dropout`、`bi_lstm`（当前代码为单向）。
  - 图相关：`is_graph`、`g_layer`、`g_hidden_dim`、`g_out_dim`、`g_dropout`。
  - MLP 切换：`is_mlp`（当 `rep_idx>1` 时在入口设为 True，见 `main.py:304`）。
  - 任务相关：`dist_fn`（`cos/l1/l2`）、`binary`（二分类或回归）、`learning_rate`、`weight_decay`。
- 优化器与损失：Adam（`models/drug_model.py:82`），MSE 或二分类对数似然（`models/drug_model.py:88`–`models/drug_model.py:93`、`models/drug_model.py:226`）。

## 表示模式与分支选择（入口联动）

- 入口在构建模型时依据 `rep_idx` 决定分支（`main.py:278`）：
  - `rep_idx==4` → 图分支（`is_graph=True`、`is_mlp=False`）。
  - `rep_idx<=1` → 序列分支（`is_mlp=False`）。
  - `rep_idx in {2,3}` → MLP 分支（`is_mlp=True`）。
- 前向根据是否提供邻接矩阵决定分支（`models/drug_model.py:210`）。

## 图卷积分支（`models/drug_model.py:38`、`models/drug_model.py:95`、`models/drug_model.py:156`）

- 权重与偏置：`weight1..4`、`bias1..4`，输入特征维设为 `feature_dim=75`（`models/drug_model.py:40`）。
- 初始化：均匀分布，尺度为前一维度平方根倒数（`models/drug_model.py:95`–`models/drug_model.py:107`）。
- 传播层：
  - 每层计算 `support = X W`，再做 `A support` 的一阶邻接传播（`models/drug_model.py:158`–`models/drug_model.py:179`）。
  - 激活与丢弃：`ReLU` 后 `F.dropout`（`models/drug_model.py:160`、`models/drug_model.py:167`、`models/drug_model.py:174`）。
  - 最后一层 `layer4_out = A support4 + bias4` 后做 `F.log_softmax`（`models/drug_model.py:180`–`models/drug_model.py:183`）。
- 池化输出：对 `log_softmax` 结果做 `MaxPool1d(num_nodes)`，得到图级嵌入（`models/drug_model.py:185`–`models/drug_model.py:188`）。
- 输入要求：`features` 形状 `[B, N, F]`，`adjs` 形状 `[B, N, N]`；任务侧会自动加自环与零填充（`tasks/drug_task.py:524`–`tasks/drug_task.py:547`）。

## 序列/LSTM 分支（`models/drug_model.py:121`）

- 嵌入层：`nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)`（`models/drug_model.py:60`）。
- 训练/评估差异：
  - 评估时按照长度降序排序并 `pack_padded_sequence`（`models/drug_model.py:127`–`models/drug_model.py:137`），以提升效率；训练时直接送入 LSTM（`models/drug_model.py:138`–`models/drug_model.py:144`）。
- LSTM 输出：以隐藏状态 `states[0]`（h）为药物嵌入，转置并 `view(-1, drug_embed_dim)`（`models/drug_model.py:145`–`models/drug_model.py:147`）；评估时再按原顺序 `unsort`（`models/drug_model.py:148`–`models/drug_model.py:151`）。
- 初始状态：零张量（`models/drug_model.py:109`–`models/drug_model.py:114`）。
- 输入要求：整数索引序列与长度张量；任务侧在 `collate_fn` 做填充与转 Long（`tasks/drug_task.py:329`–`tasks/drug_task.py:333`）。

## MLP 分支（`models/drug_model.py:66`）

- 结构：`Linear(input_dim→hidden_dim) + ReLU + Linear(hidden_dim→drug_embed_dim)`（`models/drug_model.py:67`–`models/drug_model.py:75`）。
- 输入类型：浮点向量（ECFP、Mol2vec 等）；任务侧若 `rep_idx==3` 保持 Float，否则 Long 并在模型中 `Float` 化（`tasks/drug_run.py:423`–`tasks/drug_run.py:426`、`models/drug_model.py:192`）。

## 距离层（`models/drug_model.py:194`）

- `cos`：`F.cosine_similarity(vec1, vec2, dim=-1)`（默认）。
- `l1`/`l2`：经 `nn.Linear(drug_embed_dim, 1)` 的投影后 `squeeze(1)` 输出标量（`models/drug_model.py:199`–`models/drug_model.py:203`）。
- 二分类：对相似度输出做 `sigmoid`（`models/drug_model.py:205`–`models/drug_model.py:207`）。

## 前向流程与返回（`models/drug_model.py:210`）

- 路径选择：
  - 若同时提供 `key1_adj/key2_adj` → 图分支。
  - 否则若 `not is_mlp and not is_graph` → 序列分支。
  - 否则 → MLP 分支。
- 返回值：`(similarity, embed1, embed2)`，其中 `similarity` 为 `[B]` 或 `[B]` 标量，`embedX` 为 `[B, drug_embed_dim]`。

## 损失与优化（`models/drug_model.py:82`、`models/drug_model.py:226`）

- 回归：`nn.MSELoss()`（`models/drug_model.py:91`–`models/drug_model.py:93`）。
- 二分类：自定义对数似然 `y*log(x)+(1-y)*log(1-x)`，在 `get_loss` 原样返回（`models/drug_model.py:88`–`models/drug_model.py:90`、`models/drug_model.py:231`–`models/drug_model.py:236`）。
- 优化器：Adam，支持 `weight_decay`（`models/drug_model.py:82`–`models/drug_model.py:84`）。
- 梯度裁剪：在训练循环中执行（`tasks/drug_run.py:95`、`tasks/drug_run.py:254`）。

## 断点保存与加载（`models/drug_model.py:255`、`models/drug_model.py:260`）

- 保存内容：`{'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}`（调用方在 `main.py:231`）。
- 加载：按同名键恢复（`models/drug_model.py:265`–`models/drug_model.py:266`）。

## 张量形状与类型要点

- 序列：输入为 `LongTensor` 的索引序列，嵌入输出 `[B, L, char_embed_dim]`，LSTM hidden 整合为 `[B, drug_embed_dim]`。
- MLP：输入为 `FloatTensor [B, input_dim]`，输出嵌入 `[B, drug_embed_dim]`。
- 图：
  - `features [B, N, F]`、`adjs [B, N, N]`，各层保持批维度与节点维。
  - 池化后图嵌入为 `[B, g_out_dim]`（与 `drug_embed_dim` 对齐）。

## 训练/评估模式差异

- 排序与打包：评估模式下对变长序列排序与 `pack_padded_sequence`，训练模式直接输入（`models/drug_model.py:127`–`models/drug_model.py:144`）。
- Dropout：在图分支中由 `self.g_dropout` 控制（`models/drug_model.py:160`、`models/drug_model.py:167`、`models/drug_model.py:174`）。

## 关键注意点与迁移建议

- 旧版 API：
  - `Variable` 广泛使用（如 `models/drug_model.py:110`、`models/drug_model.py:134`），迁移到新版 PyTorch 时可直接使用张量并设置 `requires_grad`。
  - `F.sigmoid` 已弃用，建议替换为 `torch.sigmoid`（`models/drug_model.py:206`）。
  - `F.log_softmax` 在新版本需指定 `dim` 参数。
- 入口依赖：`main.py` 构造模型时的 `rep_idx` 与 `is_mlp/is_graph` 映射（`main.py:278`–`main.py:325`）决定编码分支。
- 图输入准备：任务侧对邻接矩阵加自环与零填充，必要时可引入归一化（`tasks/drug_task.py:524`–`tasks/drug_task.py:547`）。

---

# main.py 文件详解（`main.py`）

## 常量与默认路径（`main.py:25`–`main.py:40`）

- `DATA_PATH`：训练/验证所用数据集默认路径（示例为 `./tasks/data/drug(v0.6).pkl`）。
- `DRUG_DIR`、`DRUG_FILES`：用于导出嵌入或验证的药物字典/文件列表。
- `PAIR_DIR`、`FP_DIR`、`EXAMPLE_DIR`：新药物对评分目录、指纹映射与 ZINC 示例对。
- `CKPT_DIR`、`MODEL_NAME`：模型检查点与默认模型名。

## 命令行参数（`main.py:47`–`main.py:131`）

- 数据与输出：`--data-path`、`--drug-dir`、`--drug-files`、`--pair-dir`、`--fp-dir`、`--example-dir`、`--checkpoint-dir`、`--model-name`。
- 运行开关：`--train`、`--pretrain`、`--valid`、`--test`、`--resume`、`--debug`、`--save-embed`、`--save-prediction`、`--perform-ensemble`、`--save-pair-score`、`--save-pair-score-zinc`、`--save-pair-score-ensemble`、`--top-only`、`--embed-d`。
- 训练配置：`--batch-size`、`--epoch`、`--learning-rate`、`--weight-decay`、`--grad-max-norm`、`--grad-clip`。
- 模型配置：`--binary`、`--hidden-dim`、`--drug-embed-dim`、`--lstm-layer`、`--lstm-dr`、`--char-dr`、`--bi-lstm`、`--linear-dr`、`--char-embed-dim`、`--s-idx`、`--rep-idx`、`--dist-fn`、`--seed`。
- 图模型：`--g_layer`、`--g_hidden_dim`、`--g_out_dim`、`--g_dropout`。
- 布尔解析：自定义 `str2bool`（`main.py:42`–`main.py:44`）并通过 `argparser.register('type', 'bool', str2bool)` 支持布尔参数解析（`main.py:49`）。

## 核心函数

- `run_experiment`（`main.py:134`）

  - Loader 装配：根据 `cell_line` 选择全体或特定细胞线的 `train/valid/test`（`main.py:137`–`main.py:144`）。
  - 指标设置：二分类用 `precision_recall_fscore_support`，回归用 `np.corrcoef`（`main.py:146`–`main.py:153`）。
  - 特殊模式：
    - 嵌入导出：加载检查点后遍历字典并保存药物嵌入（`main.py:154`–`main.py:167`；实现见 `tasks/drug_run.py:398`）。
    - 测试集预测导出（`main.py:170`–`main.py:175`；`tasks/drug_run.py:457`）。
    - 集成返回预测集合用于后续汇总（`main.py:176`–`main.py:180`）。
    - 对新对儿打分：单模型或集成，支持 ZINC（`main.py:183`–`main.py:207`；实现见 `tasks/drug_run.py:539`、`tasks/drug_run.py:607`）。
  - 训练-验证-保存最优-自适应降学习率-早停：
    - 训练一个 epoch（`main.py:221`–`main.py:223`）。
    - 验证并在性能提升时保存检查点（`main.py:225`–`main.py:235`）。
    - 连续收敛计数触发 `lr *= 0.5`（`main.py:247`–`main.py:254`），超过阈值早停（`main.py:256`–`main.py:257`）。
  - 最终测试：在有效集与测试集上运行并打印（`main.py:259`–`main.py:265`）。
- `get_dataset`（`main.py:267`）：从 `--data-path` 反序列化数据集（pickle）。
- `get_run_fn`（`main.py:271`）：依据 `--binary` 返回回归或二分类循环函数（`tasks/drug_run.py:180`/`tasks/drug_run.py:28`）。
- `get_model`（`main.py:278`）：

  - 根据 `rep_idx` 构造图/序列/MLP 分支的 `DrugModel` 并迁移到 GPU。
  - 当 `rep_idx==4`：图分支；否则根据 `rep_idx>1` 选择 `is_mlp`（`main.py:304`）。
  - 参数映射：包括 `char_vocab_size=len(dataset.char2idx)` 与图分支的 `g_*` 参数（`main.py:281`–`main.py:324`）。
- `init_logging`（`main.py:328`）：配置控制台与文件日志（`<checkpoint_dir>/logs/<model_name>.txt`）。
- `init_seed`（`main.py:343`）：打印并设置 `numpy/torch/random` 种子。
- `init_parameters`（`main.py:353`）：按 `<cell>-<name>-<idx>` 格式重写 `args.model_name`（当前仅示例注释）。

## 主函数流程 `main()`（`main.py:362`）

- 初始化日志与打印命令（`main.py:365`–`main.py:367`）。
- 载入数据集、选择运行函数、设置 `cell_line=None`（`main.py:369`–`main.py:372`）。
- 分支：
  - `--save_pair_score`：加载模型后执行对对儿打分（支持集成与 ZINC，`main.py:373`–`main.py:207`）。
  - `--perform_ensemble`：循环加载不同模型检查点，收集预测，计算 Ensemble 指标（`main.py:386`–`main.py:436`）。
  - 否则执行验证：按 `--validation-step` 次数进行训练/验证流程（`main.py:439`–`main.py:451`）。

## 关键注意点与潜在问题

- 导入路径：`from models.root.utils import *`（`main.py:20`）与仓库实际结构不符；当前工具位于顶层 `utils.py`。建议改为 `from utils import *` 或调整目录结构。
- 集成分支未定义变量：`print(model_name, _model)`（`main.py:398`）引用 `_model` 未定义，可能导致运行错误；应改为使用 `model_idx` 或移除此行。
- 旧版 PyTorch 语义：入口与循环依赖 `Variable` 的下游实现，在迁移新版本时需同步替换。

---

toolName: view_folder

status: success

d:\document\coding\python\ReSimNet\ReSimNet


**Shell 脚本**

- `train.sh`
  - 训练单模型（表示为 ECFP，`--rep-idx 2`），数据 `./tasks/data/ReSimNet-Dataset.pkl`，模型名 `trained_model.mdl`（`train.sh:4`）。
- `train_ensemble.sh`
  - 触发集成流程（`--perform-ensemble True`），对同一数据执行多模型评估与汇总，模型前缀 `trained_model.mdl`（`train_ensemble.sh:4`）。
- `predict.sh`
  - 对成对药物 ID 计算相似度分数并保存，输入对目录 `./tasks/data/pairs/`，指纹映射 `pertid2fingerprint.pkl`，模型 `ReSimNet7.mdl`（`predict.sh:5`）。
- `predict_example.sh`
  - 与 `predict.sh` 类似，但启用 10 模型集成平均（`--save-pair-score-ensemble true`），模型前缀 `ReSimNet.mdl`（`predict_example.sh:5`）。
- `predict_zinc.sh`
  - 计算 ZINC 测试集与示例药物对的相似度分数并保存，目录 `./tasks/data/pairs_zinc/zinc-test/` 与 `example_drugs.csv`（`predict_zinc.sh:5`）。
- `test.sh`
  - 导出验证/测试集预测结果到 CSV（`--save-prediction true`），模型 `ReSimNet7.mdl`，数据 `ReSimNet-Dataset.pkl`（`test.sh:5`）。

**Python 可执行脚本**

- `main.py`

  - 程序入口与运行总控：解析 CLI 参数，装配数据与模型，执行训练/验证/测试，支持导出嵌入与预测、对新药物对打分、模型集成；核心流程函数 `run_experiment`（`main.py:134`）、模型构造 `get_model`（`main.py:278`）。
- `models/drug_model.py`

  - 核心模型定义（Siamese 编码器）：序列 LSTM、MLP、图卷积分支三选一；距离/相似度层（`models/drug_model.py:194`）；损失与优化器配置；checkpoint 保存/加载（`models/drug_model.py:255`、`models/drug_model.py:260`）。
- `tasks/drug_task.py`

  - 数据集与加载：构建药物字典与字符词表（`tasks/drug_task.py:68`）、拼接各类子表示（指纹、Mol2vec、图）（`tasks/drug_task.py:136`）、细胞线数据拆分（`tasks/drug_task.py:108`）；`Representation`/`Rep_graph` 样本类与 `collate_fn`/`collate_fn_graph`；`SortedBatchSampler` 批采样器。
- `tasks/drug_run.py`

  - 训练与评估循环：回归 `run_reg`（`tasks/drug_run.py:180`）与二分类 `run_bi`（`tasks/drug_run.py:28`）；综合指标评估（相关系数、MSE@K、P@K、AUROC）；导出药物嵌入、测试集预测、新对儿得分、集成推断。
- `tasks/plot.py`

  - 可视化工具：TSNE 降维（`tasks/plot.py:134`）、带标签/图例的散点绘制（`tasks/plot.py:164`、`tasks/plot.py:213`），KMeans 辅助聚类上色。
- `tasks/run_plot.py`

  - 绘图任务脚本：加载 FDA/药物信息/嵌入数据，组织多种视图任务（task0–task4），调用 `plot.py` 输出 HTML 图。
- `utils.py`

  - 通用工具：进度条 `progress`（`utils.py:11`）、性能分析装饰器 `profile` 与统计打印（`utils.py:41`、`utils.py:63`），便于开发期性能度量。
- `load_file.py`

  - 数据检查工具：读取 `drug(v0.5).pkl`，从测试集对标注 KK/KU/UU 并保存到 `results/testset.pkl`，用于集分析或验证集划分核对。
- `load_embed.py`

  - 嵌入查看工具：读取 `./results/*.pkl` 的药物嵌入字典并打印首条记录，快速检查导出结果是否正确。
