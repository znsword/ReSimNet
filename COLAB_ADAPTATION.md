# 在 Google Colab 运行 ReSimNet 的改造方案

## 目标
- 在 Colab 上完成环境准备、数据下载、训练、预测与可视化，尽量保持与现有 CLI 流程一致；对旧版 PyTorch 接口进行必要代码适配。
- 输出一个可执行的 `.ipynb`，并保留此 Markdown 指南以便快速复用。

## 环境与依赖
- 运行时：在 Colab 中选择 GPU（Runtime → Change runtime type → GPU）。
- 安装：
  - `pip install torch numpy pandas scikit-learn seaborn plotly gdown`
  - 若启用图分支（`rep_idx=4`），当前实现不依赖 DGL/torch-geometric。

## 代码适配（兼容新版 PyTorch）
- `models/drug_model.py`
  - 替换旧接口：
    - `F.sigmoid` → `torch.sigmoid` 或采用 `BCEWithLogitsLoss`（推荐）。
    - `F.log_softmax(layer4_out)` → `F.log_softmax(layer4_out, dim=-1)`。
    - 移除 `Variable`；读取标量用 `loss.item()`；转 numpy 用 `tensor.detach().cpu().numpy()`。
  - 二分类损失修正：推荐改用 `BCEWithLogitsLoss` 并让 `distance_layer` 在二分类时返回 logits（移除手动 sigmoid）。
  - 设备管理：统一 `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`，替换 `.cuda()` 为 `.to(device)`。
- `tasks/drug_task.py`
  - 评估模式可改用 `pack_padded_sequence(..., enforce_sorted=False)`，减少排序与还原复杂度。
  - 去除 `Variable`；保持 Long/Float 类型转换逻辑。
- `tasks/drug_run.py`
  - `loss.data[0]` → `loss.item()`；`outputs.data.cpu().numpy()` → `outputs.detach().cpu().numpy()`。
  - 修正未定义变量：`perform_ensemble` 中 `print(model_name, _model)` → `print(model_name, model_idx)`。
- `main.py`
  - 导入修正：`from models.root.utils import *` → `from utils import *`。
  - 统一设备搬移：使用 `.to(device)`。

## Notebook 结构（建议单元）
1. 环境初始化
   - 选择 GPU；执行：
     - `!pip install torch numpy pandas scikit-learn seaborn plotly gdown`
2. 克隆/导入工程
   - 选项 A：`!git clone <你的仓库地址> && cd ReSimNet`
   - 选项 B：将当前工程压缩上传或通过 Drive 同步到 `/content/ReSimNet/`
3. 数据准备
   - 下载并放置到 `./tasks/data/`：
     - `ReSimNet-Dataset.pkl`、`pertid2fingerprint.pkl`、`pairs/examples.csv`、`pairs_zinc/zinc-test.zip`、`pairs_zinc/example_drugs.csv`
   - 示例：
     - `!gdown <ReSimNet-Dataset.pkl 的链接>`
     - `!mkdir -p tasks/data && mv ReSimNet-Dataset.pkl tasks/data/`
   - 可选挂载 Drive：
     - `from google.colab import drive; drive.mount('/content/drive')`
4. 代码适配补丁
   - 用 Python 文本替换或 `sed` 在 Notebook 中执行上述改动（保持最小侵入）。
5. 训练/验证
   - 单模型：
     - `!python main.py --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'colab_model.mdl' --rep-idx 2`
   - 集成：
     - `!python main.py --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'colab_model.mdl' --rep-idx 2 --perform-ensemble True`
6. 预测与导出
   - 测试集预测：
     - `!python main.py --save-prediction true --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'colab_model.mdl' --rep-idx 2`
   - 新对儿打分：
     - `!python main.py --save-pair-score true --pair-dir './tasks/data/pairs/' --fp-dir './tasks/data/pertid2fingerprint.pkl' --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'colab_model.mdl' --rep-idx 2`
   - ZINC 评分：
     - `!python main.py --save-pair-score true --save-pair-score-zinc true --pair-dir './tasks/data/pairs_zinc/zinc-test/' --example-dir './tasks/data/pairs_zinc/example_drugs.csv' --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'colab_model.mdl' --rep-idx 2`
7. 可视化
   - 导出嵌入（`--save-embed`）后，Notebook 内用 TSNE（`sklearn.manifold.TSNE`）与 Plotly 绘制交互图；或直接调用 `tasks/run_plot.py`。
8. 检查点与持久化
   - 模型与日志路径：`./results/`；可拷贝到 Drive 以持久化：
     - `!cp -r results /content/drive/MyDrive/ReSimNet_results`

## 路径规范
- 项目根：`/content/ReSimNet/`
- 数据：`/content/ReSimNet/tasks/data/`
- 结果：`/content/ReSimNet/results/`

## 风险与注意事项
- 不建议在 Colab 安装极旧的 `torch==0.3.0`；采用新版接口适配更稳妥。
- 二分类损失与距离层需同步调整（避免重复 sigmoid）。
- 数据下载链接需可访问，必要时将数据预先放入 Drive 并在 Notebook 中复制。

## 交付物
- `ReSimNet_Colab.ipynb`：包含上述单元、可直接运行。
- 本文件 `COLAB_ADAPTATION.md`：改造与运行指南。

## 关键代码位置引用
- `models/drug_model.py:88`–`models/drug_model.py:93`（损失）
- `models/drug_model.py:194`–`models/drug_model.py:207`（距离层）
- `models/drug_model.py:183`（log_softmax 需要 `dim`）
- `tasks/drug_run.py:95`、`tasks/drug_run.py:254`（梯度裁剪）
- `tasks/drug_run.py:457`（预测保存）
- `main.py:20`（导入路径修正）
- `main.py:386`–`main.py:436`（集成逻辑，修正 `_model`）