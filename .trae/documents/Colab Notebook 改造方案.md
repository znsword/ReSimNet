# 将工程改写为可在 Google Colab 运行的 Notebook 的方案

## 目标
- 在 Google Colab 上完成环境准备、数据下载、训练、预测与可视化，尽量保持与现有 CLI 流程一致；对旧版 PyTorch 接口进行必要代码适配。
- 输出一个可执行的 `.ipynb`，并在仓库新增一份 Markdown 指南，指导如何在 Colab 运行。

## 环境与依赖
- 选择 Colab GPU 运行时（Runtime → Change runtime type → GPU）。
- 安装依赖：
  - `torch`（使用 Colab 默认新版本）、`numpy`、`pandas`、`scikit-learn`、`seaborn`、`plotly`、`gdown`。
  - 若启用图分支（rep_idx=4），根据需要安装与图构建相关的依赖（当前代码无需 DGL/torch-geometric）。

## 代码适配（兼容新版 PyTorch）
- `models/drug_model.py`
  - 替换旧接口：
    - 将 `F.sigmoid` 替换为 `torch.sigmoid` 或改用 `BCEWithLogitsLoss`（推荐）。
    - 为 `F.log_softmax` 增加 `dim` 参数：`F.log_softmax(x, dim=-1)`。
    - 移除 `Variable` 使用，直接用张量；读取标量改用 `loss.item()`；张量转 numpy 使用 `tensor.detach().cpu().numpy()`。
  - 二分类损失修正：当前 `y*log(x)+(1-y)*log(1-x)`未取负，建议改为 `BCEWithLogitsLoss` 并让 `distance_layer` 在二分类时返回未激活的 logits（移除手动 sigmoid）。
  - 设备管理：用 `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`，替换 `.cuda()` 为 `.to(device)`，并在前向中确保张量在同一设备。
- `tasks/drug_task.py`
  - `pack_padded_sequence`：评估模式目前手动排序；可统一为 `enforce_sorted=False` 以减少排序/还原复杂度（与 `siamese_sequence` 逻辑配合）。
  - `collate_fn` 中 Long/Float 类型转换保持不变；返回的 `Variable` 去除，改为张量；减少 `.data` 访问。
- `tasks/drug_run.py`
  - 将 `loss.data[0]` 改为 `loss.item()`；`outputs.data.cpu().numpy()` 改为 `outputs.detach().cpu().numpy()`。
  - 修复未定义变量：`perform_ensemble` 中 `print(model_name, _model)` 改为 `print(model_name, model_idx)`（`main.py` 中同类问题）。
- `main.py`
  - 修正导入：`from models.root.utils import *` → `from utils import *`（当前仓库无 `models/root/utils.py`）。
  - 统一设备：在模型构建与数据搬移中使用 `.to(device)`。
  - 允许 Notebook 直接调用函数：保留 CLI，但在 Notebook 中也可 `import main` 并调用其中的函数（或使用 `!python main.py ...`）。

## Notebook 结构（分单元）
1. 环境初始化
   - 切换 GPU，安装依赖（`pip install torch numpy pandas scikit-learn seaborn plotly gdown`）。
2. 克隆/导入工程
   - 选项 A：直接 `git clone` 原仓库；选项 B：将当前工程压缩上传至 Colab 或挂载 Google Drive 然后复制到 `/content`。
3. 数据准备
   - 使用 `gdown` 下载 `ReSimNet-Dataset.pkl` 到 `./tasks/data/ReSimNet-Dataset.pkl`（链接见 README），以及 `pertid2fingerprint.pkl`、示例对 `examples.csv`、ZINC 测试包等。
   - 可选：挂载 Drive 以持久化数据与结果（`from google.colab import drive; drive.mount('/content/drive')`）。
4. 代码适配补丁
   - 在 Notebook 中用 Python I/O 或简单 `sed` 风格替换进行上述改动（保持最小侵入）。
5. 训练/验证
   - 运行单模型：`!python main.py --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'colab_model.mdl' --rep-idx 2`。
   - 运行集成：`!python main.py --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'colab_model.mdl' --rep-idx 2 --perform-ensemble True`。
6. 预测与导出
   - 测试集预测：`--save-prediction true`。
   - 新对儿得分：`--save-pair-score true`，指向 `pairs/` 与 `pertid2fingerprint.pkl`；ZINC 评分使用 `--save-pair-score-zinc true`。
7. 可视化
   - 导出嵌入后（`--save-embed`），用 `tasks/run_plot.py` 的逻辑或 Notebook 内嵌 TSNE + Plotly 绘图。
8. 检查点与持久化
   - `CKPT_DIR='./results/'` 下的模型与日志可同步到 Drive；Notebook 提示路径位置与拷贝方法。

## 路径与数据位置
- 项目根：`/content/ReSimNet/`。
- 数据与结果：`/content/ReSimNet/tasks/data/`、`/content/ReSimNet/results/`（或映射到 Drive）。

## 风险与注意事项
- 旧版 API 适配是关键工作量；若坚持原始环境（PyTorch 0.3），Colab 安装会非常困难，不推荐。
- 二分类损失修正需与 `distance_layer` 的激活逻辑同步改动，避免重复 sigmoid。
- 大型数据下载需保证链接可访问且速度可接受；必要时提供备用镜像或提前放置到 Drive。

## 交付物
- `COLAB_ADAPTATION.md`：详细步骤、命令与注意事项（即本方案内容）。
- `ReSimNet_Colab.ipynb`：可直接运行的 Notebook，包含上述所有单元。

## 引用与定位
- 适配修改涉及的关键代码位置：
  - `models/drug_model.py:88`–`models/drug_model.py:93`（损失）、`models/drug_model.py:194`–`models/drug_model.py:207`（距离层）、`models/drug_model.py:183`（log_softmax）。
  - `tasks/drug_run.py:95`、`tasks/drug_run.py:254`（梯度裁剪）、`tasks/drug_run.py:457`（预测保存）。
  - `main.py:20`（导入路径）、`main.py:386`–`main.py:436`（集成逻辑，修正 `_model`）。

## 后续执行
- 若你确认此方案，我将创建 `COLAB_ADAPTATION.md` 并在仓库中提交补丁，随后生成并上传 `ReSimNet_Colab.ipynb` 的初始版本（包含环境与训练/预测单元）。