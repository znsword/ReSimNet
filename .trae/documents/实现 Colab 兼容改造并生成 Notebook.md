# 实施计划：修改源码以兼容 Colab 并生成 Notebook

## 修改原则
- 按你的要求对原有代码进行“保留并注释原实现 + 新增兼容实现”的方式，不删除原代码。
- 仅做必要的适配：设备管理、旧版 PyTorch API 替换（Variable、F.sigmoid、F.log_softmax 的 dim）、loss 使用、.item/.detach、少量导入与打印修正。

## 具体改动
- `models/drug_model.py`
  - 添加 `self.device` 并统一用 `.to(self.device)` 代替 `.cuda()`。
  - `init_lstm_h`：用张量替换 `Variable(...).cuda()`；保留原行注释。
  - `siamese_sequence`：去除 `Variable(sort_idx).cuda()`，改为 `sort_idx.to(self.device)`；保留原行注释。
  - `F.log_softmax` 增加 `dim=-1`；保留原行注释。
  - 二分类激活：将 `F.sigmoid` 替换为 `torch.sigmoid`；保留原行注释。
  - 二分类损失：启用 `nn.BCELoss()`，注释掉自定义 lambda。
- `tasks/drug_run.py`
  - 用 `loss.item()` 替换 `loss.data[0]`；用 `outputs.detach().cpu().numpy()` 替换 `outputs.data.cpu().numpy()`；保留原行注释。
  - 为 `.cuda()` 调用增加设备获取 `device = next(model.parameters()).device` 并改为 `.to(device)`；保留原行注释。
- `tasks/drug_task.py`
  - `collate_fn` 与 `collate_fn_graph`：去除 `Variable(...)`；直接返回张量；保留原行注释。
- `main.py`
  - 修正导入 `from models.root.utils import *` → `from utils import *`；保留原行注释。
  - 模型构造返回 `.to(device)` 替代 `.cuda()`；在文件中定义 `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`；保留原行注释。
  - 修复 `print(model_name, _model)` 为 `print(model_name, model_idx)`；保留原行注释。

## 新增文件
- `ReSimNet_Colab.ipynb`：包含以下单元
  - 依赖安装（pip）与 GPU 检查
  - 项目路径与数据准备指引（使用 `gdown` 占位/示例）
  - 运行训练与预测的示例命令（直接调用 `main.py`）
  - 可视化示例（TSNE/Plotly，占位）

## 验证
- 运行 `python main.py --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'colab_model.mdl' --rep-idx 2` 验证训练流程在新 API 下正常。
- 运行 `--save-prediction` / `--save-pair-score` 检查预测导出是否正常。

## 交付
- 代码补丁与 `ReSimNet_Colab.ipynb` 添加至仓库，等待你审阅与后续执行。