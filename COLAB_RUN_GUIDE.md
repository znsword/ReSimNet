# Google Colab 运行指南（ReSimNet_Colab.ipynb）

## 环境准备
- 在浏览器打开 Google Colab 并登录 Google 账号。
- 切换运行时到 GPU：
  - 菜单 Runtime → Change runtime type → Hardware accelerator 选择 `GPU`。
- 基础检查：在 Notebook 第一个单元会执行 `nvidia-smi`，确保看到 GPU 信息；若提示 `No GPU`，请确认运行时设置。

## 获取工程
- 方式一：直接在 Colab 中克隆仓库
  - 在新 Notebook 的代码单元运行：
    - `!git clone <你的仓库地址>`
    - `cd ReSimNet`
- 方式二：上传或同步到 Colab
  - 将本地项目压缩上传到 Colab，解压到 `/content/ReSimNet/ReSimNet`。
  - 或挂载 Google Drive 后，将项目目录复制到 `/content`。
  - Notebook 中使用的项目根路径为：`/content/ReSimNet/ReSimNet`。

## 数据准备
- 必需文件（保存到 `./tasks/data/`）：
  - `ReSimNet-Dataset.pkl`（训练/验证数据）
  - `pertid2fingerprint.pkl`（药物 ID → 指纹向量，用于打分）
  - 示例对：`tasks/data/pairs/examples.csv`
  - ZINC 测试包与示例：`tasks/data/pairs_zinc/zinc-test/` 与 `tasks/data/pairs_zinc/example_drugs.csv`
- 下载方法：
  - 使用 `gdown` 从共享链接下载：
    - `!pip install -q gdown`
    - `!mkdir -p tasks/data`
    - `!gdown <ReSimNet-Dataset.pkl 分享链接> -O tasks/data/ReSimNet-Dataset.pkl`
  - 或将文件预先放入 Google Drive，再在 Colab 中复制到项目的 `tasks/data`。

## 打开与运行 Notebook
- 在 Colab 中打开仓库中的 `ReSimNet_Colab.ipynb`。
- 依赖安装：Notebook 第一个代码单元会安装依赖：
  - `!pip install -q numpy pandas scikit-learn seaborn plotly gdown`
- 路径设置与校验：Notebook 会切换到项目根 `CWD` 并打印当前路径。
- 训练（ECFP，`rep-idx=2`）：
  - 运行训练单元：
    - `!python main.py --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'colab_model.mdl' --rep-idx 2 --train True --valid True --test True`
  - 输出：训练日志写入 `./results/logs/colab_model.mdl.txt`，最佳模型保存到 `./results/colab_model.mdl`。
- 导出测试集预测：
  - 运行：
    - `!python main.py --save-prediction true --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'colab_model.mdl' --rep-idx 2`
  - 输出：预测 CSV 在 `./results/pred_colab_model.mdl.csv`。
- 对新药物对打分：
  - 准备 `tasks/data/pairs/`（包含需要评分的 `*.csv`，每行两列为药物 ID），以及 `tasks/data/pertid2fingerprint.pkl`。
  - 运行：
    - `!python main.py --save-pair-score true --pair-dir './tasks/data/pairs/' --fp-dir './tasks/data/pertid2fingerprint.pkl' --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'colab_model.mdl' --rep-idx 2`
  - 输出：逐文件的打分结果保存到 `./results/save_pair_score/` 下。

## 结果与持久化
- 模型与日志目录：`./results/`
  - `logs/colab_model.mdl.txt`：训练/验证日志
  - `colab_model.mdl`：最佳模型检查点
  - `pred_colab_model.mdl.csv`：测试集预测
  - `save_pair_score/*.csv`：新对儿打分结果
- 同步到 Google Drive（可选）：
  - 挂载：`from google.colab import drive; drive.mount('/content/drive')`
  - 复制：`!cp -r results /content/drive/MyDrive/ReSimNet_results`

## 参数说明（常用）
- `--data-path`：训练/验证数据集路径（`./tasks/data/ReSimNet-Dataset.pkl`）
- `--model-name`：模型文件名（保存与加载时使用）
- `--rep-idx`：输入表示类型（`0:SMILES`、`1:InChIKey`、`2:ECFP`、`3:Mol2vec`、`4:图表示`）
- `--train/--valid/--test`：是否执行训练/验证/测试流程
- `--save-prediction`：导出测试集预测 CSV
- `--save-pair-score`：对新对儿打分；配合 `--pair-dir` 与 `--fp-dir`
- `--save-pair-score-zinc`：对 ZINC 测试集执行打分；配合 `--pair-dir` 与 `--example-dir`
- `--perform-ensemble`：执行多模型集成评估

## 常见问题与排障
- 无 GPU：在 Runtime → Change runtime type 选择 GPU，并重新连接运行时。
- 依赖问题：先运行 Notebook 的安装单元；若报版本错误，执行 `!pip install -U torch` 并重试。
- 数据路径错误：确认 `tasks/data` 下文件存在且命名正确；Notebook 会打印 `CWD`，确保当前目录为项目根。
- 显存不足：减少 `--batch-size` 或缩短 `--epoch`，或关闭其他占用 GPU 的 notebook。
- 权限问题（Drive）：首次挂载需要在弹窗中授权；若复制失败，检查目标目录存在与写入权限。

## 与源码改动的匹配
- Notebook 依赖以下兼容性改动：
  - 设备管理与 `.to(device)` 替换 `.cuda()`（`models/drug_model.py`、`main.py`、`tasks/drug_run.py`）
  - 移除 `Variable`、使用 `.item()` 与 `.detach()`（`models/drug_model.py`、`tasks/drug_task.py`、`tasks/drug_run.py`）
  - `F.log_softmax(..., dim=-1)` 与二分类激活/损失的标准化（`models/drug_model.py`）

## 验证步骤（建议）
- 运行训练命令并观察 `./results/logs/colab_model.mdl.txt` 是否记录训练过程。
- 运行导出预测命令，确认 `pred_colab_model.mdl.csv` 生成。
- 准备一份小型对儿文件，运行打分命令，检查 `save_pair_score/*.csv` 输出是否包含预测分数。

---

通过以上步骤，即可在 Colab 上完整运行该 Notebook，包括训练、预测与新药物对打分。如需添加 TSNE 可视化或同步到 Drive 的更多单元，可在 Notebook 中按需扩展相应代码单元。