# 计划：编写详细的 Google Colab Notebook 运行说明文档

## 目标
- 编写一份 Markdown 文档，逐步说明如何在 Colab 上运行 `ReSimNet_Colab.ipynb`，从环境选择、依赖安装、数据准备到训练、预测与结果持久化。

## 文档结构
- 环境准备：选择 GPU、基础检查
- 获取工程：clone/上传/Drive 同步方式
- 数据准备：下载与摆放必需数据文件
- 打开与运行 Notebook：依赖安装、路径设置、执行训练与预测单元
- 结果与持久化：输出文件位置与同步到 Drive
- 参数说明：常用 CLI 参数与含义
- 常见问题与排障：依赖、数据路径、显存、权限等

## 实施方式
- 在仓库根目录新增 `COLAB_RUN_GUIDE.md`，填入上述内容，使用中文说明与具体命令示例，确保与现有 `ReSimNet_Colab.ipynb` 一致。

## 输出
- 一次性创建并提交 `COLAB_RUN_GUIDE.md` 文档。