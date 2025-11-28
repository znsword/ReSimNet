# 将 main.py 的详细解析追加到 Markdown 的计划

## 目标

* 在现有 `SOURCE_STRUCTURE.md` 末尾追加一段以 `---` 分隔的新内容，系统性解释 `main.py` 的代码结构与运行流程。

## 内容结构

* 常量与默认路径：数据/模型/结果目录等

* 命令行参数：按类别分组（数据、训练、模型、图）

* 核心函数：`run_experiment`、`get_dataset`、`get_run_fn`、`get_model`、`init_logging`、`init_seed`、`init_parameters`、`main`

* 运行分支：训练/验证/测试、保存嵌入与预测、对新对儿打分、集成

* 关键注意点与潜在问题：导入路径、未定义变量等

* 提供 `file_path:line_number` 精确引用，便于在 IDE 定位

## 实施方式

* 直接编辑 `SOURCE_STRUCTURE.md`，在文末追加新章节与分隔线，不创建新文件。

