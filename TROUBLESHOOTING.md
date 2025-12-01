# ReSimNet 近期错误与修复

## 1) AttributeError: `np.float_` 被 NumPy 2.0 移除
- 错误信息：`AttributeError: np.float_ was removed in the NumPy 2.0 release`
- 触发位置：`tasks/drug_task.py:689-690`
- 原因：使用了旧别名类型 `np.float_`、`np.int_`
- 修复：改为明确类型 `np.float64`、`np.int64`
- 代码引用：`d:\document\coding\python\ReSimNet\tasks\drug_task.py:689`

## 2) IndexError: 0 维张量索引（PyTorch 新版）
- 错误信息：`IndexError: invalid index of a 0-dim tensor. Use tensor.item()`
- 触发位置：`tasks/drug_run.py:240`
- 原因：使用旧式 `loss.data[0]` 访问标量损失
- 修复：改为 `loss.item()`；同时将 `score.cuda()` 统一为 `score.to(device)`
- 代码引用：
  - `d:\document\coding\python\ReSimNet\tasks\drug_run.py:239-241`

## 3) ValueError: Pearson 相关系数长度不足
- 错误信息：`ValueError: x and y must have length at least 2.`
- 触发位置：`tasks/drug_run.py:397`（`evaluation` 函数）
- 原因：某集合（如 KU/UU）在当前批次样本数不足 2 条
- 修复：
  - `evaluation` 中长度不足返回 `nan` 而非抛错
  - `precision_at_k` 在 `topk` 为空时返回 `nan`
  - `AUROC` 需同时存在正负样本才计算
- 代码引用：
  - `d:\document\coding\python\ReSimNet\tasks\drug_run.py:397`
  - `d:\document\coding\python\ReSimNet\tasks\drug_run.py:376`
  - `d:\document\coding\python\ReSimNet\tasks\drug_run.py:405-409`

## 备注
- 以上修复保持训练流程不中断；当样本不足时日志显示 `nan` 属预期表现
- 建议在 epoch 结束汇总再评估，减少 `nan`
