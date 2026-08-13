# experiments

这里按照科学问题组织 controlled/probe 实验，而不是按照 PCA、Gram、CKA 等分析工具组织实验。

每个实验目录至少保留：

- `README.md`：科学问题、控制变量、输入条件和分析计划。
- `config.yaml`：sequence、condition、geometry source、token source 和分析方法。

当前接口：

- `exp01_history_conditioning/`
- `exp02_order_sensitivity/`
- `exp03_geometry_appearance_conflict/`
- `controlled_16_exp_v3/`
- `exp_template/`

新增实验：

```bash
python scripts/make_experiment_folder.py --name exp04_xxx
```
