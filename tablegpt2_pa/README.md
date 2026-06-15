# TableGPT2 PA 本地微调与对比实验

## 目录

- `download_model.py`：下载 `tablegpt/TableGPT2-7B`
- `train_qlora.py`：执行 PA 二分类 `QLoRA` 微调与生成式评估
- `run_comparison.py`：运行 `RandomForest` 与 `XGBoost` 对照实验
- `package_bundle.py`：打包脚本、输出和可选模型目录

## 推荐流程

1. 安装依赖
2. 下载模型
3. 跑传统基线
4. 跑 `QLoRA`
5. 打包上传服务器

## 示例命令

```bash
python -m tablegpt2_pa.download_model --endpoint https://hf-mirror.com --method auto --local-dir ./models/TableGPT2-7B
python -m tablegpt2_pa.run_comparison --protocol A --top-p 16
python -m tablegpt2_pa.train_qlora --model-name-or-path ./models/TableGPT2-7B --protocol A --top-p 16 --k-shot 8 --use-4bit --bf16
python -m tablegpt2_pa.package_bundle --include-model-dir ./models/TableGPT2-7B
```

也可以先设置环境变量，再直接运行：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python -m tablegpt2_pa.download_model --local-dir ./models/TableGPT2-7B
```

如果镜像站对 `snapshot_download` 的元数据接口不稳定，推荐直接改用 `direct-resolve`：

```powershell
python -m tablegpt2_pa.download_model --endpoint https://hf-mirror.com --method direct-resolve --local-dir ./models/TableGPT2-7B
```

如果你本机的 `git-lfs` 网络更稳定，也可以改用：

```powershell
python -m tablegpt2_pa.download_model --endpoint https://hf-mirror.com --method git-lfs --local-dir ./models/TableGPT2-7B
```

## 说明

- 当前公开版 `TableGPT2-7B` 可直接作为 decoder checkpoint 使用
- 论文中的独立 `Semantic Table Encoder` 暂未完整公开，因此当前实现采用“公开 decoder + TAP-GPT 风格表格 prompt + QLoRA”的可运行方案
- 默认主协议为协议 A：严格确诊二分类
- `download_model.py` 和 `train_qlora.py` 都支持 `HF_ENDPOINT` / `--endpoint` / `--hf-endpoint`，可直接切换到镜像站
- `download_model.py` 默认 `--method auto`，会先尝试 `snapshot_download`，失败后自动回退到 `direct-resolve`，最后才回退 `git-lfs`
