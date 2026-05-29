<div align="center">

# minimal-cue-llm-reasoning

**细微的提示线索真的能左右 LLM 的推理吗**

![Status](https://img.shields.io/badge/status-dormant-lightgrey)
![Language](https://img.shields.io/badge/language-Python-3776AB?logo=python&logoColor=white)
[![Verify](https://github.com/jumincho/minimal-cue-llm-reasoning/actions/workflows/verify.yml/badge.svg)](https://github.com/jumincho/minimal-cue-llm-reasoning/actions/workflows/verify.yml)
![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)
![Closure](https://img.shields.io/badge/closure-2026--03-blue)

[한국어](./README.md) · [English](./README.md#english) · **中文**

</div>

> 🧊 **休眠中的研究试点。**

## ⭐ 核心结果 (TL;DR)

- **出发假设(用多种表达灌输同一语义会更强地引导推理)被证伪**——并未胜过单纯重复或词面重叠等对照,四个模型结果一致。
- **留下的是更重要的方法论结论**——必须把"求解"与"绑定到选项"分离(绑定接近上限,真正差异在求解),否则会高估线索效应。
- 残余的小效应来自**"表面"而非"语义"**(词重叠、措辞、位置等界面因素)。

## 这项研究想看什么

在语言模型前面加上很短的一句话(提示线索),同样的题目也常常会得到不同的答案。
本项目的出发假设比这更进一步:

> **当用不同的表达把"同一个意思"反复轻轻地灌入模型时**,模型是否会比单纯重复或
> 表层词重叠等对照条件更明显地朝某种推理方式倾斜?

为了验证这个假设需要两件东西:

- 一个能干净地看到"是什么在推动推理"的、受控良好的小型基准。
- 把"求解阶段"和"把答案绑定到某个选项的阶段"分离开来评估的方法。

把这两样都做出来,然后在 Qwen 7B/14B、DeepSeek 7B、Ministral 8B 上做完全一致的对比。

## 发现了什么

- **出发假设没有得到支持。** 用多种表达灌输同一个意思,并没有比"重复同一句"或"仅词面重叠"这种对照条件做得更好,四个模型上的结果都一致。
- **但留下了一个更重要的方法论结论。** "把答案绑定到某个选项"这个阶段几乎总是接近上限,真正的差异出现在"求解阶段"。如果不分开看,提示线索的效果就很容易被高估。
- **剩下的那点小效应,其实更像"表面"而非"语义"。** 提示线索的确会影响输出,但更多由词重叠、定型用语、过程性提示,以及线索放在提示的什么位置这种"界面因素"来解释,而不是"语义本身"。

完整结果可在以下两份关闭报告中查阅:

- 🇰🇷 [`reports/project_closure_report_ko_20260327.md`](reports/project_closure_report_ko_20260327.md)
- 🇬🇧 [`reports/project_closure_report_20260327.md`](reports/project_closure_report_20260327.md)

## 为什么暂停

最初瞄准的大主张(语义性的灌输能强烈引导推理)在最干净的检验中没有立住。
但过程中诞生的两件东西 —— 受控的小基准 和 求解/绑定分离的评估方法 —— 本身仍然有价值。
如果未来重启,自然的方向不是回头再去争论"语义灌输"这件事,而是把这两件工具用起来,
重新框定为 **"基准 + 提示线索效应的分解"** 或 **"为什么必须分离评估"**。

## 重启时先看哪里

- 📖 [`GLOSSARY.md`](GLOSSARY.md) —— 把代码与关闭报告里出现的内部术语 (`phaseC`、V2 family、SSI、评估模式、条件标签等) 翻成日常用语的对照表
- 🇬🇧 [`reports/project_closure_report_20260327.md`](reports/project_closure_report_20260327.md) —— 一篇完整篇幅的关闭报告,建议第一个看(中文读者可读英文版)
- 🇰🇷 [`reports/project_closure_report_ko_20260327.md`](reports/project_closure_report_ko_20260327.md) —— 韩文版关闭报告
- [`prompts/bundles_v3.yaml`](prompts/bundles_v3.yaml) —— 比较中用到的提示线索家族定义
- [`prompts/templates_v2.yaml`](prompts/templates_v2.yaml) —— 提示模板冻结后的接口(线索放在题目前后什么位置)
- [`configs/construct_validity_phaseC_v3.yaml`](configs/construct_validity_phaseC_v3.yaml) —— 受控基准的生成配置
- [`configs/`](configs/) 下四个模型对应的最终确证配置

## 代码地图

| 文件 | 做什么 |
|---|---|
| [`src/build_benchmark_v3.py`](src/build_benchmark_v3.py) | 生成受控小型基准 |
| [`src/prepare_confirmatory_v3_inputs.py`](src/prepare_confirmatory_v3_inputs.py) | 准备最终确证阶段所需的输入 |
| [`src/run_decoupled_eval.py`](src/run_decoupled_eval.py) | 跑求解/绑定分离的评估 |
| [`src/postprocess_confirmatory_v3.py`](src/postprocess_confirmatory_v3.py) | 用配对统计后处理结果 |
| [`src/prompt_building.py`](src/prompt_building.py) | 把提示线索与模板组合成实际的 prompt |
| [`src/scoring_methods.py`](src/scoring_methods.py), [`src/score_mc.py`](src/score_mc.py) | 选择题打分 |
| [`src/stats.py`](src/stats.py) | bootstrap / 配对统计 |
| [`src/data_loading.py`](src/data_loading.py), [`src/utils.py`](src/utils.py) | I/O 与共用工具 |

## 目录概览

```
.
├── src/                实验代码
├── configs/            基准生成与每个模型的最终确证配置
├── prompts/            提示线索定义 / 模板
├── reports/            关闭报告(韩文 / 英文)
├── GLOSSARY.md         内部术语词典
└── requirements.txt
```

## 环境

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -U -r requirements.txt
export HF_TOKEN=...   # 仅在需要时
```

## 重新跑一遍的大致流程

> **注意：** `data/` 目录不包含在仓库中。第 1 步（`build_benchmark_v3`）必须先运行——它会创建 `data/processed/` 及后续步骤所需的全部子目录。在全新克隆的仓库中，如果跳过第 1 步直接运行第 2 步或第 3 步，会因找不到对应路径而报错，请务必按顺序执行。

```bash
# 1) 生成受控基准 —— data/processed/ 目录在此步骤首次创建
python -m src.build_benchmark_v3 --config configs/construct_validity_phaseC_v3.yaml

# 2) 准备最终确证阶段的输入
python -m src.prepare_confirmatory_v3_inputs

# 3) 跑分离评估(每个模型独占一张 GPU)
python -m src.run_decoupled_eval --config configs/confirmatory_v3_qwen7b.yaml      --gpu-id 0
python -m src.run_decoupled_eval --config configs/confirmatory_v3_qwen14b.yaml     --gpu-id 1
python -m src.run_decoupled_eval --config configs/confirmatory_v3_deepseek7b.yaml  --gpu-id 2
python -m src.run_decoupled_eval --config configs/confirmatory_v3_ministral8b.yaml --gpu-id 3

# 4) 配对统计后处理
python -m src.postprocess_confirmatory_v3 --bootstrap-samples 1000
```

## 状态

🧊 **休眠中** —— 起初的大主张被证伪了,但留下的基准与分离评估方法仍然完好可用。

## 许可证

以 [CC BY-NC 4.0](./LICENSE) 发布。
