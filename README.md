<div align="center">

# minimal-cue-llm-reasoning

**미세 단서가 LLM 추론을 정말로 흔드는가**
**Do minimal prompt cues actually steer LLM reasoning?**

![Status](https://img.shields.io/badge/status-dormant-lightgrey)
![Language](https://img.shields.io/badge/language-Python-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)
![Closure](https://img.shields.io/badge/closure-2026--03-blue)

**한국어** · [English](#english) · [中文](./README.zh-CN.md)

</div>

> 🧊 **휴면(dormant) 중인 연구 파일럿입니다.**

## 무엇을 보려던 연구였나

언어모델 앞에 아주 짧은 한 문장(단서)를 붙이면, 같은 문제라도 답이 달라지곤 합니다. 이 프로젝트의 출발 가설은 그것보다 한 발 더 들어갔습니다.

> **같은 의미를 다른 표현으로 살짝씩 바꿔 가며 여러 번 흘려 주면**, 단순 반복이나 표면 단어 겹침 같은 통제 조건보다 모델이 특정 추론 방식을 더 강하게 따라간다.

이 가설을 검증하려면 두 가지가 필요했습니다.

- 무엇이 추론을 흔드는지 깔끔하게 볼 수 있는 잘 통제된 작은 벤치마크
- "문제를 푸는 단계" 와 "답을 보기 중 하나에 매핑하는 단계" 를 분리해서 보는 평가 방법

이 둘을 만들고, 여러 모델(Qwen 7B/14B, DeepSeek 7B, Ministral 8B) 에서 동일하게 비교했습니다.

## 무엇을 알아냈나

- **출발 가설은 지지받지 못했습니다.** 같은 의미를 여러 표현으로 흘려 주는 방식이, 단순히 같은 문장을 반복하거나 단어가 살짝 겹치는 문장을 넣는 것보다 **더 잘 작동하진 않았습니다.** 네 모델 모두에서 같은 패턴이 나왔습니다.
- **대신 더 중요한 방법론적 결론이 남았습니다.** "답을 보기에 매핑하는 단계" 는 거의 항상 정답이 나오고, 진짜 차이는 "문제를 푸는 단계" 에서 생긴다는 점입니다. 이 둘을 분리하지 않고 보면 단서의 효과를 과대평가하기 쉽습니다.
- **남은 작은 효과는 "의미" 가 아니라 "표면" 쪽이었습니다.** 단서가 영향을 주긴 하지만, 그 영향은 같은 의미를 어떻게 표현했느냐 보다 단어 겹침, 정형화된 문구, 절차 힌트, 단서를 어디에 어떻게 두느냐 같은 인터페이스 요인에서 더 잘 설명됐습니다.

자세한 결과가 궁금하시면:

- 🇰🇷 [`reports/project_closure_report_ko_20260327.md`](reports/project_closure_report_ko_20260327.md)
- 🇬🇧 [`reports/project_closure_report_20260327.md`](reports/project_closure_report_20260327.md)

## 왜 잠시 멈춰 두는가

원래 노렸던 큰 주장(의미 단서가 강하게 steering 한다) 은 가장 깨끗한 검증에서 지지받지 못했습니다. 다만 그 과정에서 만들어진 두 가지 — 잘 통제된 작은 벤치마크와 풀이/매핑 분리 평가 — 자체는 분명한 기여입니다. 다시 시작한다면 "의미가 강하게 steering 한다" 쪽으로 돌아가는 게 아니라, 그 두 도구를 살려서 **벤치마크 + 단서 효과 분해** 또는 **분리 평가의 필요성** 쪽으로 framing 해야 자연스럽습니다.

## 다시 들여다볼 때는 어디부터

- 🇰🇷 [`reports/project_closure_report_ko_20260327.md`](reports/project_closure_report_ko_20260327.md) — 한 편 분량의 종료 보고서. 가장 먼저 읽으면 좋은 글
- [`prompts/bundles_v3.yaml`](prompts/bundles_v3.yaml) — 비교에 쓴 단서 종류들이 정의돼 있음
- [`prompts/templates_v2.yaml`](prompts/templates_v2.yaml) — 단서를 문제 앞뒤 어디에 어떻게 두는지의 고정된 인터페이스
- [`configs/construct_validity_phaseC_v3.yaml`](configs/construct_validity_phaseC_v3.yaml) — 작은 벤치마크 생성 설정
- [`configs/`](configs/) 아래의 모델별 최종 확인 설정 4개

## 코드 어디에 뭐가 있나

| 파일 | 하는 일 |
|---|---|
| [`src/build_benchmark_v3.py`](src/build_benchmark_v3.py) | 잘 통제된 작은 벤치마크 생성 |
| [`src/prepare_confirmatory_v3_inputs.py`](src/prepare_confirmatory_v3_inputs.py) | 마지막 확인 단계용 입력 준비 |
| [`src/run_decoupled_eval.py`](src/run_decoupled_eval.py) | 풀이 단계와 매핑 단계를 분리한 평가 실행 |
| [`src/postprocess_confirmatory_v3.py`](src/postprocess_confirmatory_v3.py) | 결과를 짝지은 통계로 후처리 |
| [`src/prompt_building.py`](src/prompt_building.py) | 단서와 템플릿을 합쳐 실제 프롬프트로 조립 |
| [`src/scoring_methods.py`](src/scoring_methods.py), [`src/score_mc.py`](src/score_mc.py) | 객관식 채점 |
| [`src/stats.py`](src/stats.py) | 부트스트랩 / 짝지은 통계 |
| [`src/data_loading.py`](src/data_loading.py), [`src/utils.py`](src/utils.py) | 입출력 / 공용 유틸 |

## 폴더 지도

```
.
├── src/                실험 코드
├── configs/            벤치마크 생성과 모델별 확인 설정
├── prompts/            단서 정의 / 템플릿
├── reports/            종료 보고서 (한국어 / 영문)
└── requirements.txt
```

## 환경

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -U -r requirements.txt
export HF_TOKEN=...   # 필요한 경우에만
```

## 다시 돌리는 큰 흐름

```bash
# 1) 잘 통제된 벤치마크 생성
python -m src.build_benchmark_v3 --config configs/construct_validity_phaseC_v3.yaml

# 2) 최종 확인용 입력 준비
python -m src.prepare_confirmatory_v3_inputs

# 3) 분리 평가 실행 (모델마다 GPU 한 장)
python -m src.run_decoupled_eval --config configs/confirmatory_v3_qwen7b.yaml      --gpu-id 0
python -m src.run_decoupled_eval --config configs/confirmatory_v3_qwen14b.yaml     --gpu-id 1
python -m src.run_decoupled_eval --config configs/confirmatory_v3_deepseek7b.yaml  --gpu-id 2
python -m src.run_decoupled_eval --config configs/confirmatory_v3_ministral8b.yaml --gpu-id 3

# 4) 짝지은 통계로 후처리
python -m src.postprocess_confirmatory_v3 --bootstrap-samples 1000
```

## 상태

🧊 **휴면 중** — 원래 큰 주장은 반증됐지만, 만들어진 벤치마크와 분리 평가 방법은 살아 있는 상태입니다.

---

<a name="english"></a>

## English

> 🧊 **Dormant research pilot.**

### What this set out to test

If you stick a short sentence (a "cue") in front of a language model, the same question can yield different answers. This project pushed one step further than that observation:

> **If you flood the model with several rephrasings of the same underlying meaning,** does that steer reasoning more reliably than control conditions like plain repetition or surface word overlap?

Testing that needed two things:

- A small, well-controlled benchmark where you can cleanly see what is moving reasoning.
- An evaluation that separates "solving the problem" from "binding the solution to an answer choice."

Both were built, then run identically on Qwen 7B/14B, DeepSeek 7B, and Ministral 8B.

### What it found

- **The starting hypothesis didn't hold.** Multiple semantic paraphrasings of a cue did **not** outperform plain repetition or near-lexical overlap controls. The same pattern showed up across all four models.
- **A more important methodological finding stayed.** The "binding" stage is near ceiling almost always; the real movement happens at the "solve" stage. If you don't separate them, the apparent cue effect is overstated.
- **The small remaining effect is on surface, not meaning.** Cues do move things, but the movement is better explained by lexical overlap, canonical phrasing, procedural hints, and where the cue sits in the prompt — interface factors, not semantics.

Full results:

- 🇰🇷 [`reports/project_closure_report_ko_20260327.md`](reports/project_closure_report_ko_20260327.md)
- 🇬🇧 [`reports/project_closure_report_20260327.md`](reports/project_closure_report_20260327.md)

### Why it's on hold

The big original claim ("semantic flooding steers reasoning") didn't survive the cleanest test. The two artifacts produced along the way — the controlled benchmark and the solve/bind decoupled evaluation — remain useful on their own. A natural restart would not re-litigate the semantic-flooding claim but would reframe around either **benchmark + cue-effect decomposition** or **the need for decoupled evaluation**.

### Where to look first when revisiting

- 🇬🇧 [`reports/project_closure_report_20260327.md`](reports/project_closure_report_20260327.md) — A full closure report. Read this first.
- [`prompts/bundles_v3.yaml`](prompts/bundles_v3.yaml) — the cue families being compared.
- [`prompts/templates_v2.yaml`](prompts/templates_v2.yaml) — the frozen prompt interface (where the cue sits).
- [`configs/construct_validity_phaseC_v3.yaml`](configs/construct_validity_phaseC_v3.yaml) — benchmark generation config.
- [`configs/`](configs/) — the four model-specific confirmatory configs.

### Code map

| File | What it does |
|---|---|
| [`src/build_benchmark_v3.py`](src/build_benchmark_v3.py) | Generates the controlled benchmark |
| [`src/prepare_confirmatory_v3_inputs.py`](src/prepare_confirmatory_v3_inputs.py) | Prepares confirmatory inputs |
| [`src/run_decoupled_eval.py`](src/run_decoupled_eval.py) | Runs the solve/bind decoupled evaluation |
| [`src/postprocess_confirmatory_v3.py`](src/postprocess_confirmatory_v3.py) | Paired-statistics postprocessing |
| [`src/prompt_building.py`](src/prompt_building.py) | Assembles cues + templates into final prompts |
| [`src/scoring_methods.py`](src/scoring_methods.py), [`src/score_mc.py`](src/score_mc.py) | Multiple-choice scoring |
| [`src/stats.py`](src/stats.py) | Bootstrap / paired statistics |
| [`src/data_loading.py`](src/data_loading.py), [`src/utils.py`](src/utils.py) | I/O and shared utilities |

### Folder map

```
.
├── src/                experiment code
├── configs/            benchmark and per-model confirmatory configs
├── prompts/            cue definitions and templates
├── reports/            closure reports (KO / EN)
└── requirements.txt
```

### Environment

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -U -r requirements.txt
export HF_TOKEN=...   # only if needed
```

### Big-picture rerun

```bash
# 1) build the controlled benchmark
python -m src.build_benchmark_v3 --config configs/construct_validity_phaseC_v3.yaml

# 2) prepare confirmatory inputs
python -m src.prepare_confirmatory_v3_inputs

# 3) run the decoupled evaluation (one GPU per model)
python -m src.run_decoupled_eval --config configs/confirmatory_v3_qwen7b.yaml      --gpu-id 0
python -m src.run_decoupled_eval --config configs/confirmatory_v3_qwen14b.yaml     --gpu-id 1
python -m src.run_decoupled_eval --config configs/confirmatory_v3_deepseek7b.yaml  --gpu-id 2
python -m src.run_decoupled_eval --config configs/confirmatory_v3_ministral8b.yaml --gpu-id 3

# 4) paired-statistics postprocessing
python -m src.postprocess_confirmatory_v3 --bootstrap-samples 1000
```

### Status

🧊 **Dormant** — the headline claim was refuted, but the benchmark and the decoupled-evaluation method are intact.

### License

Released under [CC BY-NC 4.0](./LICENSE).
