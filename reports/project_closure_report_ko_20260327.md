# 프로젝트 종료 보고서

## 이 프로젝트는 무엇이었나

이 프로젝트는 아주 작은 프롬프트 단서가 대형 언어모델의 추론을 실제로 바꾸는지, 그리고 그 변화가 어떤 종류의 효과인지를 검증하려는 연구였다.

처음의 핵심 가설은 단순히 “단서가 영향을 준다”가 아니었다. 더 강한 주장, 즉 같은 개념을 여러 방식으로 바꿔 쓴 짧은 의미 단서 묶음이 정확 반복, 중립 문구, 정형화된 표현, 어휘 중첩, 절차 힌트 같은 통제 조건보다 더 강하고 더 안정적으로 특정 추론 패밀리를 선택적으로 유도한다는 가설이었다.

프로젝트 종료 시점의 결론은 명확하다. 이 넓은 의미의 “semantic flooding” 주장은 지지되지 않았다.

## 종료 시점의 핵심 연구 질문

프로젝트 후반부에는 연구 질문이 다음처럼 바뀌었다.

“답안 바인딩을 추론 과정에서 분리하면, LLM의 미세 단서 효과는 주로 의미적 flooding이 아니라 canonical 표현, operator/procedural 힌트, lexical overlap, prompt interface 효과에 존재하는가?”

쉽게 말하면 다음을 묻는 연구였다.

- 단서가 정말 의미 때문에 추론을 바꾸는가?
- 아니면 정형화된 표현, 연산자 설정, 표면 어휘 겹침, 위치 같은 인터페이스 요인 때문에 바뀌는가?

## 프로젝트가 어떻게 진행되었나

프로젝트는 여러 단계로 진행되었다.

1. 초기 탐색:
   자연어 과제와 합성 과제에서 cue 효과가 있는 것처럼 보였지만, 그 효과가 진짜 추론 변화인지 아니면 답안 형식이나 인터페이스 아티팩트인지 분리되지 않았다.

2. 대규모 감사:
   자연어 audit와 synthetic subset diagnostic을 거치면서 원래의 넓은 semantic flooding 가설은 약화되었다. 여러 경우에 semantic bundle보다 single canonical cue가 더 강했다.

3. construct-valid synthetic benchmark 구축:
   보다 깨끗한 검증을 위해 통제 가능한 합성 benchmark v3를 구축했다. 이 benchmark는 패밀리, 난이도, lexicalization regime을 명시적으로 조절할 수 있도록 설계되었다.

4. decoupled evaluation 도입:
   추론을 실제로 푸는 단계와, 푼 답을 최종 답안 형식으로 매핑하는 단계를 분리했다. 이 단계가 프로젝트 전체에서 가장 중요한 방법론적 전환이었다.

5. Phase E held-out confirmatory:
   Qwen2.5-7B 기반의 held-out synthetic confirmatory에서 semantic flooding은 여전히 약했다. 특히 binding-only 성능은 거의 ceiling에 가까워서, 취약점은 answer binding이 아니라 solve stage에 있다는 점이 드러났다.

6. Phase R0 rigour audit:
   Phase E가 여러 차례 끊기며 수행되었기 때문에 raw provenance를 다시 감사했다. 그 결과 기존 Phase E 숫자는 방향성은 있지만 publication-grade는 아니라고 판단했고, 깨끗한 confirmatory namespace를 새로 만들었다.

7. Phase R1 multimodel confirmatory:
   프로젝트의 최종적이고 가장 강한 증거는 이 깨끗한 4모델 confirmatory rerun에서 나왔다.

## 최종 실험 설정

최종 보관본에서 다시 재현할 수 있는 clean branch의 핵심 설정은 다음과 같다.

- benchmark 출처: construct-validity benchmark v3
- benchmark v3 전체 패밀리:
  boolean logic, ordering constraints, state tracking, temporal reasoning, truth consistency, causal reasoning
- 최종 confirmatory frozen slice:
  그중 4개 패밀리만 사용
- confirmatory 패밀리:
  `boolean_logic`, `ordering_constraints`, `state_tracking`, `temporal_reasoning`
- confirmatory 전체 크기:
  7,200개 항목
- lexicalization regime:
  `surface_aligned`, `paraphrastic`, `delexicalized`
- 평가 모드:
  `free_form_only`, `cot_before_options`, `standard_mc`, `binding_only`
- 논문 중심 primary mode:
  `free_form_only`, `cot_before_options`
- 최종 confirmatory cue 조건:
  `no_cue`, `generic_neutral`, `single_canonical`, `exact_repetition`, `matched_semantic`, `matched_procedural`, `matched_mixed`, `matched_lexical_overlap`, `matched_near_miss`
- frozen prompt interface:
  cue placement는 `after_question`, formatting은 `plain_sentence`

최종 clean model registry:

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `mistralai/Ministral-8B-Instruct-2410`

모델 접근 관련 주의:

- 원래는 `Llama-3.1-8B-Instruct` 또는 `Gemma-2-9b-it`도 후보였다.
- 그러나 이 환경에서는 gated weight 다운로드가 불가능했다.
- 그래서 open fallback으로 `Ministral-8B-Instruct-2410`을 사용했다.

## 무엇을 알아냈는가

### 원래의 큰 주장은 실패했다

이 저장소는 “paraphrastic same-concept semantic bundle이 가장 강하고 가장 안정적인 cue 효과의 원천”이라는 주장을 지지하지 않는다.

최종 clean multimodel confirmatory 기준으로 보면:

- 4개 모델 런이 모두 완주했다.
- 4개 모델 모두 completion과 integrity check를 통과했다.
- pooled primary-mode 분석에서 `matched_semantic`은 `no_cue`보다 낮았다.
- `single_canonical`보다도 낮았다.
- `matched_lexical_overlap`보다도 낮았다.
- `exact_repetition`보다도 낮았다.

`matched_semantic`이 pooled 기준으로 양의 차이를 보인 것은 `matched_procedural` 대비뿐이었지만, 그 효과는 작았고 모델별로도 일관적이지 않았다.

### 진짜 취약한 부분은 binding이 아니라 solve stage였다

이 프로젝트에서 가장 중요한 방법론적 결론 가운데 하나는 solving과 answer binding을 반드시 분리해야 한다는 점이다.

- `binding_only`는 거의 ceiling이었다.
- 불안정성과 차이는 주로 solve stage에서 나타났다.
- 따라서 solving과 binding을 분리하지 않으면 cue gain을 과대평가하거나 잘못 해석할 가능성이 크다.

### 무엇이 비교적 그럴듯하게 남았는가

프로젝트가 지지하는 더 좁은 주장은 다음과 같다.

- minimal cue 효과는 존재할 수 있다.
- 하지만 그 안정적인 residue는 broad semantic flooding이 아니다.
- 남는 효과는 더 잘게 분해하면 대체로 다음의 혼합처럼 보인다.
  canonical phrasing,
  lexical overlap,
  procedural/operator hint,
  prompt interface effect

즉, “의미 묶음이 추론을 강하게 steering한다”보다 “작은 단서 효과는 표면 표현과 연산자 설정, 인터페이스에 크게 의존한다”가 더 정직한 결론이다.

## 최종적으로 가져가야 할 세 가지 결론

이 프로젝트를 한 문단으로 기억해야 한다면 아래 세 줄이면 충분하다.

1. construct-valid하고 decoupled한 평가에서는 broad semantic flooding이 재현되지 않았다.
2. answer binding은 cue 효과를 숨기거나 부풀릴 수 있으므로 decoupled evaluation이 필요하다.
3. 남는 현상은 semantic steering보다 canonical, lexical, procedural, interface decomposition 쪽이 훨씬 더 잘 설명한다.

## 끝까지 풀리지 않은 것

모든 좁은 질문이 완전히 해결된 것은 아니다.

- procedural/operator residue는 깔끔하게 지배적이지 않았고 모델마다 섞여 있었다.
- placement와 interface 효과가 크다는 정황은 강했지만, 최종 placement-factorial branch를 완결된 paper package로 밀어붙이지는 못했다.
- operator-state probing도 강한 mechanistic conclusion까지는 가지 못했다.

따라서 이 프로젝트는 강한 decomposition 또는 falsification 결과는 남겼지만, 완결된 mechanism 논문으로 닫히지는 않았다.

## 종료 시점의 논문성 판단

종료 시점에서 가장 방어 가능한 논문 방향은 다음 둘 중 하나였다.

- benchmark + decomposition paper
- decoupled evaluation necessity를 강조하는 methods / falsification paper

반대로 가장 방어하기 어려운 방향은, 초반의 broad semantic flooding positive framing을 다시 살리는 것이다.

이 프로젝트를 미래에 다시 열더라도 그 방향으로 돌아가면 안 된다.

## 최종 clean run의 상태

최종 clean multimodel confirmatory는 완전히 끝난 상태다.

- 네 모델 모두 `259,200`행으로 완료되었다.
- `run_manifest`에서 모든 모델이 `completed`였다.
- `integrity_check`에서 duplicate key `0`, missing key `0`, extra key `0`, cross-model interface match `True`가 확인되었다.

즉, 최종 clean branch는 “완주했고, provenance와 completeness도 확인된 상태”로 닫혔다.

## 이 보관본이 의도적으로 남기는 것

이 보관본은 작게 만들었다.

장기 보관 목적상, 다음을 다시 이해하고 재가동할 수 있는 최소 집합만 남겼다.

- benchmark 생성 코드
- frozen confirmatory 입력 생성 코드
- decoupled evaluation 실행 코드
- confirmatory 후처리 코드
- prompt 자산
- 최종 config
- 이 종료 보고서

반대로, 대용량 raw 결과 디렉터리와 과거의 많은 중간 산출물은 넣지 않았다.

이 보관본의 목적은 “무엇을 했고 무엇을 알아냈는지 보존하는 것”이지, 모든 대형 중간 파일을 영구 보관하는 것이 아니기 때문이다.

## 보관본에 포함된 핵심 파일

포함된 source 파일:

- `src/__init__.py`
- `src/utils.py`
- `src/data_loading.py`
- `src/prompt_building.py`
- `src/scoring_methods.py`
- `src/score_mc.py`
- `src/stats.py`
- `src/build_benchmark_v3.py`
- `src/prepare_confirmatory_v3_inputs.py`
- `src/run_decoupled_eval.py`
- `src/postprocess_confirmatory_v3.py`

포함된 config:

- `configs/construct_validity_phaseC_v3.yaml`
- `configs/confirmatory_v3_qwen7b.yaml`
- `configs/confirmatory_v3_qwen14b.yaml`
- `configs/confirmatory_v3_deepseek7b.yaml`
- `configs/confirmatory_v3_ministral8b.yaml`

포함된 prompt 자산:

- `prompts/bundles_v3.yaml`
- `prompts/templates_v2.yaml`

포함된 환경 사양:

- `requirements.txt`

포함된 종료 보고서:

- `reports/project_closure_report_20260327.md`
- `reports/project_closure_report_ko_20260327.md`

## 최종 파이프라인을 다시 실행하는 방법

archive 루트 기준 핵심 워크플로는 다음과 같다.

1. construct-valid benchmark 생성:

```bash
python -m src.build_benchmark_v3 --config configs/construct_validity_phaseC_v3.yaml
```

2. frozen 4-family confirmatory slice 준비:

```bash
python -m src.prepare_confirmatory_v3_inputs
```

3. decoupled confirmatory 실행:

```bash
python -m src.run_decoupled_eval --config configs/confirmatory_v3_qwen7b.yaml --gpu-id 0
python -m src.run_decoupled_eval --config configs/confirmatory_v3_qwen14b.yaml --gpu-id 1
python -m src.run_decoupled_eval --config configs/confirmatory_v3_deepseek7b.yaml --gpu-id 2
python -m src.run_decoupled_eval --config configs/confirmatory_v3_ministral8b.yaml --gpu-id 3
```

4. 완료 후 후처리:

```bash
python -m src.postprocess_confirmatory_v3 --bootstrap-samples 1000
```

종료 시점에는 후처리 시간을 현실적으로 맞추기 위해 bootstrap을 `1000`으로 사용했다. paired confidence interval과 verdict label은 그대로 생성된다.

## 미래 독자를 위한 한 줄 해석

이 프로젝트는 LLM 추론에서 micro-cue steering을 연구했지만, 가장 깨끗한 증거는 원래의 broad semantic flooding 이야기를 지지하지 않았다. 대신 이 프로젝트의 진짜 기여는 construct-valid benchmark와 decoupled evaluation 파이프라인, 그리고 cue 효과를 canonical, lexical, procedural, interface 요인으로 분해해야 한다는 결론에 있다.

## 종료 권고

프로젝트를 계속 닫아둘 경우, 이 보관본이면 핵심은 충분히 남는다.

만약 미래에 다시 시작한다면, benchmark / protocol / decomposition framing에서 재출발해야 한다.
예전 semantic-flooding mainline으로 되돌아가면 안 된다.
