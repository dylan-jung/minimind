# Week 8: Agentic RL & Tool Use

> 목표: 모델이 외부 도구를 호출하는 능력을 RL로 학습하는 과정을 이해한다.
> LLM이 "행동"하는 Agent로 진화하는 최신 기법.

---

## 공식 학습 자료

공식 Deep Mastery 커리큘럼 이후의 심화 영역.
MiniMind-3에서 새로 추가된 Agentic RL은 공식 wiki에 아직 별도 모듈이 없으므로,
코드와 논문 중심으로 학습한다.

| 공식 가이드 | URL |
|-------------|-----|
| Deep Mastery (전체 로드맵) | https://www.minimind.wiki/en/docs/guide/mastery |

---

## 배경: LLM Agent란

```
기존 LLM:   질문 → 답변 (텍스트만)
Agent LLM:  질문 → 생각 → 도구 호출 → 결과 확인 → 최종 답변
```

모델이 `<tool_call>` 태그를 생성하면, 환경이 도구를 실행하고 결과를 `<tool_response>`로 돌려준다.
이 멀티턴 상호작용을 RL로 학습.

---

## 1. Tool Use 개념

### 논문

- **Toolformer: Language Models Can Teach Themselves to Use Tools** (Schick et al., 2023)
  - https://arxiv.org/abs/2302.04761
  - 모델이 API 호출을 자율적으로 학습하는 방법. Section 3이 핵심.

- **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2023)
  - https://arxiv.org/abs/2210.03629
  - Reasoning + Acting 패턴. Think → Act → Observe 루프의 원형.

- (선택) **Gorilla: Large Language Model Connected with Massive APIs** (Patil et al., 2023)
  - https://arxiv.org/abs/2305.15334
  - API 호출에 특화된 LLM 학습.

### 코드: `trainer/train_agent.py` — 도구 정의 (line 39~63)

```python
TOOLS = [
    {"type": "function", "function": {
        "name": "calculate_math",
        "description": "계산 수학표현식",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}
        }}
    }},
    # unit_converter, get_current_weather, get_current_time,
    # get_exchange_rate, translate_text
]
```

6개의 도구가 정의되어 있고, 각각 모의 데이터로 실행 결과를 시뮬레이션:
```python
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(args["expression"]))},
    "get_current_weather": lambda args: {"city": ..., "temperature": ...},
    ...
}
```

---

## 2. 도구 호출 파싱 & 실행

### 코드: `trainer/train_agent.py` (line 76~)

```python
def parse_tool_calls(text):
    """모델 출력에서 <tool_call> 태그를 파싱"""
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        calls.append(json.loads(m.strip()))
    return calls
```

모델이 생성하는 형식:
```
<think>사용자가 날씨를 물어보고 있으니 도구를 사용해야 해</think>
<tool_call>{"name": "get_current_weather", "arguments": {"location": "서울"}}</tool_call>
```

환경이 돌려주는 형식:
```
<tool_response>{"city": "서울", "temperature": "22°C", "condition": "맑음"}</tool_response>
```

---

## 3. Agentic RL 학습

### 핵심 아이디어

SFT로 기본적인 tool call 형식을 학습한 뒤, **RL로 실제 도구 사용 능력을 강화**:
- 올바른 도구를 선택했는가?
- 파라미터가 정확한가?
- 도구 결과를 바탕으로 적절히 답변했는가?

### Reward 설계

Tool Use 전용 reward는 여러 요소로 구성:

```python
# 1. 형식 보상: <tool_call> 태그가 올바른 JSON인가
# 2. 도구 선택 보상: 적절한 도구를 선택했는가
# 3. 파라미터 보상: 필수 파라미터가 모두 있는가 (CHECK_ARGS)
# 4. 결과 활용 보상: tool_response를 받아 적절히 답변했는가
# 5. 기본 보상: 길이, 반복 패널티, reward model 점수
```

파라미터 검증 (line 66~73):
```python
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    ...
}
```

### 멀티턴 Rollout

일반 GRPO와 다른 점: **환경과의 상호작용이 있는 rollout**

```
1. 모델이 응답 생성 (도구 호출 포함 가능)
2. <tool_call> 감지 → 도구 실행 → <tool_response> 추가
3. 모델이 이어서 생성 (최종 답변)
4. 전체 대화에 대해 reward 계산
5. GRPO/CISPO로 policy 업데이트
```

### GRPO와의 관계

Agentic RL은 GRPO의 확장:
- 같은 prompt에 대해 여러 rollout 생성
- 각 rollout에 tool use 품질 기반 reward 부여
- Group 내 상대 비교로 advantage 계산
- Policy 업데이트

---

## 4. 전체 파이프라인 복습

```
[Week 1-2] 모델 구조 이해
     ↓
[Week 3] Pretrain: 언어 자체를 학습
     ↓
[Week 4] SFT: 지시를 따르는 법 학습 (+ LoRA)
     ↓
[Week 5] DPO: 인간 선호도 반영
     ↓
[Week 6] PPO/GRPO: RL로 정책 최적화
     ↓
[Week 7] MoE/Distillation/YaRN: 모델 확장 기법
     ↓
[Week 8] Agentic RL: 도구 사용 능력 강화  ← 지금 여기
```

모든 단계가 서로 연결되어 있다:
- Pretrain 없이 SFT 불가
- SFT 없이 DPO/PPO 불가 (기본 형식조차 모르는 상태에서 선호 학습은 무의미)
- Tool call은 SFT에서 형식을 배우고, Agentic RL에서 실제 활용 능력을 키움

---

## 이번 주 실습

### 필수

1. `train_agent.py`의 도구 정의 (TOOLS) → 파싱 (parse_tool_calls) → 실행 (MOCK_RESULTS) 흐름을 전부 읽기
2. 모델이 생성한 `<tool_call>` 텍스트가 어떻게 파싱되고 실행되는지 수동으로 추적:
   ```python
   text = '<tool_call>{"name": "calculate_math", "arguments": {"expression": "2+3"}}</tool_call>'
   calls = parse_tool_calls(text)
   result = MOCK_RESULTS[calls[0]["name"]](calls[0]["arguments"])
   print(result)  # {"result": "5"}
   ```

3. Reward 함수를 분석하고, 각 구성 요소가 모델 행동에 어떤 영향을 미치는지 정리

### 선택

4. 새로운 도구를 추가해보기:
   - 예: `search_wikipedia` — 키워드로 위키 검색
   - TOOLS, MOCK_RESULTS, CHECK_ARGS에 각각 추가
   - 모델이 새 도구를 학습하는지 확인

5. Agent RL 실행:
   ```bash
   uv run python trainer/train_agent.py --device mps
   ```

---

## 추천 후속 학습

커리큘럼을 마친 후 확장할 수 있는 방향:

- **MiniMind-V** (https://github.com/jingyaogong/minimind-v): 멀티모달 (Vision + Language)
- **llama.cpp / ollama 배포**: 학습한 모델을 GGUF로 변환해서 로컬 서빙
- **Qwen3 / DeepSeek 코드 읽기**: MiniMind 구조를 이해한 뒤 실제 대규모 모델 코드 비교
- **논문 구현 연습**: 새로운 기법 논문을 읽고 MiniMind에 직접 추가해보기
