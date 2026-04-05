# Week 12: 데이터 파이프라인, 평가 & 서빙

> 목표: 학습 전(데이터), 학습 후(평가), 배포(서빙)까지 MiniMind 코드로 실전 파이프라인을 완성한다.
> 이 주를 마치면 "데이터 준비 → 학습 → 평가 → 서빙"의 전체 사이클을 경험하게 된다.

---

## 공식 학습 자료

| 공식 가이드 | URL |
|-------------|-----|
| Deep Mastery — Week 2: Data Preparation | https://www.minimind.wiki/en/docs/guide/mastery |

공식 가이드 핵심:
- **Data Cleaning/Preprocessing** (4시간): `dataset/lm_dataset.py` 분석
- **Format Conversion** (2시간): pretrain / SFT / DPO 3가지 형식

---

## Part 1: 데이터 파이프라인

### 1-1. 데이터셋 클래스 구조

코드: `dataset/lm_dataset.py` — 5가지 Dataset 클래스

| 클래스 | 용도 | 데이터 형식 | 사용처 |
|--------|------|------------|--------|
| `PretrainDataset` | 사전학습 | `{"text": "..."}` | `train_pretrain.py` |
| `SFTDataset` | 지도 미세조정 | `{"conversations": [...]}` | `train_full_sft.py`, `train_lora.py` |
| `DPODataset` | 선호 최적화 | `{"chosen": [...], "rejected": [...]}` | `train_dpo.py` |
| `RLAIFDataset` | RL 학습 | `{"conversations": [...]}` | `train_grpo.py`, `train_ppo.py` |
| `AgentRLDataset` | Agent RL | `{"conversations": [...], "gt": ...}` | `train_agent.py` |

### 1-2. PretrainDataset — 가장 단순한 형태

```python
# dataset/lm_dataset.py line 37~55
def __getitem__(self, index):
    sample = self.samples[index]
    tokens = self.tokenizer(str(sample['text']), max_length=self.max_length - 2, truncation=True).input_ids
    tokens = [bos] + tokens + [eos]                         # 시작/끝 토큰 추가
    input_ids = tokens + [pad] * (self.max_length - len(tokens))  # 패딩
    labels = input_ids.clone()
    labels[input_ids == pad] = -100                          # 패딩은 loss에서 제외
    return input_ids, labels
```

포인트:
- BOS/EOS 토큰으로 문장 경계를 알림
- 패딩 → `-100`으로 마스킹 → `cross_entropy(ignore_index=-100)`

### 1-3. SFTDataset — Label Masking의 핵심

```python
# dataset/lm_dataset.py line 88~104
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)  # 전부 -100으로 시작
    # bos_id = "<|im_start|>assistant\n" 패턴을 찾아
    # eos_id = "<|im_end|>\n" 패턴까지의 구간만 labels에 복사
    # → assistant 응답 부분만 학습 대상
```

```
input:  <|im_start|>user\n질문<|im_end|><|im_start|>assistant\n답변<|im_end|>
label:  [-100, -100, ..., -100,           답, 변, <|im_end|>]
                                           ↑ 여기만 loss 계산
```

이것이 **Pretrain과 SFT의 가장 핵심적인 차이**. 코드로 확인하기:
- `bos_id`: `<|im_start|>assistant\n` 토큰 시퀀스
- `eos_id`: `<|im_end|>\n` 토큰 시퀀스
- 이 두 마커 사이의 토큰만 label로 복사

### 1-4. DPODataset — Chosen/Rejected 쌍

```python
# dataset/lm_dataset.py line 122~192
def __getitem__(self, index):
    # chosen과 rejected를 각각 chat template으로 변환
    chosen_prompt = tokenizer.apply_chat_template(chosen, ...)
    rejected_prompt = tokenizer.apply_chat_template(rejected, ...)
    # generate_loss_mask: SFT와 동일한 로직으로 assistant 부분만 마스킹
```

### 1-5. 데이터 전처리 함수

**`pre_processing_chat`** (line 9~29):
```python
# 20% 확률로 system prompt 추가 — 다양성 확보
if random.random() < add_system_ratio:
    return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
```

**`post_processing_chat`** (line 31~35):
```python
# 80% 확률로 빈 <think> 태그 제거 — 불필요한 사고 태그 학습 방지
if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
    prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
```

### 1-6. 데이터 품질 & 중복 제거

MiniMind가 사용하는 라이브러리 (`requirements.txt` / `pyproject.toml`):
- **`datasketch`**: MinHash LSH 기반 근사 중복 탐지 — 대규모 데이터에서 유사 문서 제거
- **`simhash`**: SimHash 기반 근사 중복 탐지 — 빠른 fingerprint 비교
- **`jieba`**: 중국어 형태소 분석 — 토크나이저 학습 전 분절

### 논문 (선택)

- **Deduplicating Training Data Makes Language Models Better** (Lee et al., 2022)
  - https://arxiv.org/abs/2107.06499
  - 중복 제거가 모델 품질에 미치는 영향을 정량적으로 분석

---

## Part 2: 평가 (Evaluation)

### 2-1. CLI 추론 & 정성 평가

코드: `eval_llm.py`

```python
# eval_llm.py line 63~91
model, tokenizer = init_model(args)
# 자동 테스트 모드: 8개 기본 prompt로 생성 결과 확인
# 수동 입력 모드: 직접 대화하며 품질 체크

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generated_ids = model.generate(
    inputs=inputs["input_ids"],
    max_new_tokens=args.max_new_tokens,
    temperature=args.temperature,
    top_p=args.top_p,
    streamer=streamer,  # 토큰 단위 스트리밍 출력
)
# 속도 측정: tokens/s
```

주요 옵션:
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--weight` | 어떤 체크포인트를 로드할지 | `full_sft` |
| `--open_thinking` | `<think>` 사고 과정 표시 | 0 |
| `--historys` | 멀티턴 대화 히스토리 유지 | 0 |
| `--inference_rope_scaling` | YaRN 장문 외추 활성화 | False |
| `--lora_weight` | LoRA 가중치 로드 | None |

### 2-2. 각 학습 단계별 평가 비교

```bash
# Pretrain 모델: 문장 완성만 가능, 대화 불가
uv run python eval_llm.py --load_from ./model --weight pretrain

# SFT 모델: 기본 대화 가능
uv run python eval_llm.py --load_from ./model --weight full_sft

# DPO 모델: 더 정제된 답변
uv run python eval_llm.py --load_from ./model --weight dpo

# GRPO 모델: 추론 능력 강화
uv run python eval_llm.py --load_from ./model --weight grpo --open_thinking 1

# LoRA 모델: 특정 도메인 적응
uv run python eval_llm.py --load_from ./model --weight full_sft --lora_weight lora_identity
```

### 2-3. 벤치마크 평가

README에 언급된 벤치마크:
- **C-Eval**: 중국어 다지선다 (인문, 사회, STEM 등 52과목)
- **C-MMLU**: 중국어 MMLU
- **OpenBookQA**: 상식 추론

### 2-4. 속도 측정

```python
# eval_llm.py line 90~91
gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
print(f'[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s')
```

- M4 Pro MPS에서 fp16 기준 예상 속도를 직접 측정해보기
- `--max_new_tokens`를 바꿔가며 긴 생성과 짧은 생성의 throughput 비교

---

## Part 3: 모델 변환

코드: `scripts/convert_model.py`

### 3-1. PyTorch ↔ Transformers 변환

```python
# MiniMind PyTorch 가중치 → Transformers 포맷
convert_torch2transformers_minimind(torch_path, transformers_path)

# MiniMind → Qwen3 호환 Transformers 포맷 (llama.cpp, vllm, ollama 호환)
convert_torch2transformers(torch_path, transformers_path)

# Transformers → PyTorch (역변환)
convert_transformers2torch(transformers_path, torch_path)
```

### 3-2. MiniMind → Qwen3 변환의 의미

```python
# convert_model.py line 40~96
# MiniMind config를 Qwen3Config로 매핑
qwen_config = Qwen3Config(
    vocab_size=lm_config.vocab_size,
    hidden_size=lm_config.hidden_size,
    num_attention_heads=lm_config.num_attention_heads,
    num_key_value_heads=lm_config.num_key_value_heads,
    ...
)
qwen_model = Qwen3ForCausalLM(qwen_config)
qwen_model.load_state_dict(state_dict, strict=True)  # 가중치 직접 로드!
```

**핵심**: MiniMind와 Qwen3의 구조가 동일하기 때문에, 가중치를 그대로 옮길 수 있다.
이렇게 변환하면:
- `llama.cpp`로 GGUF 변환 → CPU에서 고속 추론
- `vllm`으로 서빙 → 프로덕션 배포
- `ollama`로 로컬 실행 → `ollama run jingyaogong/minimind-3`

### 3-3. LoRA 합병

```python
# convert_model.py line 105~112
convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path)
# → model_lora.py의 merge_lora() 호출
# → weight += B @ A → 추론 시 추가 연산 없음
```

---

## Part 4: 서빙 (OpenAI API 호환)

코드: `scripts/serve_openai_api.py`

### 4-1. FastAPI 서버 구조

```python
# serve_openai_api.py
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # OpenAI API 형식의 요청을 받아 MiniMind로 생성
```

### 4-2. ChatRequest 스키마

```python
class ChatRequest(BaseModel):
    model: str
    messages: list           # [{"role": "user", "content": "..."}]
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = True
    tools: list = []         # MCP-style tool 정의
    open_thinking: bool = False  # <think> 사고 과정 활성화
```

### 4-3. 스트리밍 생성

```python
# serve_openai_api.py line 105~168
def generate_stream_response(...):
    # 별도 Thread에서 model.generate() 실행
    # Queue를 통해 토큰 단위로 스트리밍
    # <think>...</think> 구간은 reasoning_content로 분리
    # <tool_call>...</tool_call> 구간은 tool_calls로 파싱
    yield json.dumps({"choices": [{"delta": {"content": "..."}}]})
```

**핵심 기능**:
- **SSE 스트리밍**: `text/event-stream`으로 토큰 단위 실시간 전송
- **reasoning_content 분리**: `<think>` 태그를 별도 필드로 파싱 → 클라이언트에서 사고 과정 표시
- **tool_calls 파싱**: `<tool_call>` 태그를 OpenAI 형식의 tool_calls로 변환
- 비스트리밍 모드도 지원 (`stream: false`)

### 4-4. `parse_response` — 출력 파싱

```python
# serve_openai_api.py line 83~102
def parse_response(text):
    # 1. <think>...</think> 추출 → reasoning_content
    # 2. <tool_call>...</tool_call> 추출 → tool_calls (OpenAI 형식)
    # 3. 나머지 텍스트 → content
    return content, reasoning_content, tool_calls
```

### 4-5. 호환 클라이언트

이 서버는 OpenAI API 호환이므로:
- **OpenAI Python SDK**: `openai.ChatCompletion.create(base_url="http://localhost:8998/v1")`
- **FastGPT / Open-WebUI**: base URL만 변경하면 연결
- **curl**: `curl -X POST http://localhost:8998/v1/chat/completions -d '...'`

---

## Part 5: WebUI 데모

코드: `scripts/web_demo.py`

- **Streamlit 기반** 채팅 인터페이스
- `<think>` 사고 과정 표시 지원
- 도구 선택 UI & 멀티턴 Tool Call 지원
- `scripts/` 디렉토리 내의 모델 폴더를 자동 스캔

```bash
# 실행
cp -r minimind-3 ./scripts/minimind-3
cd scripts && uv run streamlit run web_demo.py
```

---

## 이번 주 실습

### 필수

1. `dataset/lm_dataset.py` 전체를 읽고, 5개 Dataset 클래스의 `__getitem__`을 비교:
   - 특히 `SFTDataset.generate_labels`의 label masking 로직을 한 줄씩 추적

2. 디버그 코드 활성화 (line 114~118의 주석 해제):
   ```python
   # 각 토큰의 input → label 매핑을 직접 확인
   for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
       print(f"{i:3d}: X={tokenizer.decode([x])!r:16s} ---> Y=... label={y}")
   ```

3. 각 학습 단계의 모델을 로드해서 동일 질문에 대한 답변 비교:
   ```bash
   uv run python eval_llm.py --load_from ./model --weight pretrain
   uv run python eval_llm.py --load_from ./model --weight full_sft
   # pretrain은 문장 이어쓰기, sft는 대화 형식 응답
   ```

4. OpenAI API 서버 실행:
   ```bash
   cd scripts
   uv run python serve_openai_api.py --device mps --load_from ../minimind-3
   ```
   ```python
   # 클라이언트 테스트
   from openai import OpenAI
   client = OpenAI(base_url="http://localhost:8998/v1", api_key="none")
   resp = client.chat.completions.create(
       model="minimind",
       messages=[{"role": "user", "content": "안녕?"}],
       stream=True
   )
   for chunk in resp:
       print(chunk.choices[0].delta.content or "", end="")
   ```

### 선택

5. 모델 변환 실행:
   ```bash
   cd scripts
   uv run python convert_model.py  # MiniMind → Qwen3 Transformers 포맷
   ```

6. Streamlit 데모 실행해서 사고 과정 + Tool Call 체험

---

## 셀프 체크

- [ ] PretrainDataset과 SFTDataset의 label 차이를 코드로 설명할 수 있는가
- [ ] SFTDataset의 `generate_labels`가 어떤 토큰을 -100으로 마스킹하는지 설명할 수 있는가
- [ ] `pre_processing_chat`에서 system prompt를 확률적으로 추가하는 이유를 설명할 수 있는가
- [ ] pretrain 모델 vs SFT 모델의 출력 차이를 직접 확인했는가
- [ ] OpenAI API 호환 서버를 로컬에서 실행하고 요청을 보낼 수 있는가
- [ ] MiniMind → Qwen3 변환이 가능한 이유를 구조적으로 설명할 수 있는가
- [ ] `parse_response`에서 reasoning_content와 tool_calls를 어떻게 분리하는지 설명할 수 있는가

---

## 전체 커리큘럼 회고

```
Week 1-2:   모델 구조를 이해했다
Week 3:     모델이 언어를 학습하는 원리를 배웠다
Week 4:     대화 능력을 부여하는 방법을 배웠다
Week 5-6:   인간 선호를 반영하는 방법을 배웠다
Week 7:     모델을 확장하는 기법을 배웠다
Week 8:     모델이 도구를 사용하도록 학습시켰다
Week 9:     추론 능력이 RL로 출현하는 원리를 배웠다
Week 10:    코드 에이전트와 벤치마크를 이해했다
Week 11:    프로토콜, 멀티에이전트, 최신 모델을 조망했다
Week 12:    데이터 → 학습 → 평가 → 서빙 전체 사이클을 완성했다
```

12주를 마치면 MiniMind의 모든 코드를 읽었고,
40편의 논문을 통해 이론적 배경을 갖추었으며,
직접 학습 → 평가 → 배포까지 경험한 상태가 된다.
