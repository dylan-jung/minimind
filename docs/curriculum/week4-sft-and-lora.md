# Week 4: SFT & LoRA

> 목표: Pretrain된 모델을 "대화"가 가능하도록 미세조정하는 과정을 이해한다.
> SFT와 LoRA의 차이를 코드 레벨에서 비교한다.

---

## 공식 학습 자료

공식 Deep Mastery Week 3의 후반부에 해당한다.

| 공식 가이드 | URL |
|-------------|-----|
| Deep Mastery — Week 3: Model Training | https://www.minimind.wiki/en/docs/guide/mastery |

공식 가이드 핵심:
- **SFT** (3시간): `python train_full_sft.py --from_weight pretrain` — pretrain vs SFT 행동 차이 관찰
- **LoRA** (3시간): `python train_lora.py --from_weight full_sft` — PEFT 수학, 도메인 적응
- **성공 기준**: perplexity < 3.0

---

## 1. Supervised Fine-Tuning (SFT)

### 배경 논문

- **Training language models to follow instructions with human feedback** (Ouyang et al., 2022 — InstructGPT)
  - https://arxiv.org/abs/2203.02176
  - Section 3.1 "Supervised fine-tuning (SFT)"
  - Pretrain → SFT → RLHF 3단계 파이프라인의 원본. SFT가 왜 필요한지 설명.

### 코드 비교: `train_pretrain.py` vs `train_full_sft.py`

두 파일의 학습 루프는 거의 동일하다. **차이는 데이터에 있다.**

| 구분 | Pretrain | SFT |
|------|----------|-----|
| 데이터 | `PretrainDataset` | `SFTDataset` |
| 형식 | 일반 텍스트 | 대화 형식 (system/user/assistant) |
| 목적 | 언어 자체를 학습 | 지시 따르기를 학습 |
| Label | 전체 텍스트 | assistant 응답 부분만 (나머지 -100) |

### 핵심 개념

**Label Masking**
```
input:  <|im_start|>user\n질문<|im_end|><|im_start|>assistant\n답변<|im_end|>
label:  [-100, -100, ..., -100,                      답, 변, <|im_end|>]
```
- user 메시지 부분의 label을 -100으로 마스킹 → loss 계산에서 제외
- 모델은 "assistant가 뭐라고 답해야 하는지"만 학습
- 이게 pretrain과 SFT의 **가장 핵심적인 차이**

### 읽기 포인트

1. `dataset/lm_dataset.py`에서 `PretrainDataset`과 `SFTDataset`의 `__getitem__`을 비교
2. SFT 데이터 형식 (`sft_t2t_mini.jsonl`)을 열어서 실제 대화 구조 확인
3. `train_full_sft.py`와 `train_pretrain.py`를 diff로 비교: 거의 import와 Dataset 클래스만 다름

---

## 2. LoRA (Low-Rank Adaptation)

### 논문

- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)
  - https://arxiv.org/abs/2106.09685
  - 전체를 읽되, Section 4 "Our Method"가 핵심.
  - Figure 1의 다이어그램이 전부를 설명.

### 코드: `model/model_lora.py` (전체 66줄)

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        self.A = nn.Linear(in_features, rank, bias=False)   # d x r
        self.B = nn.Linear(rank, out_features, bias=False)   # r x d
        self.A.weight.data.normal_(mean=0.0, std=0.02)       # A는 Gaussian 초기화
        self.B.weight.data.zero_()                            # B는 zero 초기화

    def forward(self, x):
        return self.B(self.A(x))  # x → (d→r→d) 저차원 경로
```

### 핵심 개념

**아이디어**: 원래 가중치 W를 직접 수정하지 않고, 저차원 행렬 BA를 더한다.
```
y = Wx + BAx
    ↑      ↑
  frozen   trainable (r << d)
```

**초기화의 의미**:
- B = 0으로 초기화 → 학습 시작 시 `BA = 0` → 원본 모델과 동일한 출력
- 학습이 진행되면서 BA가 점진적으로 원본 W를 보정

**적용 전략** (line 21~32):
```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        # 정방 행렬인 Linear에만 적용
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(...)
            # forward를 교체: original(x) + lora(x)
            module.forward = lambda x: original_forward(x) + lora(x)
```

**LoRA 합병** (line 56~65):
```python
def merge_lora(model, lora_path, save_path):
    # weight += B @ A  →  추론 시 추가 연산 없음
    state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data)
```

### 코드: `trainer/train_lora.py`

SFT와 거의 동일하되:
- `apply_lora(model, rank=16)`로 LoRA 모듈 주입
- 원본 파라미터는 freeze, LoRA 파라미터만 학습
- `clip_grad_norm_(lora_params, ...)` — LoRA 파라미터에만 gradient clipping
- `save_lora(model, path)` — LoRA 가중치만 별도 저장 (매우 작음)

### Rank에 따른 파라미터 수 비교

```
원본 Linear: d=768 → d=768  →  768 * 768 = 589,824 params
LoRA (r=16): (768*16) + (16*768) = 24,576 params  →  약 4% 수준
```

### 읽기 포인트

1. `apply_lora`에서 정방 행렬에만 적용하는 이유 — Q, K, V, O projection이 정방
2. `forward_with_lora`의 closure 바인딩 트릭 (line 29~30)
3. `merge_lora`에서 `B.weight @ A.weight`로 합치면 왜 추론 시 추가 비용이 0인지
4. `train_lora.py` vs `train_full_sft.py`를 diff로 비교: LoRA 적용/저장 부분만 다름

---

## 이번 주 실습

### 필수

1. SFT 실행:
   ```bash
   # dataset/sft_t2t_mini.jsonl 다운로드 필요
   uv run python trainer/train_full_sft.py --device mps --epochs 2
   ```

2. LoRA 실행:
   ```bash
   uv run python trainer/train_lora.py --device mps --epochs 2
   ```

3. 두 방식의 차이 체감:
   - Full SFT 저장 파일 크기 vs LoRA 저장 파일 크기
   - 학습 속도 차이
   - 생성 품질 차이

### 선택

4. LoRA rank를 바꿔가며 실험 (rank=4, 8, 16, 32):
   - rank가 작을수록 파라미터 적고 빠르지만, 표현력 제한
   - rank가 클수록 Full SFT에 가까워짐

5. `merge_lora` 실행 후 합병된 모델로 추론해보기:
   ```bash
   uv run python scripts/convert_model.py  # LoRA 합병
   uv run python eval_llm.py --load_from ./model --weight lora
   ```
