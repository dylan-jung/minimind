# Week 5: DPO (Direct Preference Optimization)

> 목표: 인간의 선호도를 모델에 반영하는 RLHF의 원리를 이해하고,
> PPO 없이 직접 최적화하는 DPO를 코드 레벨에서 파악한다.

---

## 공식 학습 자료

공식 Deep Mastery Week 4의 RLHF/RLAIF 선택 과정에 해당한다.

| 공식 가이드 | URL |
|-------------|-----|
| Deep Mastery — Week 4: Advanced Topics | https://www.minimind.wiki/en/docs/guide/mastery |

공식 가이드 핵심:
- Week 4에서 3가지 선택 과정 중 **RLHF/RLAIF** 트랙
- DPO → PPO/GRPO 순서로 학습 권장

---

## 배경: 왜 RLHF가 필요한가

Pretrain → SFT까지만 하면 모델은 "답변을 생성"할 수 있지만:
- 유해한 답변도 자신있게 생성
- 여러 가능한 답변 중 "더 좋은" 것을 구분하지 못함
- 인간의 선호를 반영하지 않음

**RLHF 파이프라인** (InstructGPT):
```
Step 1: SFT
Step 2: Reward Model 학습 (인간 선호 데이터로)
Step 3: PPO로 policy 최적화 (reward model을 보상으로)
```

DPO는 Step 2, 3을 하나로 합친다.

---

## 1. RLHF 배경

### 논문

- **Training language models to follow instructions with human feedback** (Ouyang et al., 2022 — InstructGPT)
  - https://arxiv.org/abs/2203.02176
  - RLHF의 전체 파이프라인. Section 3을 전부 읽기.
  - 이 논문을 먼저 이해해야 DPO가 왜 등장했는지 맥락이 잡힘.

- **Learning to summarize from human feedback** (Stiennon et al., 2020)
  - https://arxiv.org/abs/2009.01325
  - RLHF를 요약 task에 적용한 초기 논문. reward model 학습 방법을 자세히 설명.

---

## 2. DPO

### 논문

- **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** (Rafailov et al., 2023)
  - https://arxiv.org/abs/2305.18290
  - Section 4 "Direct Preference Optimization"이 핵심.
  - 특히 Eq. 7 (DPO objective)을 이해하면 코드가 바로 읽힘.

### 핵심 아이디어

PPO 방식:
```
1. Reward model r(x, y) 학습
2. Policy π를 r(x,y)로 강화학습 → 복잡하고 불안정
```

DPO 방식:
```
Reward model 없이 chosen/rejected 쌍에서 직접 최적화
→ 수학적으로 PPO와 동일한 최적해에 도달함을 증명
```

### 코드: `trainer/train_dpo.py`

**Log probability 계산** (line 24~30):
```python
def logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token
```
- 모델 출력에서 실제 정답 토큰의 log 확률만 추출
- `torch.gather`: vocab 전체가 아닌, 해당 토큰 위치의 확률만 가져옴

**DPO Loss 함수** (line 33~49):
```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # 시퀀스 전체의 log prob 합산
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # chosen과 rejected 분리
    chosen_ref = ref_log_probs[:batch_size // 2]
    reject_ref = ref_log_probs[batch_size // 2:]
    chosen_policy = policy_log_probs[:batch_size // 2]
    reject_policy = policy_log_probs[batch_size // 2:]

    # DPO objective
    pi_logratios = chosen_policy - reject_policy
    ref_logratios = chosen_ref - reject_ref
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

### 수식 ↔ 코드 매핑

논문 Eq. 7:
```
L_DPO = -E[log σ(β · (log π(y_w|x)/π(y_l|x) - log π_ref(y_w|x)/π_ref(y_l|x)))]
```

코드에서:
```
pi_logratios   = log π(y_w|x) - log π(y_l|x)         ← policy의 chosen vs rejected
ref_logratios  = log π_ref(y_w|x) - log π_ref(y_l|x)  ← reference의 chosen vs rejected
logits         = pi_logratios - ref_logratios           ← 차이의 차이
loss           = -logsigmoid(β * logits)                ← Bradley-Terry 모델
```

### 학습 루프의 구조 (line 52~)

```python
# 1. chosen과 rejected를 concat해서 한번에 forward
x = torch.cat([x_chosen, x_rejected], dim=0)

# 2. Reference model (frozen)로 log prob 계산
with torch.no_grad():
    ref_logits = ref_model(x).logits
ref_log_probs = logits_to_log_probs(ref_logits, y)

# 3. Policy model (학습 중)로 log prob 계산
policy_logits = model(x).logits
policy_log_probs = logits_to_log_probs(policy_logits, y)

# 4. DPO loss
loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.1)
```

### 핵심 개념

- **Reference Model**: SFT 완료된 모델의 복사본. frozen 상태. policy가 너무 멀리 벗어나지 않도록 anchor 역할
- **Beta (β)**: KL penalty 강도. 클수록 reference에 가깝게 유지. 보통 0.1~0.5
- **Chosen/Rejected pair**: 같은 질문에 대해 "좋은 답변"과 "나쁜 답변"이 쌍으로 제공
- **Mask**: padding 토큰이나 prompt 부분을 log prob 계산에서 제외

---

## DPO 데이터 형식

```json
{
  "prompt": "질문 내용",
  "chosen": "좋은 답변",
  "rejected": "나쁜 답변"
}
```

---

## 이번 주 실습

### 필수

1. `trainer/train_dpo.py` 전체를 읽기 — 특히 `dpo_loss` 함수를 손으로 수식 전개해보기
2. 논문 Eq. 7과 코드를 한 줄씩 대응시켜보기
3. DPO 실행:
   ```bash
   # DPO 데이터 다운로드 필요
   uv run python trainer/train_dpo.py --device mps --epochs 1
   ```

### 선택

4. beta 값을 바꿔가며 실험 (0.05, 0.1, 0.5):
   - beta가 작으면: reference에서 더 자유롭게 벗어남
   - beta가 크면: reference에 가깝게 유지
5. DPO 전후 모델의 생성 결과를 비교해보기
6. `logits_to_log_probs`에서 `torch.gather`가 하는 일을 시각화:
   ```python
   # vocab_size = 6400일 때
   # log_probs shape: (batch, seq_len, 6400)
   # labels shape:    (batch, seq_len)
   # gather로 각 위치의 정답 토큰 확률만 추출 → (batch, seq_len)
   ```
