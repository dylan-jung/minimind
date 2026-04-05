# Week 7: MoE, Knowledge Distillation, YaRN

> 목표: 현대 LLM의 확장 기법 3가지를 이해한다.
> 효율적으로 모델을 키우고(MoE), 줄이고(Distillation), 긴 문맥을 처리(YaRN)하는 방법.

---

## 공식 학습 자료

공식 Deep Mastery Week 4의 **Inference Optimization** 트랙과 연관.

| 공식 가이드 | URL |
|-------------|-----|
| Deep Mastery — Week 4: Advanced Topics | https://www.minimind.wiki/en/docs/guide/mastery |

공식 가이드 핵심:
- KV Cache, Flash Attention, INT8/INT4 양자화
- YaRN은 RoPE 확장의 구현으로, Week 1 Position Encoding 모듈의 심화

---

## 1. Mixture of Experts (MoE)

### 논문

- **Mixtral of Experts** (Jiang et al., 2024)
  - https://arxiv.org/abs/2401.04088
  - Mistral 기반 MoE 모델. Section 2가 핵심.

- (선택) **Switch Transformers: Scaling to Trillion Parameter Models** (Fedus et al., 2022)
  - https://arxiv.org/abs/2101.03961
  - MoE를 Transformer에 적용한 초기 논문. Top-1 routing의 아이디어.

### 코드: `model/model_minimind.py` — `MOEFeedForward` 클래스 (line 147~175)

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        # Router: hidden_state → expert 선택 확률
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # N개의 독립적인 FFN expert
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.num_experts)
        ])

    def forward(self, x):
        # 1. Router가 각 토큰별 expert 점수 계산
        scores = F.softmax(self.gate(x_flat), dim=-1)

        # 2. Top-K expert 선택 (기본 k=1)
        topk_weight, topk_idx = torch.topk(scores, k=num_experts_per_tok)

        # 3. 선택된 expert만 실행
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)
```

### 핵심 개념

**왜 MoE인가**:
- Dense 모델: 모든 토큰이 모든 파라미터를 통과 → 파라미터 수 = 연산량
- MoE: 각 토큰이 일부 expert만 사용 → 총 파라미터는 크지만, 활성 파라미터(연산량)는 적음
- MiniMind: 198M total / 64M active (A64M)

**Router**:
- 각 토큰의 hidden state를 보고 어떤 expert에 보낼지 결정
- `softmax(gate(x))` → 확률 분포 → top-k 선택

**Load Balancing Loss** (line 170~172):
```python
load = F.one_hot(topk_idx, num_experts).float().mean(0)
self.aux_loss = (load * scores.mean(0)).sum() * num_experts * router_aux_loss_coef
```
- **문제**: router가 특정 expert만 선호하면 나머지 expert는 학습이 안 됨 (routing collapse)
- **해결**: 모든 expert가 골고루 사용되도록 보조 loss 추가
- `load`: 각 expert의 실제 사용 비율 / `scores.mean(0)`: 각 expert의 평균 확률
- 둘의 곱을 최소화 → 고르게 분포하도록 유도

**Dead expert 방지** (line 168~169):
```python
elif self.training:
    y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
```
- 아무 토큰도 할당되지 않은 expert의 gradient가 0이 되는 문제
- `0 *`을 곱해서 값은 안 바뀌지만 computation graph에는 포함 → gradient 흐름 유지

### MiniMind 설정

```python
num_experts = 4          # expert 수
num_experts_per_tok = 1  # 토큰당 활성 expert 수
norm_topk_prob = True    # top-k 가중치 정규화
router_aux_loss_coef = 5e-4  # load balancing loss 계수
```

### 읽기 포인트

1. `MiniMindBlock` (line 183)에서 `use_moe` 설정에 따라 `FeedForward` 또는 `MOEFeedForward` 선택
2. `MiniMindModel.forward` (line 226)에서 모든 layer의 `aux_loss`를 합산
3. `train_pretrain.py`에서 `loss = res.loss + res.aux_loss` — main loss와 aux loss를 합쳐서 학습

---

## 2. Knowledge Distillation

### 논문

- **Distilling the Knowledge in a Neural Network** (Hinton et al., 2015)
  - https://arxiv.org/abs/1503.02531
  - Section 2 "Distillation"이 핵심. Temperature의 역할을 이해하는 게 전부.

### 코드: `trainer/train_distillation.py` — `distillation_loss` (line 24~35)

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    # Teacher의 soft label (T로 나눠서 분포를 부드럽게)
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Student의 soft prediction
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL Divergence * T^2
    kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return (temperature ** 2) * kl
```

### 핵심 개념

**아이디어**: 큰 모델(teacher)의 "부드러운 확률 분포"를 작은 모델(student)이 모방.

**Temperature의 역할**:
```
T=1:  teacher가 "정답" 토큰에 99% 확률 → 다른 토큰 정보가 거의 없음
T=5:  teacher가 "정답" 40%, "유사어" 30%, ... → 풍부한 "dark knowledge" 전달
```
- 높은 T → soft label이 더 부드러움 → 토큰 간 관계 정보가 드러남
- `T^2` 보정: temperature로 스케일이 바뀐 것을 원래 크기로 맞춤

**학습 구조** (line 38~):
```python
# Teacher는 frozen
teacher_model.eval()
teacher_model.requires_grad_(False)

# Student forward → Teacher forward (no_grad) → KL divergence
student_logits = model(input_ids).logits
teacher_logits = teacher_model(input_ids).logits
loss = alpha * ce_loss + (1 - alpha) * distillation_loss(student, teacher, T)
```
- `alpha`: CE loss와 distillation loss의 비율. 보통 0.0~0.5

### 읽기 포인트

1. Teacher와 student의 모델 크기가 다를 수 있음 — MiniMind에서는 같은 구조의 다른 config 사용
2. `loss_mask`로 padding 부분 제외 (line 50)
3. Teacher는 SFT까지 완료된 모델, student는 scratch부터 학습

---

## 3. YaRN (장문 외추)

### 논문

- **YaRN: Efficient Context Window Extension of Large Language Models** (Peng et al., 2023)
  - https://arxiv.org/abs/2309.00071
  - Section 3 "Method"의 NTK-by-parts interpolation이 핵심.

- (배경) **Extending Context Window of Large Language Models via Positional Interpolation** (Chen et al., 2023)
  - https://arxiv.org/abs/2306.15595
  - 선형 보간(PI)의 원본 논문. YaRN이 이것을 개선.

### 코드: `model/model_minimind.py` — `precompute_freqs_cis` (line 63~72)

```python
if rope_scaling is not None:
    # YaRN: 주파수별로 다른 스케일링 적용
    # 저주파(긴 패턴) → 많이 압축, 고주파(짧은 패턴) → 거의 유지
    inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
    low = max(math.floor(inv_dim(beta_fast)), 0)
    high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

    # linear ramp: low~high 구간에서 0→1로 선형 변환
    ramp = torch.clamp(
        (torch.arange(dim // 2) - low) / max(high - low, 0.001), 0, 1
    )
    # f'(i) = f(i) * ((1 - ramp) + ramp / factor)
    freqs = freqs * (1 - ramp + ramp / factor)
```

### 핵심 개념

**문제**: RoPE는 학습 시 본 최대 길이(예: 2048)까지만 잘 동작. 그 이상에서 성능 급락.

**PI (Position Interpolation)**: 모든 주파수를 동일하게 압축 → 고주파 정보 손실

**YaRN의 개선**:
- 주파수를 3구간으로 나눔:
  - **고주파** (짧은 패턴, 인접 토큰 관계): 변경 없음
  - **저주파** (긴 패턴, 먼 토큰 관계): factor로 나눠 압축
  - **중간 구간**: linear ramp으로 부드럽게 전환

**설정값** (MiniMind):
```python
rope_scaling = {
    "original_max_position_embeddings": 2048,  # 원래 학습 길이
    "factor": 16,                              # 16배 확장 → 32K까지
    "beta_fast": 32,                           # 고주파 경계
    "beta_slow": 1,                            # 저주파 경계
    "attention_factor": 1.0,                   # attention logit 스케일
}
```

### 읽기 포인트

1. `inference_rope_scaling=True`일 때만 YaRN 활성화 (추론 시에만)
2. `ramp` 텐서를 시각화해서 어떤 차원이 압축되는지 확인
3. `beta_fast`, `beta_slow`가 고주파/저주파 경계를 결정하는 원리

---

## 이번 주 실습

### 필수

1. `MOEFeedForward.forward`를 한 줄씩 읽고, 토큰 하나가 어떤 경로로 처리되는지 추적
2. `distillation_loss`의 temperature를 1, 3, 5, 10으로 바꾸며 teacher_probs가 어떻게 변하는지 시각화:
   ```python
   logits = torch.tensor([5.0, 2.0, 1.0, 0.5, 0.1])
   for T in [1, 3, 5, 10]:
       probs = F.softmax(logits / T, dim=-1)
       print(f"T={T:2d}: {probs.numpy().round(3)}")
   ```

3. YaRN ramp 시각화:
   ```python
   dim, low, high = 48, 5, 20
   ramp = torch.clamp((torch.arange(dim) - low) / max(high - low, 0.001), 0, 1)
   # plt.plot(ramp) → 어떤 차원이 압축되는지 확인
   ```

### 선택

4. MoE 모델로 pretrain 돌려보기:
   ```bash
   uv run python trainer/train_pretrain.py --device mps --use_moe True
   ```
   - Dense 대비 loss curve 비교
   - `aux_loss`가 어떻게 변하는지 관찰

5. Distillation 실행 (teacher 모델 필요):
   ```bash
   uv run python trainer/train_distillation.py --device mps
   ```
