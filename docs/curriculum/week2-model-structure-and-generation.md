# Week 2: 모델 전체 구조 & 추론

> 목표: Transformer 블록이 쌓여서 전체 모델이 되는 구조를 이해하고, 텍스트 생성 과정을 파악한다.
> `model/model_minimind.py` 후반부 집중.

---

## 공식 학습 자료

| 공식 모듈 | URL | 소요 시간 |
|-----------|-----|-----------|
| FeedForward | https://www.minimind.wiki/en/modules/01-foundation/04-feedforward/teaching | 1시간 |
| Architecture (Transformer Block) | https://www.minimind.wiki/en/modules/02-architecture/ | 0.5시간 |

---

## 1. SwiGLU Feed-Forward Network

### 공식 wiki

- **모듈**: `04-feedforward` (1시간)
- **핵심 실험**: `exp1_feedforward.py` — 차원 변화, activation 비교, gating 시각화
- **주방 비유**:
  - 입력: 재료 (768차원)
  - 확장: 재료를 잘게 써는 과정 (768 → 2048)
  - 활성화: 불로 조리 (비선형 변환)
  - 압축: 플레이팅 (2048 → 768)
- **왜 확장 후 압축?**: 768→768 직접 변환은 선형에 불과. 고차원에서 비선형을 거쳐야 복잡한 결정 경계 생성 가능

### 논문

- **GLU Variants Improve Transformer** (Shazeer, 2020)
  - https://arxiv.org/abs/2002.05202

### 코드: `model/model_minimind.py` — `FeedForward` 클래스 (line 135~145)

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### 핵심 개념

**기존 FFN vs SwiGLU** (공식 wiki 기준):
```
기존:   FFN(x) = W₂ · ReLU(W₁ · x)           — 2개 Linear
SwiGLU: SwiGLU(x) = W_down · (SiLU(W_gate · x) ⊙ W_up · x)  — 3개 Linear
```

**SiLU (Swish) 활성화 함수**:
```
SiLU(x) = x · σ(x) = x / (1 + e^(-x))
```
- 매끄럽고, 모든 곳에서 미분 가능, 비단조 (음수도 보존) — ReLU보다 유연

**GLU Variants 비교** (공식 wiki):

| Variant | 수식 | Gating |
|---------|------|--------|
| GLU | σ(W₁x) ⊙ W₂x | sigmoid |
| ReGLU | ReLU(W₁x) ⊙ W₂x | ReLU |
| GeGLU | GELU(W₁x) ⊙ W₂x | GELU |
| **SwiGLU** | **SiLU(W₁x) ⊙ W₂x** | **SiLU** |

**3개 projection의 역할** (공식 wiki):
- `gate_proj`: gate 신호 생성 (얼마나 통과시킬지)
- `up_proj`: value 신호 생성 (실제 정보)
- `down_proj`: 고차원 → 원래 차원으로 압축

**Attention vs FFN 역할 분담** (공식 wiki):
- Attention: 전역 정보 교환 (팀 회의)
- FeedForward: 위치별 독립 특징 변환 (개인 사고)

### 읽기 포인트

1. `gate_proj`와 `up_proj`가 같은 입출력 차원인데 역할이 다름
2. `bias=False` — 현대 LLM에서 bias를 제거하는 트렌드
3. `intermediate_size = ceil(hidden_size * pi / 64) * 64` — pi 비율의 경험적 선택

---

## 2. Grouped-Query Attention (GQA)

### 논문

- **GQA: Training Generalized Multi-Query Transformer Models** (Ainslie et al., 2023)
  - https://arxiv.org/abs/2305.13245

### 코드: `model/model_minimind.py` — `repeat_kv` 함수 (line 85~88)

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim))
```

### 핵심 개념 (공식 wiki 기준)

```
MHA:  Q=8, K=8, V=8  →  메모리 100%
GQA:  Q=8, K=2, V=2  →  메모리 ~25%   ← MiniMind
MQA:  Q=8, K=1, V=1  →  메모리 ~12%   (너무 극단적)
```

- `repeat_kv`: 2개의 KV head를 8개로 복제해서 Q head 수에 맞춤
- KV-cache 크기가 head 수에 비례 → KV head를 줄이면 추론 메모리 절약

---

## 3. Transformer Block & 전체 모델

### 공식 wiki

- **모듈**: `02-architecture` — Transformer Block Assembly
- **핵심**: Pre-LN 구조에서 residual path를 통한 데이터 흐름 이해

### 코드: `MiniMindBlock` (line 177~193), `MiniMindModel` (line 195~227)

```python
class MiniMindBlock(nn.Module):
    def forward(self, hidden_states, ...):
        # Pre-Norm + Attention + Residual
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), ...)
        hidden_states += residual

        # Pre-Norm + FFN + Residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states
```

```python
class MiniMindModel(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)  # 토큰 → 벡터
        self.layers = nn.ModuleList([MiniMindBlock(...) for l in range(num_hidden_layers)])
        self.norm = RMSNorm(hidden_size)                           # 최종 정규화
```

### 핵심 개념

- **Pre-Norm**: `x + Attn(Norm(x))` — 학습 안정성이 Post-Norm보다 우수 (Week 1 실험에서 확인)
- **Residual Connection**: 깊은 네트워크에서 gradient 흐름 보장
- **Weight Tying**: `embed_tokens.weight = lm_head.weight` (line 236) — 임베딩과 출력 layer 가중치 공유
- **Loss**: `F.cross_entropy(logits[:-1], labels[1:])` — next-token prediction의 본질

---

## 4. 텍스트 생성 (Inference)

### 코드: `MiniMindForCausalLM.generate` (line 250~280)

```python
for _ in range(max_new_tokens):
    outputs = self.forward(input_ids[:, past_len:], ..., use_cache=True)
    logits = outputs.logits[:, -1, :] / temperature

    # Top-K: 상위 k개만 남기고 나머지 -inf
    logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

    # Top-P (Nucleus Sampling): 누적 확률 p 이하만 남김
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p

    # 샘플링
    next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
```

### 핵심 개념

- **Temperature**: logits/T → T<1이면 확정적, T>1이면 다양
- **Top-K**: 확률 상위 K개만 후보
- **Top-P (Nucleus)**: 누적 확률 P까지만 후보
- **KV-Cache**: 이전 K, V 저장 → 새 토큰 1개만 계산 → O(n) → O(1)
- **Repetition Penalty**: 이미 생성된 토큰의 logit을 penalty로 나눔

### 논문 (선택)

- **The Curious Case of Neural Text Degeneration** (Holtzman et al., 2020)
  - https://arxiv.org/abs/1904.09751
  - Top-P sampling 원본 논문.

---

## 이번 주 실습

### 필수: 공식 wiki

1. **FeedForward 모듈** 실험 (1시간):
   - https://www.minimind.wiki/en/modules/01-foundation/04-feedforward/teaching
   - `exp1_feedforward.py` 실행

2. **Architecture 모듈** 읽기 (0.5시간):
   - https://www.minimind.wiki/en/modules/02-architecture/

### 필수: MiniMind 코드

3. `model/model_minimind.py` line 85~280을 읽기

4. 모델 구조 확인:
   ```python
   from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

   config = MiniMindConfig(hidden_size=512, num_hidden_layers=4)
   model = MiniMindForCausalLM(config)
   total = sum(p.numel() for p in model.parameters())
   print(f"Total params: {total / 1e6:.2f}M")

   for name, param in model.named_parameters():
       print(f"{name:60s} {str(param.shape):20s} {param.numel():>10,}")
   ```

### 셀프 체크 (공식 wiki 기준)

- [ ] SwiGLU의 3개 projection 역할을 각각 설명할 수 있는가
- [ ] SiLU vs ReLU 차이를 설명할 수 있는가
- [ ] Gating 메커니즘의 목적을 설명할 수 있는가
- [ ] Attention과 FFN의 역할 분담을 설명할 수 있는가
- [ ] Pre-LN Transformer block의 데이터 흐름을 스케치할 수 있는가
- [ ] SwiGLU를 참고 없이 구현할 수 있는가

---

## 참고 자료

- MiniMind 공식 Systematic Study: https://www.minimind.wiki/en/docs/guide/systematic
- How GPT3 Works (Jay Alammar): https://jalammar.github.io/how-gpt3-works-visualizations-animations/
