# Week 1: Transformer 핵심 구조

> 목표: `model/model_minimind.py`의 전반부를 완전히 이해한다.
> 이 파일 하나에 현대 LLM 구조의 핵심이 모두 들어있다.

---

## 공식 학습 자료

이번 주는 MiniMind 공식 wiki의 Foundation 모듈과 병행한다.

| 공식 모듈 | URL | 소요 시간 |
|-----------|-----|-----------|
| Normalization | https://www.minimind.wiki/en/modules/01-foundation/01-normalization/teaching | 1시간 |
| Position Encoding | https://www.minimind.wiki/en/modules/01-foundation/02-position-encoding/teaching | 1.5시간 |
| Attention | https://www.minimind.wiki/en/modules/01-foundation/03-attention/teaching | 2시간 |

> 공식 Quick Start (30분)를 먼저 해보는 것을 추천:
> https://www.minimind.wiki/en/docs/guide/quick-start

---

## 1. RMSNorm — 정규화

### 공식 wiki

- **모듈**: `01-normalization` (1시간)
- **핵심 실험**: `exp1_gradient_vanishing.py` — 정규화 없이 8 layer를 쌓으면 activation std가 1.04 → 0.016으로 붕괴하는 것을 직접 확인
- **비교 실험**: NoNorm / Post-LN / Pre-LN+LayerNorm / Pre-LN+RMSNorm 4가지 설정 비교
- **결론**: Pre-LN + RMSNorm이 안정성과 성능 모두 최적 (final loss ~2.7)

### 논문

- **Root Mean Square Layer Normalization** (Zhang & Sennrich, 2019)
  - https://arxiv.org/abs/1910.07467

### 코드: `model/model_minimind.py` — `RMSNorm` 클래스 (line 49~59)

```python
def norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def forward(self, x):
    return (self.weight * self.norm(x.float())).type_as(x)
```

### 핵심 개념

- **수식**: `RMSNorm(x) = x / RMS(x) * γ` where `RMS(x) = sqrt(1/d * Σxi² + ε)`
- **LayerNorm과의 차이**: mean centering을 제거 → 파라미터 d개 (LayerNorm은 2d개)
- **성능**: 7~64% 속도 향상, BF16/FP16 반정밀도에서도 안정
- **Pre-LN 구조**: `x = x + Attn(Norm(x))` — 현대 LLM의 표준 (Week 2에서 상세)
- `self.weight`는 learnable scale γ (per-dimension)

### 읽기 포인트

1. `rsqrt` = `1/sqrt()` — 나눗셈 대신 역수 곱셈으로 효율화
2. `x.float()`로 fp32 변환 후 norm → `.type_as(x)`로 원래 dtype 복원 (수치 안정성)
3. 공식 wiki의 gradient vanishing 실험 결과와 코드를 대조

---

## 2. Rotary Position Embedding (RoPE)

### 공식 wiki

- **모듈**: `02-position-encoding` (1.5시간)
- **핵심 실험**:
  - `exp1_why_position.py` — RoPE 없이 순서를 바꿔도 동일 출력 → 위치 정보의 필요성 증명
  - `exp2_rope_basics.py` — 2D 회전 시각화
  - `exp3_multi_frequency.py` — 다중 주파수 사인 곡선으로 인코딩 패턴 확인
  - `exp4_rope_explained.py` — MiniMind 구현 + YaRN 외추
- **시계 비유**: 고주파 = 초침 (인접 토큰 구분), 중주파 = 분침 (중거리), 저주파 = 시침 (전역 위치)

### 논문

- **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021)
  - https://arxiv.org/abs/2104.09864
  - Section 3.4가 핵심. 복소수 표현을 이해하면 직관적.

### 위치 인코딩의 3세대 (공식 wiki 기준)

| 세대 | 방식 | 대표 모델 | 한계 |
|------|------|----------|------|
| 1세대 | 절대 위치 인코딩 | BERT, GPT-2 | 학습 길이 초과 불가 |
| 2세대 | 상대 위치 인코딩 | T5, XLNet | 계산 복잡 |
| 3세대 | **RoPE** | **LLaMA, MiniMind** | 효율적, 상대적, 외추 가능 |

### 코드: `model/model_minimind.py` (line 61~83)

```python
# 주파수 계산: θ_i = 1 / (base^(2i/d))
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

# position × frequency → 회전 각도
t = torch.arange(end)
freqs = torch.outer(t, freqs)

# cos, sin 테이블을 미리 계산
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
```

```python
# 적용: q, k에만 RoPE를 적용 (v에는 적용하지 않음)
def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

### 핵심 개념

- **문제**: Self-Attention은 permutation invariant → "나는 너를 좋아해"와 "너를 나는 좋아해"가 동일
- **RoPE 아이디어**: 위치 m의 벡터를 각도 m×θ만큼 회전 → Q_m과 K_n의 내적이 상대 위치 (m-n)에만 의존
- **MiniMind 주파수 설정** (head_dim=64, base=1,000,000):
  - 32개 주파수 생성
  - 고주파 θ₀=1.0 → ~6 토큰마다 한 바퀴
  - 저주파 θ₃₁=0.000001 → ~628만 토큰마다 한 바퀴
- **`rotate_half` 트릭**: 복소수 곱셈 `(a+bi) × e^(imθ)`을 실수 연산 2개로 분해

### 읽기 포인트

1. `rope_base = 1e6` — base가 클수록 긴 시퀀스에서 주파수가 천천히 변화
2. Q, K에만 적용하고 V에는 적용하지 않는 이유: 위치 정보는 "관련성 판단"에만 필요
3. `register_buffer`로 저장 (line 204~206) — 학습 파라미터가 아닌 상수

---

## 3. Attention 메커니즘

### 공식 wiki

- **모듈**: `03-attention` (2시간)
- **핵심 실험**:
  - `exp1_attention_basics.py` — 기본 계산과 permutation invariance 확인
  - `exp2_qkv_explained.py` — Q, K, V 생성과 projection 효과
  - `exp3_multihead_attention.py` — 단일 head vs 다중 head 비교
- **도서관 비유**:
  - Query = "무엇을 찾고 있는가?" (검색 의도)
  - Key = 책의 태그/키워드 (매칭용)
  - Value = 실제 문서 내용 (정보)
  - Attention = 매칭 점수가 정보 검색량을 결정

### 논문

- **Attention Is All You Need** (Vaswani et al., 2017)
  - https://arxiv.org/abs/1706.03762
  - Section 3.2 "Scaled Dot-Product Attention"이 핵심.

### 코드: `model/model_minimind.py` — `Attention` 클래스 (line 90~133)

```python
# Q, K, V를 각각 Linear projection
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

# Scaled Dot-Product Attention
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

# Causal mask: 미래 토큰을 보지 못하게 upper triangle을 -inf로
scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)

# Softmax → Value에 가중합
output = F.softmax(scores.float(), dim=-1) @ xv
```

### 핵심 수식 (공식 wiki 기준)

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

4단계 분해:
1. **QK^T**: 토큰 간 유사도 → seq_len × seq_len 행렬
2. **÷√d_k**: softmax 포화 방지 → gradient 소멸 방지
3. **softmax**: 확률 분포로 변환 (합 = 1)
4. **×V**: 가중 평균으로 정보 집약

### MiniMind 설정 (공식 wiki 기준)

```
num_attention_heads = 8   (Q heads)
num_key_value_heads = 2   (KV heads — GQA)
head_dim = 64
hidden_size = 512
```

### 읽기 포인트

1. `q_proj`, `k_proj`, `v_proj`, `o_proj` — 각각의 역할과 차원
2. `view`와 `transpose`로 (batch, seq, heads, dim) 형태를 어떻게 조작하는지
3. Flash Attention 분기 (line 124~125): `F.scaled_dot_product_attention`은 PyTorch 내장 최적화

---

## 이번 주 실습

### 필수: 공식 wiki 실험 (Quick Start 포함)

1. **Quick Start 3개 실험** (30분):
   - https://www.minimind.wiki/en/docs/guide/quick-start
   - Normalization → RoPE → Attention 순서로 실행

2. **Foundation 모듈 실험** (4.5시간):
   - 01-normalization: gradient vanishing 실험, 4가지 설정 비교
   - 02-position-encoding: exp1~exp4 전부 실행
   - 03-attention: exp1~exp3 전부 실행

### 필수: MiniMind 코드

3. `model/model_minimind.py` line 1~133을 프린트해서 한 줄씩 주석 달아보기

4. 직접 실험:
   ```python
   from model.model_minimind import MiniMindConfig, RMSNorm, precompute_freqs_cis
   import torch

   # RMSNorm 동작 확인
   norm = RMSNorm(768)
   x = torch.randn(2, 10, 768)
   print(norm(x).shape, norm(x).mean().item(), norm(x).std().item())

   # RoPE frequency 시각화
   cos, sin = precompute_freqs_cis(dim=96, end=512)
   print(cos.shape)  # (512, 96)
   # matplotlib으로 cos[:100, :8] 히트맵을 그려보면 주파수 패턴이 보임
   ```

### 셀프 체크 (공식 wiki 기준)

- [ ] RMSNorm 수식을 외워서 쓸 수 있는가
- [ ] Pre-LN과 Post-LN의 차이를 설명할 수 있는가
- [ ] RoPE의 회전 원리를 그림으로 그릴 수 있는가
- [ ] Q, K, V 각각의 역할을 설명할 수 있는가
- [ ] √d_k로 나누는 이유를 설명할 수 있는가
- [ ] Causal mask의 목적을 설명할 수 있는가

---

## 참고 자료

- MiniMind 공식 wiki Systematic Study: https://www.minimind.wiki/en/docs/guide/systematic
- The Illustrated Transformer (Jay Alammar): https://jalammar.github.io/illustrated-transformer/
- The Annotated Transformer (Harvard NLP): https://nlp.seas.harvard.edu/annotated-transformer/
