# Week 9: Reasoning 모델 & RLVR

> 목표: RL만으로 추론 능력이 출현하는 원리를 이해한다.
> Week 6의 GRPO가 여기서 핵심 도구가 된다.

---

## 연결 고리

```
Week 6: GRPO 알고리즘 학습 (train_grpo.py)
Week 8: Agentic RL — 도구 사용 + GRPO (train_agent.py)
Week 9: → GRPO로 "생각하는 능력" 자체를 학습 (DeepSeek-R1)
```

---

## 1. DeepSeek-R1: RL만으로 추론 능력 획득

- **논문**: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2025)
  - https://arxiv.org/abs/2501.12948
- **핵심**: SFT 없이 GRPO + 검증 가능한 보상(RLVR)만으로 chain-of-thought가 **자발적으로 출현**
- **의미**: 자기 반성, 역추적, 전략 전환이 명시적 학습 없이 emerge

```
DeepSeek-R1 파이프라인:
1. Base model (pretrain만 된 상태)
2. GRPO + verifiable reward (정답 맞으면 +1, 틀리면 0)
3. → <think> 태그 안에서 자발적 추론 출현
4. → 수학/코드에서 GPT-4o 급 성능
```

**MiniMind 연결**: `train_grpo.py`의 GRPO가 이 논문의 핵심 알고리즘. `train_agent.py`의 `<think>` reward 설계도 같은 맥락.

---

## 2. Dr. GRPO: R1 학습의 함정과 개선

- **논문**: Understanding R1-Zero-Like Training: A Critical Perspective (2025)
  - https://arxiv.org/abs/2503.20783
- **문제 발견**: GRPO의 length bias — 긴 응답에 보상이 편향
- **개선**: Dr. GRPO가 bias를 제거, 토큰 효율성 향상
- **MiniMind 연결**: `train_grpo.py`의 `rep_penalty` 함수가 이 문제를 부분적으로 다루는 시도

---

## 3. RLVR (RL from Verifiable Rewards)

**RLVR = 검증 가능한 보상으로 RL 학습**

기존 RLHF: 사람이 "좋은 답변"을 판단 → reward model → 비용 크고 주관적
RLVR: 수학 정답, 코드 테스트 통과 등 **자동 검증** 가능한 보상 사용

### 3-1. RLVR이 수학/코드를 넘어 일반화

- **논문**: Expanding RL with Verifiable Rewards Across Diverse Domains (2025)
  - https://arxiv.org/abs/2503.23829
- 의학, 화학, 심리학 등에도 RLVR 적용 가능
- 범용 verifier로 distilled generative reward model 사용

### 3-2. 최종 답만 검증해도 중간 추론이 올바르게 학습됨

- **논문**: RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs (2025)
  - https://arxiv.org/abs/2506.14245
- 이론적 증명: outcome-level reward만으로 process-level correctness가 유도됨
- "왜 R1이 되는가"에 대한 수학적 답

---

## 4. Process Reward Model vs Outcome Reward Model

- **논문**: Beyond Outcome Verification: Verifiable Process Reward Models (2026)
  - https://arxiv.org/abs/2601.17223

```
ORM (Outcome Reward Model):
  - 최종 답만 평가 (맞았나/틀렸나)
  - 장점: 자동화 쉬움
  - 단점: 어디서 틀렸는지 모름 (credit assignment 문제)

PRM (Process Reward Model):
  - 중간 추론 단계마다 평가
  - 장점: 정확한 피드백
  - 단점: 사람이 annotation 해야 함 → 비용 높음

VPRM (Verifiable PRM) — 2026 최신:
  - 규칙 기반 검증기로 중간 단계 자동 평가
  - ORM의 자동화 + PRM의 정확성 결합
  - Agent의 multi-step tool call 검증에 핵심
```

---

## 이번 주 읽기 순서

| 순서 | 논문 | 이유 |
|------|------|------|
| 1 | DeepSeek-R1 | 전체 맥락의 기초. train_grpo.py 코드와 1:1 대응 |
| 2 | Dr. GRPO | R1의 한계와 개선. GRPO 실전 적용 시 주의점 |
| 3 | RLVR Implicitly Incentivizes | "왜 되는가"의 이론적 답 |
| 4 | RLVR Across Domains | 수학 너머로의 확장 가능성 |
| 5 | Verifiable PRM | ORM/PRM/VPRM 비교 — reward 설계의 최전선 |

---

## 실습 (MiniMind 연결)

1. `train_grpo.py`의 `calculate_rewards` 함수를 다시 읽기 — DeepSeek-R1의 reward 설계와 비교
2. MiniMind의 reward를 순수 RLVR 스타일로 바꿔보기:
   ```python
   # 기존: 길이 보상 + 사고 보상 + 반복 패널티 + reward model
   # RLVR: 정답이면 +1, 아니면 0 (검증 가능한 task 필요)
   ```
3. `train_agent.py`의 `<think>` 보상이 R1의 사고 출현과 어떻게 연결되는지 정리

---

## 셀프 체크

- [ ] DeepSeek-R1에서 GRPO가 어떻게 사용되는지 설명할 수 있는가
- [ ] RLVR과 기존 RLHF의 차이를 설명할 수 있는가
- [ ] ORM vs PRM vs VPRM의 차이를 설명할 수 있는가
- [ ] R1 학습의 length bias 문제를 설명할 수 있는가
- [ ] `train_grpo.py` ↔ DeepSeek-R1 논문의 대응 관계를 코드로 짚을 수 있는가
