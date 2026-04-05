# Week 6: PPO & GRPO

> 목표: RL 기반 언어모델 학습의 두 가지 접근법을 이해한다.
> PPO의 actor-critic 구조와 GRPO의 critic-free 방식을 비교한다.

---

## 공식 학습 자료

Week 5에 이어 공식 Deep Mastery Week 4의 RLHF/RLAIF 트랙 후반부.

| 공식 가이드 | URL |
|-------------|-----|
| Deep Mastery — Week 4: Advanced Topics | https://www.minimind.wiki/en/docs/guide/mastery |

---

## 1. PPO (Proximal Policy Optimization)

### 논문

- **Proximal Policy Optimization Algorithms** (Schulman et al., 2017)
  - https://arxiv.org/abs/1707.06347
  - Section 3 "Clipped Surrogate Objective"가 핵심.
  - RL 배경이 없으면 어려울 수 있음 — 아래 개념 정리를 먼저 읽기.

### RL 기본 개념 (LLM 맥락)

```
Agent  = 언어 모델 (policy π)
State  = 지금까지 생성한 토큰 시퀀스
Action = 다음 토큰 선택
Reward = 전체 응답에 대한 점수 (reward model이 판단)
```

PPO의 목표: **reward를 최대화하되, 기존 policy에서 너무 멀리 벗어나지 않게.**

### 코드: `trainer/train_ppo.py`

**Critic Model** (line 36~48):
```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)  # 가치 추정용

    def forward(self, input_ids, ...):
        hidden_states = self.model(input_ids, ...)[0]
        values = self.value_head(hidden_states).squeeze(-1)
        return values  # 각 토큰 위치의 예상 보상
```

**Actor-Critic 구조**:
- **Actor** (policy model): 토큰을 생성 → 더 좋은 응답을 생성하도록 학습
- **Critic** (value model): 현재 상태의 "가치"를 예측 → advantage 계산에 사용
- **Reference model**: frozen. policy가 너무 벗어나지 않도록 KL penalty 제공

**Reward 계산** (line 51~75):
```python
def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses))
    # 1. 길이 기반 보상
    rewards[i] += 0.5 if 20 <= len(response) <= 800 else -0.5
    # 2. 사고 과정 보상 (<think> 태그)
    if '</think>' in response:
        rewards[i] += 1.0 if 20 <= len(thinking) <= 300 else -0.5
    # 3. 반복 패널티
    rewards[i] -= rep_penalty(answer)
    # 4. Reward model 점수
    score = reward_model.get_score(messages, answer)
    rewards += reward_model_scores
```

### PPO 핵심 요소

1. **GAE (Generalized Advantage Estimation)**: 미래 보상을 현재로 할인하여 advantage 계산
2. **Clipped Objective**: ratio = π_new/π_old를 [1-ε, 1+ε] 범위로 클리핑
3. **Value Loss**: critic의 가치 추정과 실제 return의 차이
4. **KL Penalty**: reference model과의 KL divergence를 보상에서 차감

### PPO의 단점

- Actor + Critic + Reference + Reward = **4개 모델**이 동시에 메모리에 있어야 함
- 학습이 불안정하고, 하이퍼파라미터에 민감
- 이 문제를 해결하기 위해 GRPO가 등장

---

## 2. GRPO (Group Relative Policy Optimization)

### 논문

- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (Shao et al., 2024)
  - https://arxiv.org/abs/2402.03300
  - Section 3.2 "Group Relative Policy Optimization"
  - PPO의 어떤 부분을 제거하고 어떻게 대체했는지에 집중.

### 코드: `trainer/train_grpo.py`

**핵심 아이디어**: Critic을 제거하고, 같은 prompt에 대해 여러 응답을 생성한 뒤 **그룹 내 상대 비교**로 advantage를 구한다.

```python
# 같은 prompt에 대해 num_generations개의 응답 생성
rollout_result = rollout_engine.rollout(
    prompt_ids=prompt_inputs["input_ids"],
    prompt_mask=prompt_inputs["attention_mask"],
    max_new_tokens=args.max_new_tokens,
    num_generations=args.num_generations,  # 예: 4개
    temperature=1.0,
)
```

**Group Advantage 계산**:
```
prompt X에 대해 4개 응답 생성 → reward: [0.3, 0.8, 0.1, 0.5]
그룹 평균: 0.425, 표준편차: 0.264
normalized advantage: [-0.47, 1.42, -1.23, 0.28]
```
- Critic 없이도 "이 그룹에서 상대적으로 좋은/나쁜 응답"을 판단 가능
- 절대적 reward가 아닌 상대적 순위가 중요

### PPO vs GRPO 비교

| | PPO | GRPO |
|--|-----|------|
| 필요 모델 | Actor + Critic + Ref + Reward | Actor + Ref + Reward |
| Advantage | GAE (Critic 기반) | 그룹 내 상대 비교 |
| 메모리 | 높음 | 낮음 |
| 안정성 | 불안정 (Critic 학습이 어려움) | 상대적으로 안정 |
| 샘플 효율 | 좋음 | 다수 생성 필요 |

### Rollout Engine

```python
# trainer/rollout_engine.py
# 모델로 응답을 생성하는 엔진 — PPO와 GRPO 모두 공유
rollout_engine = create_rollout_engine(model, tokenizer, ...)
```
- Rollout = 모델이 실제로 텍스트를 생성하는 과정
- 생성된 텍스트에 reward를 매기고, 그 reward로 policy를 업데이트

### Reward 설계

PPO와 GRPO 모두 동일한 reward 구조를 사용 (line 30~67):
1. **길이 보상**: 너무 짧거나 긴 응답에 패널티
2. **사고 과정 보상**: `<think>` 태그를 적절히 사용하면 보상
3. **반복 패널티**: n-gram 반복이 많으면 감점
4. **Reward model 점수**: 별도 학습된 모델이 품질 평가

---

## 이번 주 실습

### 필수

1. PPO 논문의 clipped objective 수식을 이해하고, `train_ppo.py`에서 해당 코드 찾기
2. GRPO 논문의 group advantage 수식을 이해하고, `train_grpo.py`에서 해당 코드 찾기
3. `calculate_rewards` 함수를 읽고 reward가 어떻게 구성되는지 정리

### 선택

4. GRPO 실행:
   ```bash
   uv run python trainer/train_grpo.py --device mps --epochs 1
   ```
   - PPO보다 GRPO가 메모리 사용량이 적은 것을 확인

5. `num_generations` 값을 바꿔가며 실험:
   - 그룹 크기가 클수록 advantage 추정이 정확하지만 느림
   - 작으면 빠르지만 노이즈가 많음

6. Reward 함수를 수정해서 실험:
   - 예: 영어 단어가 포함되면 감점하는 reward 추가
   - reward 설계가 모델 행동에 미치는 영향을 직접 체감

---

## 참고 자료 (선택)

- Spinning Up in Deep RL (OpenAI): https://spinningup.openai.com/
  - PPO를 이해하기 위한 RL 기초. Policy Gradient → TRPO → PPO 순서로 읽기.
- The N Implementation Details of RLHF with PPO (Huang et al.): https://arxiv.org/abs/2403.17031
