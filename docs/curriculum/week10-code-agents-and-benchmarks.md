# Week 10: 코드 에이전트 & Agent 벤치마크

> 목표: LLM Agent가 실전에서 어떻게 평가되는지 이해한다.
> SWE-bench를 중심으로 코드 에이전트의 발전과 한계를 파악한다.

---

## Part 1: 코드 에이전트

### 1-1. SWE-bench: 코드 에이전트의 표준 벤치마크

- **논문**: SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (Princeton, ICLR 2024)
  - https://arxiv.org/abs/2310.06770
- **핵심**: 실제 GitHub 이슈를 해결하는 능력 테스트
  - 이슈 설명 → 코드 수정 → 기존 테스트 통과
  - 초기 GPT-4: ~2% 해결률
  - 2026년 최상위 에이전트: ~43% (SWE-bench Verified)

```
SWE-bench 난이도 계층:
  SWE-bench Lite     — 300개 쉬운 이슈
  SWE-bench Full     — 2,294개 전체
  SWE-bench Verified — 500개, 사람이 검증한 서브셋
  SWE-bench Pro      — 더 복잡한 실전 이슈 (최신)
```

**왜 중요한가**: 모든 코드 에이전트(Claude Code, Cursor, Devin 등)의 능력을 비교하는 공통 잣대.

---

### 1-2. SWE-agent: 인터페이스 설계의 중요성

- **논문**: SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering (Princeton, 2024)
  - https://arxiv.org/abs/2405.15793
- **핵심**: raw shell 대신 전용 **Agent-Computer Interface(ACI)** 설계
  - 파일 탐색, 검색, 편집 등의 특화 명령어 제공
  - 12.5% pass@1 달성 (이전 ~0% 대비)
- **핵심 인사이트**: **모델 능력만큼 인터페이스 설계가 중요**

```
Claude Code의 도구:
  Read, Edit, Grep, Glob, Bash ...
  ↑ SWE-agent ACI 개념의 실제 적용
```

모델에 "grep으로 검색해" vs "전용 검색 도구 사용해" — 후자가 성공률이 훨씬 높음.

---

### 1-3. Agentless: 단순함의 힘

- **논문**: Agentless: Demystifying LLM-based Software Engineering Agents (2024)
  - https://arxiv.org/abs/2407.01489
- **핵심**: 자율 agent 루프 없이 2단계 파이프라인만으로 경쟁력 있는 성능
  ```
  Step 1: Fault Localization — 어디가 문제인지 찾기
  Step 2: Patch Generation  — 수정 코드 생성
  ```
- **인사이트**: 복잡한 agent 구조가 항상 좋은 건 아님

```
Agent 접근:    루프 (관찰 → 행동 → 관찰 → ...) — 유연하지만 불안정
Agentless:     직선 (분석 → 수정) — 단순하지만 강력한 baseline
```

---

## Part 2: Agent 벤치마크

### 2-1. 주요 벤치마크 비교

| 벤치마크 | 측정 대상 | 인간 성능 | 최고 AI 성능 | Gap |
|----------|----------|----------|-------------|-----|
| **SWE-bench Verified** | 코드 이슈 해결 | ~75% | ~43% | 큼 |
| **GAIA** | 범용 어시스턴트 | 92% | ~50% | 큼 |
| **AgentBench** | 8개 환경 | - | 상업 >> 오픈소스 | 격차 존재 |
| **tau-bench** | 도구+사용자 멀티턴 | - | GPT-4o < 50% | 매우 큼 |

---

### 2-2. AgentBench: 범용 Agent 능력 측정

- **논문**: AgentBench: Evaluating LLMs as Agents (Tsinghua/CMU, ICLR 2024)
  - https://arxiv.org/abs/2308.03688
- 8개 환경: 웹 브라우저, OS, 데이터베이스, 게임 등
- **발견**: 상업 모델과 오픈소스 모델의 agent 능력 격차가 매우 큼
- 최초의 체계적 agent 벤치마크

---

### 2-3. GAIA: 범용 어시스턴트

- **논문**: GAIA: A Benchmark for General AI Assistants (Meta/HuggingFace, 2023)
  - https://arxiv.org/abs/2311.12983
- 466개 실제 질문: 추론 + 멀티모달 + 웹 검색 + 도구 사용
- 인간 92% vs 초기 GPT-4+plugins 15% → 2026년 ~50%
- 여전히 포화되지 않은 장기 목표 벤치마크

---

### 2-4. tau-bench: 일관성의 문제 (가장 중요)

- **논문**: tau-bench: A Benchmark for Tool-Agent-User Interaction (Princeton, 2024)
  - https://arxiv.org/abs/2406.12045
- **핵심 지표**: `pass^k` — k번 시도 중 **매번** 성공하는 비율

```
예시:
  pass@1 = 70%  (한 번 시도 시 70% 확률로 성공)
  pass^5 = 30%  (5번 연속 시도 시 30%만 매번 성공)

→ 10번 중 7번 성공하지만, "항상 성공"하지는 못함
→ 프로덕션에서는 pass^k가 중요
```

- **의미**: Agent가 "가끔 되는 것"과 "항상 되는 것"의 차이가 엄청나게 큼
- 이 **일관성 격차**가 실제 배포의 가장 큰 장벽

---

## 이번 주 읽기 순서

| 순서 | 논문 | 이유 |
|------|------|------|
| 1 | SWE-bench | 코드 에이전트의 기준점 |
| 2 | SWE-agent | 인터페이스 설계 = 모델 만큼 중요 |
| 3 | tau-bench | 일관성 문제 — 실전 배포의 핵심 |
| 4 | Agentless | 단순한 baseline의 위력 |
| 5 | GAIA | 범용 agent의 현 위치 |
| 6 | AgentBench | 상업/오픈소스 격차 |

---

## 실습

1. SWE-bench 리더보드를 확인하고, 최신 1~3위 모델/에이전트가 무엇인지 조사
2. Claude Code, Cursor, Devin 등의 아키텍처를 SWE-agent ACI 관점에서 비교
3. tau-bench의 pass@1 vs pass^k 개념을 직접 시뮬레이션:
   ```python
   import random
   success_rate = 0.7
   k = 5
   trials = 10000
   pass_at_1 = sum(random.random() < success_rate for _ in range(trials)) / trials
   pass_k = sum(all(random.random() < success_rate for _ in range(k)) for _ in range(trials)) / trials
   print(f"pass@1: {pass_at_1:.2%}, pass^{k}: {pass_k:.2%}")
   ```

---

## 셀프 체크

- [ ] SWE-bench가 측정하는 것과 현재 최고 성능을 알고 있는가
- [ ] SWE-agent의 ACI 개념을 설명할 수 있는가
- [ ] Agentless의 접근법이 왜 강력한 baseline인지 설명할 수 있는가
- [ ] tau-bench의 pass@1 vs pass^k 격차가 의미하는 바를 설명할 수 있는가
- [ ] 4개 벤치마크의 차이점을 한 문장씩 설명할 수 있는가
