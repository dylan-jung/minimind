# Week 11: Agent 프로토콜, 멀티에이전트 & 최신 모델

> 목표: Agent 시스템의 통신 표준, 협업 패턴, 그리고 최신 모델 아키텍처 트렌드를 파악한다.
> 개별 Agent에서 **시스템**으로의 확장을 이해한다.

---

## Part 1: Agent 프로토콜 — MCP & A2A

### 1-1. MCP (Model Context Protocol)

- **출시**: Anthropic, 2024.11
- **Survey**: A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, and ANP (2025)
  - https://arxiv.org/abs/2505.02279
- **핵심**: LLM이 외부 도구/데이터에 접근하는 표준 프로토콜
  - JSON-RPC 기반
  - **Tool**: 함수 호출 (계산, API 호출 등)
  - **Resource**: 데이터 읽기 (파일, DB 등)
  - **Prompt**: 재사용 가능한 프롬프트 템플릿

```json
// MCP Tool 정의 예시 — MiniMind의 TOOLS 정의와 비교해보기
{
  "name": "get_current_weather",
  "description": "Get weather for a location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {"type": "string"}
    },
    "required": ["location"]
  }
}
```

**MiniMind 연결**: `train_agent.py`의 TOOLS 정의가 MCP tool schema와 거의 동일한 구조.

---

### 1-2. A2A (Agent-to-Agent Protocol)

- **출시**: Google, 2025.04
- **핵심**: Agent 간 통신 표준
  - **Agent Card**: 능력, endpoint, 인증 정보를 선언하는 JSON 메타데이터
  - 비동기 태스크 위임 및 결과 수신
  - 상태 추적 (submitted → working → completed)

---

### 1-3. MCP vs A2A — 경쟁이 아닌 보완

- **논문**: A Study on the MCP x A2A Framework (2025)
  - https://arxiv.org/abs/2506.01804

```
MCP: Agent ↔ Tool (도구 호출 — 수직적)
A2A: Agent ↔ Agent (협업/위임 — 수평적)

┌─────────────────────────────────────┐
│           Orchestrator Agent         │
│  ┌─────────┐  ┌─────────┐          │
│  │ Agent A  │←→│ Agent B  │  ← A2A  │
│  └────┬─────┘  └────┬─────┘         │
│       │              │               │
│  ┌────┴────┐   ┌────┴────┐          │
│  │  DB     │   │  API    │  ← MCP   │
│  └─────────┘   └─────────┘          │
└─────────────────────────────────────┘
```

### 1-4. 4대 프로토콜 비교 (Survey 기준)

| 프로토콜 | 출처 | 통신 방식 | 주 용도 |
|----------|------|----------|---------|
| **MCP** | Anthropic | JSON-RPC | Agent ↔ Tool/Resource |
| **A2A** | Google | Agent Card + REST | Agent ↔ Agent |
| **ACP** | IBM/Linux Foundation | RESTful messaging | Enterprise agent 통신 |
| **ANP** | 커뮤니티 | W3C DID 기반 | 탈중앙화 agent 네트워크 |

---

## Part 2: 멀티에이전트 시스템

### 2-1. 협업 패턴 분류

- **논문**: Multi-Agent Collaboration Mechanisms: A Survey of LLMs (2025)
  - https://arxiv.org/abs/2501.06322

| 패턴 | 설명 | 예시 |
|------|------|------|
| **Debate** | 서로 반박하며 정답 수렴 | 수학 증명 검증 |
| **Relay** | 순차적 작업 전달 | 코드 작성 → 리뷰 → 테스트 |
| **Report** | 결과를 중앙에 보고 | 병렬 검색 후 종합 |
| **Memory** | 공유 메모리로 협업 | 장기 프로젝트 |

### 2-2. RL로 학습하는 오케스트레이터

- **논문**: Multi-Agent Collaboration via Evolving Orchestration (2025)
  - https://arxiv.org/abs/2505.19591
- **핵심**: 고정 파이프라인 대신 RL로 학습된 오케스트레이터가 동적으로 agent 순서 결정
- 정적 워크플로우의 취약성 해결 — 태스크 복잡도에 적응

### 2-3. 멀티턴 Agent RL

- **논문**: SWEET-RL (Meta FAIR / UC Berkeley, 2025)
  - https://arxiv.org/abs/2503.15478
  - 학습 시에만 사용 가능한 추가 정보로 critic 학습 → 턴별 보상 정확히 할당
  - Llama-3.1-8B가 GPT-4o를 매칭

- **논문**: Agent-R1 (2025)
  - https://arxiv.org/abs/2511.14460
  - DeepSeek-R1의 RL을 agent 환경에 직접 적용
  - MiniMind `train_agent.py`의 대규모 버전

- **논문**: AgentGym-RL (Fudan University, 2025)
  - https://arxiv.org/abs/2509.08755
  - 27개 환경에서 멀티턴 RL로 agent 학습
  - ScalingInter-RL: 상호작용 범위를 점진적으로 확장

---

## Part 3: Tool Use의 진화

### 3-1. 단일 호출에서 Agentic Skills로

- **논문**: SoK: Agentic Skills — Beyond Tool Use in LLM Agents (2026)
  - https://arxiv.org/abs/2602.20867
- **논문**: The Evolution of Tool Use in LLM Agents (2026)
  - https://arxiv.org/abs/2603.22862

```
2023: Toolformer   — 단일 API 호출 삽입
2024: Function Call — 구조화된 도구 호출 (JSON schema)
2025: Multi-tool    — 여러 도구를 순서대로 호출, 중간 결과 반영
2026: Agentic Skill — 상태 유지, 적용 조건 판단, 종료 조건 판단
```

**핵심 전환**: "도구를 호출할 수 있는가" → "도구 시퀀스를 부분 피드백과 함께 추론할 수 있는가"

---

## Part 4: 최신 모델 아키텍처 트렌드

### 4-1. Qwen3 / Qwen3.5 — MiniMind가 따르는 구조

- **Qwen3** (2025): MiniMind-3의 주 구조 정렬 대상
  - Dense + MoE 이중 라인업
  - `<think>` / `</think>` 기반 적응형 사고 (thinking/non-thinking 모드 전환)
  - GQA + RoPE + SwiGLU — MiniMind에서 배운 것 그대로
- **Qwen3.5** (2025~2026): Qwen3 후속
  - 더 긴 컨텍스트 (128K+)
  - 향상된 tool calling / agentic 능력
  - MoE 효율성 개선

```
MiniMind ↔ Qwen3 구조 대응:
  MiniMindConfig       ↔  Qwen3Config
  Attention (GQA)      ↔  Qwen3Attention
  FeedForward (SwiGLU) ↔  Qwen3MLP
  MOEFeedForward       ↔  Qwen3MoE
  RoPE + YaRN          ↔  동일
```

### 4-2. NVIDIA Nemotron — 활성 파라미터 효율의 극한

- **Nemotron-Super-120B-A12B**: 총 120B 파라미터, 활성 12B
  - MoE의 극한: 파라미터 대비 연산량 1/10
  - **NVFP4 양자화**: 4bit 부동소수점으로 메모리 대폭 절감
  - NAS (Neural Architecture Search) 기반 expert 구성 최적화
- **MiniMind 연결**: `MOEFeedForward`에서 배운 MoE 원리의 산업 스케일 적용

```
같은 원리, 다른 스케일:
  MiniMind:  198M total / 64M active   (A64M)
  Nemotron:  120B total / 12B active   (A12B)
```

### 4-3. 양자화 (Quantization) — 추론 효율

```
FP32:   32bit  — 학습 시 기본
BF16:   16bit  — 학습 시 mixed precision (MiniMind의 AMP)
INT8:    8bit  — 추론 최적화
INT4:    4bit  — GPTQ, AWQ 등
NVFP4:   4bit  — NVIDIA 전용 부동소수점 4bit (Nemotron)
```

모델 크기 ÷ bit = VRAM 필요량
- 120B × FP16 = ~240GB VRAM 필요
- 120B × NVFP4 = ~60GB VRAM → 단일 GPU 가능

### 4-4. 최신 모델 공통 패턴 요약

| 패턴 | MiniMind에서 배운 곳 | 산업 적용 |
|------|---------------------|----------|
| GQA | Week 2 | Qwen3, Llama 3, Gemma |
| SwiGLU | Week 2 | 거의 모든 최신 모델 |
| RoPE + YaRN | Week 1, 7 | Qwen3, Llama, DeepSeek |
| MoE + Load Balancing | Week 7 | Qwen3-MoE, Nemotron, Mixtral |
| `<think>` adaptive thinking | Week 8 | Qwen3, DeepSeek-R1, Claude |
| RLVR / GRPO | Week 6, 9 | DeepSeek-R1, Qwen3 |
| Weight Tying | Week 2 | 소형 모델에서 보편적 |

---

## 이번 주 읽기 순서

| 순서 | 논문 | 이유 |
|------|------|------|
| 1 | MCP/A2A Survey | Agent 시스템의 통신 표준 지형도 |
| 2 | SWEET-RL | 멀티턴 credit assignment — agent RL의 핵심 |
| 3 | Multi-Agent Survey | 협업 패턴 분류 |
| 4 | Agent-R1 | R1 → Agent RL 확장 |
| 5 | Agentic Skills SoK | tool use → skill의 패러다임 전환 |

나머지는 관심에 따라 선택:
- 멀티에이전트: Evolving Orchestration, AgentGym-RL
- Tool Use 진화: Evolution of Tool Use
- 프로토콜 실전: MCP x A2A Framework

---

## 실습

1. MiniMind `train_agent.py`의 TOOLS 정의를 MCP tool schema 형식으로 변환해보기
2. Qwen3의 모델 config와 MiniMind의 `MiniMindConfig`를 비교:
   ```python
   from transformers import AutoConfig
   qwen_config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
   # MiniMindConfig와 하나씩 대응시켜 보기
   ```
3. `MOEFeedForward`의 `num_experts`, `num_experts_per_tok` 설정을 Nemotron/Mixtral 스펙과 비교

---

## 셀프 체크

- [ ] MCP와 A2A의 차이와 보완 관계를 설명할 수 있는가
- [ ] 멀티에이전트 협업 패턴 4가지를 설명할 수 있는가
- [ ] 멀티턴 Agent RL에서 credit assignment가 왜 어려운지 설명할 수 있는가
- [ ] Qwen3와 MiniMind의 구조적 대응을 설명할 수 있는가
- [ ] Nemotron의 120B/A12B가 의미하는 바를 MoE 관점에서 설명할 수 있는가
- [ ] NVFP4 같은 양자화가 왜 중요한지 설명할 수 있는가
- [ ] Tool Use의 2023→2026 진화 과정을 설명할 수 있는가

---

## 전체 커리큘럼 완료 후 다음 단계

```
옵션 A: MiniMind 확장
  - MiniMind-V (Vision 멀티모달) 학습
  - 새로운 tool 추가 및 Agentic RL 실험
  - MoE expert 수 늘려보기

옵션 B: 실제 모델 코드 읽기
  - Qwen3 transformers 구현 코드 vs MiniMind 비교
  - DeepSeek-R1 공개 코드 분석
  - Llama 3 코드 비교

옵션 C: 실전 적용
  - MCP 서버 직접 구현
  - SWE-bench에 MiniMind 기반 에이전트 도전
  - 자체 RLVR 파이프라인 구축
```
