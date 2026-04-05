# MiniMind LLM 학습 커리큘럼

12주 과정으로 LLM의 전체 파이프라인을 논문 + 코드로 학습한다.
논문 40편 + MiniMind 전체 코드를 커버한다. 매주 하나의 파일을 읽고 실습한다.

---

## 환경

- **로컬 실행**: M4 Pro 24GB, `--device mps`
- **패키지 관리**: `uv run python ...`
- **데이터 다운로드**: https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files

## 공식 자료

- MiniMind Wiki: https://www.minimind.wiki/en/
- Quick Start (30분): https://www.minimind.wiki/en/docs/guide/quick-start
- Systematic Study (6시간): https://www.minimind.wiki/en/docs/guide/systematic
- Deep Mastery (30시간+): https://www.minimind.wiki/en/docs/guide/mastery

---

## 커리큘럼

### Phase 1: 모델 구조 (Week 1~2)

| Week | 주제 | 논문 | 코드 | 공식 wiki |
|------|------|------|------|----------|
| [Week 1](week1-transformer-architecture.md) | Transformer 핵심 구조 | Attention Is All You Need, RMSNorm, RoPE | `model/model_minimind.py` 전반부 | Normalization, Position Encoding, Attention |
| [Week 2](week2-model-structure-and-generation.md) | 전체 모델 & 추론 | GQA, GLU Variants | `model/model_minimind.py` 후반부 | FeedForward, Architecture |

### Phase 2: 학습 파이프라인 (Week 3~4)

| Week | 주제 | 논문 | 코드 | 공식 wiki |
|------|------|------|------|----------|
| [Week 3](week3-tokenizer-and-pretraining.md) | 토크나이저 & Pretrain | BPE, GPT-2, Scaling Laws | `train_tokenizer.py`, `train_pretrain.py` | Deep Mastery Week 2~3 |
| [Week 4](week4-sft-and-lora.md) | SFT & LoRA | InstructGPT, LoRA | `train_full_sft.py`, `model_lora.py` | Deep Mastery Week 3 |

### Phase 3: RLHF / RLAIF (Week 5~6)

| Week | 주제 | 논문 | 코드 | 공식 wiki |
|------|------|------|------|----------|
| [Week 5](week5-dpo.md) | DPO | InstructGPT, DPO | `train_dpo.py` | Deep Mastery Week 4 |
| [Week 6](week6-ppo-and-grpo.md) | PPO & GRPO | PPO, DeepSeekMath | `train_ppo.py`, `train_grpo.py` | Deep Mastery Week 4 |

### Phase 4: 고급 주제 (Week 7~8)

| Week | 주제 | 논문 | 코드 | 공식 wiki |
|------|------|------|------|----------|
| [Week 7](week7-moe-distillation-yarn.md) | MoE, Distillation, YaRN | Mixtral, Hinton Distillation, YaRN | `model_minimind.py` MoE, `train_distillation.py` | Deep Mastery Week 4 |
| [Week 8](week8-agentic-rl-and-tooluse.md) | Agentic RL & Tool Use | Toolformer, ReAct | `train_agent.py` | 심화 |

### Phase 5: 프론티어 (Week 9~11)

| Week | 주제 | 논문 | 핵심 키워드 |
|------|------|------|------------|
| [Week 9](week9-agentic-flow-and-frontier.md) | Reasoning 모델 & RLVR | DeepSeek-R1, Dr. GRPO, VPRM | GRPO→추론출현, ORM/PRM/VPRM |
| [Week 10](week10-code-agents-and-benchmarks.md) | 코드 에이전트 & 벤치마크 | SWE-bench, SWE-agent, tau-bench | ACI 설계, pass@1 vs pass^k |
| [Week 11](week11-protocols-multiagent-and-latest-models.md) | 프로토콜, 멀티에이전트 & 최신 모델 | MCP/A2A, SWEET-RL, Agent-R1 | Qwen3.5, Nemotron, 양자화 |

### Phase 6: 실전 파이프라인 (Week 12)

| Week | 주제 | 코드 | 핵심 키워드 |
|------|------|------|------------|
| [Week 12](week12-data-eval-serving.md) | 데이터, 평가 & 서빙 | `lm_dataset.py`, `eval_llm.py`, `serve_openai_api.py`, `convert_model.py` | Label masking, 벤치마크, OpenAI API 호환, Qwen3 변환 |

---

## 학습 방법론 (공식 wiki 권장)

1. **실험 우선**: 이론보다 코드 실행을 먼저. 직관을 쌓은 뒤 논문으로 깊이 확보.
2. **비교 학습**: "이것 없으면 무엇이 깨지는가?"를 항상 질문.
3. **반복 읽기**:
   - 1회차: 전체 흐름 파악
   - 2회차: 수식/기술 세부사항
   - 3회차: 직접 구현

---

## 전체 논문 목록

### Week 1~8: 기초 → 고급 (20편)

| # | 논문 | 연도 | Week |
|---|------|------|------|
| 1 | Attention Is All You Need | 2017 | 1 |
| 2 | Root Mean Square Layer Normalization | 2019 | 1 |
| 3 | RoFormer (RoPE) | 2021 | 1 |
| 4 | GQA | 2023 | 2 |
| 5 | GLU Variants Improve Transformer | 2020 | 2 |
| 6 | Neural Text Degeneration (Top-P) | 2020 | 2 |
| 7 | BPE (Subword Units) | 2016 | 3 |
| 8 | GPT-2 | 2019 | 3 |
| 9 | Scaling Laws for Neural Language Models | 2020 | 3 |
| 10 | InstructGPT (RLHF) | 2022 | 4, 5 |
| 11 | LoRA | 2021 | 4 |
| 12 | Direct Preference Optimization (DPO) | 2023 | 5 |
| 13 | Learning to Summarize from Human Feedback | 2020 | 5 |
| 14 | Proximal Policy Optimization (PPO) | 2017 | 6 |
| 15 | DeepSeekMath (GRPO) | 2024 | 6 |
| 16 | Mixtral of Experts | 2024 | 7 |
| 17 | Distilling the Knowledge in a Neural Network | 2015 | 7 |
| 18 | YaRN | 2023 | 7 |
| 19 | Toolformer | 2023 | 8 |
| 20 | ReAct | 2023 | 8 |

### Week 9~11: 프론티어 (20편)

| # | 논문 | 연도 | Week |
|---|------|------|------|
| 21 | DeepSeek-R1 | 2025 | 9 |
| 22 | Understanding R1-Zero (Dr. GRPO) | 2025 | 9 |
| 23 | RLVR Across Diverse Domains | 2025 | 9 |
| 24 | RLVR Implicitly Incentivizes Reasoning | 2025 | 9 |
| 25 | Verifiable Process Reward Models | 2026 | 9 |
| 26 | SWE-bench | 2023/2024 | 10 |
| 27 | SWE-agent | 2024 | 10 |
| 28 | Agentless | 2024 | 10 |
| 29 | AgentBench | 2023/2024 | 10 |
| 30 | GAIA | 2023/2024 | 10 |
| 31 | tau-bench | 2024 | 10 |
| 32 | Agent Interoperability Protocols Survey | 2025 | 11 |
| 33 | MCP x A2A Framework | 2025 | 11 |
| 34 | Multi-Agent Collaboration Survey | 2025 | 11 |
| 35 | Multi-Agent Evolving Orchestration | 2025 | 11 |
| 36 | SWEET-RL | 2025 | 11 |
| 37 | Agent-R1 | 2025 | 11 |
| 38 | AgentGym-RL | 2025 | 11 |
| 39 | SoK: Agentic Skills | 2026 | 11 |
| 40 | Evolution of Tool Use | 2026 | 11 |
