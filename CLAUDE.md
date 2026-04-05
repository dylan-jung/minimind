# MiniMind

## Overview

MiniMind는 LLM(Large Language Model)을 처음부터 직접 만들어보는 교육용 프로젝트.
64M 파라미터의 초소형 언어모델을 PyTorch로 밑바닥부터 구현하며,
모든 핵심 알고리즘이 서드파티 라이브러리 없이 원시 구현되어 있다.

- Upstream: https://github.com/jingyaogong/minimind
- Fork: https://github.com/dylan-jung/minimind
- 공식 Wiki: https://www.minimind.wiki/en/

## Branch Strategy

- `master` — upstream sync 전용
- `study-base` — 본인 작업 base (uv 설정 + 학습 커리큘럼)
  - `study-xxx` — 코드 변경/실험용 (필요 시 study-base에서 분기)

## Project Structure

```
model/
  model_minimind.py   — 모델 전체 구조 (Attention, FFN, MoE, RoPE, RMSNorm)
  model_lora.py       — LoRA 구현 (66줄)

trainer/
  train_pretrain.py       — 사전학습
  train_full_sft.py       — SFT (Supervised Fine-Tuning)
  train_lora.py           — LoRA 미세조정
  train_dpo.py            — DPO (Direct Preference Optimization)
  train_ppo.py            — PPO (Proximal Policy Optimization)
  train_grpo.py           — GRPO (Group Relative Policy Optimization)
  train_distillation.py   — Knowledge Distillation
  train_agent.py          — Agentic RL (Tool Use + GRPO)
  train_tokenizer.py      — 토크나이저 학습
  trainer_utils.py        — 학습 유틸리티
  rollout_engine.py       — Rollout 엔진 (PPO/GRPO 공유)

dataset/
  lm_dataset.py       — 5가지 Dataset 클래스 (Pretrain, SFT, DPO, RLAIF, AgentRL)

scripts/
  serve_openai_api.py  — OpenAI API 호환 서버 (FastAPI)
  web_demo.py          — Streamlit 채팅 데모
  convert_model.py     — PyTorch ↔ Transformers(Qwen3) 변환, LoRA 합병
  eval_toolcall.py     — Tool Call 평가
  chat_api.py          — 채팅 API 예시

eval_llm.py            — CLI 추론 & 평가
docs/curriculum/       — 12주 학습 커리큘럼 (논문 40편 + 코드 매핑)
```

## Setup

```bash
uv sync            # 의존성 설치
uv run python ...  # 스크립트 실행
```

- Python 3.12, uv로 관리
- `--device mps` (M4 Pro 24GB에서 로컬 실행 가능)
- torch/torchvision은 dev dependency

## Model Architecture (Qwen3 aligned)

- GQA (Grouped-Query Attention): 8 Q heads, 4 KV heads
- SwiGLU FFN
- RoPE + YaRN (장문 외추)
- RMSNorm (Pre-LN)
- MoE: 4 experts, top-1 routing, load balancing loss
- Weight Tying (embed_tokens = lm_head)

## Data

- 다운로드: https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
- 최소 필요: `pretrain_t2t_mini.jsonl`, `sft_t2t_mini.jsonl`
- `dataset/` 폴더에 저장

## Conventions

- 코드는 중국어 주석이 포함되어 있음 (upstream 원본)
- 학습 커리큘럼 문서는 한국어로 작성
- 커밋 메시지는 한국어 또는 영어
