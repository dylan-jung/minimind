# Week 3: 토크나이저 & 사전학습 (Pretraining)

> 목표: 텍스트가 어떻게 숫자로 변환되고, 모델이 언어를 "학습"하는 원리를 이해한다.
> 이번 주부터 직접 훈련을 돌려본다.

---

## 공식 학습 자료

이번 주부터 공식 Deep Mastery 과정의 Week 2~3에 해당한다.

| 공식 가이드 | URL |
|-------------|-----|
| Deep Mastery (전체 로드맵) | https://www.minimind.wiki/en/docs/guide/mastery |

공식 가이드 Week 2의 핵심:
- **Tokenizer Training** (2시간): `python scripts/train_tokenizer.py` 실행
- **Data Cleaning/Preprocessing** (4시간): `dataset/lm_dataset.py` 분석
- **Format Conversion** (2시간): pretrain / SFT / DPO 3가지 형식 이해

공식 가이드 Week 3의 핵심:
- **Pretraining** (4시간): causal LM 학습, loss/lr 모니터링, NaN/OOM 디버깅

---

## 1. BPE 토크나이저

### 논문

- **Neural Machine Translation of Rare Words with Subword Units** (Sennrich et al., 2016)
  - https://arxiv.org/abs/1508.07909
  - BPE(Byte Pair Encoding)의 원본 논문. Section 3이 핵심 알고리즘.

- **SentencePiece: A simple and language independent subword tokenizer** (Kudo & Richardson, 2018)
  - https://arxiv.org/abs/1808.06226
  - Google의 범용 토크나이저. Unigram LM 기반 서브워드 분할.

### 코드: `trainer/train_tokenizer.py`

### 핵심 개념

- **왜 토크나이저가 필요한가**: 모델은 숫자만 처리 → 텍스트를 정수 시퀀스로 변환해야 함
- **BPE 알고리즘**:
  1. 모든 문자를 개별 토큰으로 시작
  2. 가장 자주 등장하는 인접 토큰 쌍을 하나로 합침
  3. 원하는 vocab 크기가 될 때까지 반복
- **ByteLevel BPE**: 바이트 단위로 시작 → 어떤 언어/문자도 OOV(Out-of-Vocabulary) 없이 처리
- **Special tokens**: `<|im_start|>`, `<|im_end|>`, `<tool_call>`, `<think>` 등 — 모델의 행동을 제어하는 특수 토큰
- MiniMind vocab size: 6400 (매우 작음 — GPT-4는 ~100K)

### 읽기 포인트

1. Vocab 크기가 모델에 미치는 영향: 작으면 시퀀스가 길어지고, 크면 embedding 파라미터가 많아짐
2. `chat_template` 형식: `<|im_start|>system\n...<|im_end|>` — ChatML 포맷

---

## 2. Pretraining 데이터

### 코드: `dataset/lm_dataset.py` — `PretrainDataset`

### 핵심 개념

- **데이터 형식**: `pretrain_t2t_mini.jsonl` — 각 줄이 하나의 텍스트 샘플
- **Input과 Label의 관계**: 
  ```
  input:  [A, B, C, D, E]
  label:  [B, C, D, E, F]  ← 한 칸 shift
  ```
  모델은 "현재까지의 토큰을 보고 다음 토큰을 예측"하는 것을 반복
- **`ignore_index=-100`**: padding 등 loss를 계산하지 않을 위치에 -100을 넣으면 cross_entropy가 무시

### 읽기 포인트

1. `PretrainDataset`과 `SFTDataset`의 차이를 비교해보기
2. 데이터 전처리: truncation, padding 방식

---

## 3. Pretraining 학습 루프

### 논문 (배경 지식)

- **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019 — GPT-2 논문)
  - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  - Next-token prediction만으로 다양한 task를 학습할 수 있다는 핵심 인사이트.

- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020)
  - https://arxiv.org/abs/2001.08361
  - 모델 크기, 데이터 크기, 연산량과 성능의 관계. "왜 크게 만드는가"에 대한 답.

### 코드: `trainer/train_pretrain.py` + `trainer/trainer_utils.py`

```python
# 핵심 학습 루프 (train_pretrain.py line 23~58)
for step, (input_ids, labels) in enumerate(loader):
    # 1. Learning rate 스케줄링
    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)

    # 2. Forward pass (mixed precision)
    with autocast_ctx:
        res = model(input_ids, labels=labels)
        loss = res.loss + res.aux_loss
        loss = loss / args.accumulation_steps  # gradient accumulation

    # 3. Backward pass
    scaler.scale(loss).backward()

    # 4. Gradient clipping + optimizer step
    if step % args.accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

### 핵심 개념

**3-1. Cross-Entropy Loss**
```python
# model_minimind.py line 244~245
x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```
- logits의 마지막 토큰을 제외하고, labels의 첫 토큰을 제외 → 1칸 shift
- 모델 출력(확률 분포)과 정답(토큰 ID)의 차이를 측정

**3-2. Cosine Learning Rate Schedule**
```python
# trainer_utils.py line 41
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```
- 학습 초반에 높은 lr → 점진적으로 감소 → 안정적인 수렴
- `0.1 * lr`이 최저점 (완전히 0이 되지 않음)

**3-3. Mixed Precision Training (AMP)**
- `autocast_ctx`: fp16/bf16으로 forward 계산 → 메모리 절약, 속도 향상
- `GradScaler`: fp16에서 gradient underflow 방지

**3-4. Gradient Accumulation**
- GPU 메모리가 부족할 때 batch를 여러 step에 나눠서 gradient를 축적
- `loss / accumulation_steps`로 평균을 맞춤

**3-5. Gradient Clipping**
- `clip_grad_norm_`: gradient의 L2 norm이 임계값을 넘으면 잘라냄 → 학습 안정성

**3-6. 분산 훈련 (DDP)**
```python
# trainer_utils.py line 44~51
def init_distributed_mode():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
```
- 여러 GPU에서 동일 모델을 복제하고, gradient를 all-reduce로 동기화
- `DistributedSampler`로 데이터를 GPU별로 분배

---

## 이번 주 실습

### 필수

1. 데이터 다운로드:
   ```bash
   # ModelScope에서 pretrain_t2t_mini.jsonl 다운로드
   # https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
   # dataset/ 폴더에 저장
   ```

2. Pretrain 실행:
   ```bash
   # MPS (Mac) 또는 CPU로 소규모 테스트
   uv run python trainer/train_pretrain.py --device mps --epochs 1
   ```

3. `train_pretrain.py`를 읽으면서 위의 6가지 개념이 코드 어디에 해당하는지 표시해보기

### 선택

4. Loss curve를 관찰: wandb 또는 swanlab 연동하면 시각화 가능
5. `get_lr` 함수를 직접 플롯해서 cosine schedule 형태 확인:
   ```python
   import matplotlib.pyplot as plt
   steps = range(1000)
   lrs = [get_lr(s, 1000, 1e-4) for s in steps]
   plt.plot(steps, lrs)
   plt.xlabel('step'); plt.ylabel('lr')
   plt.show()
   ```
