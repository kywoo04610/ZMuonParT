# 모델 설계

## 입력 데이터

한 이벤트

```
(8 × 6)
```

- 최대 Muon 개수 : 8
- Feature 개수 : 6

Mask

```
(8)
```

Padding된 Muon은 Mask=False

---

# Feature

각 Muon

- pt
- eta
- phi
- isolation
- dxy
- dz

---

# Feature 전처리

Muon_pt

```
log1p(pt)
```

Muon_eta

```
변환 없음
```

Muon_phi

```
변환 없음
```

Muon_pfRelIso04_all

```
log1p(x)
```

Muon_dxy

```
sign(x) × log1p(|x|)
```

Muon_dz

```
sign(x) × log1p(|x|)
```

---

# Normalization

Train dataset에서

평균(mean)

표준편차(std)

계산

↓

Validation / Test도 동일한 값을 사용

---

# Baseline

Muon MLP

↓

Average Pooling

↓

Classifier

AUC

0.9512

---

# Simple Transformer

Input Projection

↓

CLS Token

↓

Transformer Encoder

↓

Classifier

Embedding

128

Head

4

Layer

2

FeedForward

256

AUC

0.9735

---

# 최종 목표

Particle Transformer 구현

추가 예정

- Pairwise feature
- Pairwise attention bias
- Lorentz embedding