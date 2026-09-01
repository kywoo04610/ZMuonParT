# ZMuonParT

CMS Open Data를 이용하여

**재구성된(reconstructed) 뮤온 이벤트가 Z→μμ 신호인지 아닌지를 분류하는 Particle Transformer(ParT) 프로젝트**

---

# Data

본 프로젝트는 CERN Open Data Portal을 통해 공개된 **CMS 2016 Open Data**
(NANOAOD / NANOAODSIM)를 사용한다.

대용량 ROOT 파일은 GitHub repository에 포함하지 않는다.
대신 본 분석에서 실제로 사용한 ROOT 파일들의 XRootD 경로를
`dataset/` 디렉터리에 저장하였다.

## Monte Carlo

### DYJetsToLL

Drell-Yan Monte Carlo sample:

`/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM`

CMS Open Data:

https://opendata.cern.ch/record/35671

사용한 ROOT 파일 목록:

- `dataset/DYJetsToLL_url1.txt`
- `dataset/DYJetsToLL_url2.txt`

### TTJets

Top-antitop Monte Carlo sample:

`/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM`

CMS Open Data:

https://opendata.cern.ch/record/67733

사용한 ROOT 파일 목록:

- `dataset/TTbar_url1.txt`
- `dataset/TTbar_url2.txt`
- `dataset/TTbar_url3.txt`

## Real collision data

### SingleMuon Run2016H

2016 CMS Run H SingleMuon dataset:

`/SingleMuon/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD`

CMS Open Data:

https://opendata.cern.ch/record/30563

사용한 ROOT 파일 목록:

- `dataset/SingleMuon_url1.txt`
- `dataset/SingleMuon_url2.txt`
- `dataset/SingleMuon_url3.txt`
- `dataset/SingleMuon_url4.txt`
- `dataset/SingleMuon_url5.txt`

ROOT 파일은 CMS Open Data public XRootD endpoint를 통해 접근하였다:

`root://eospublic.cern.ch/`

다운로드 관련 코드는 다음 파일에 있다.

- `src/download.py`
- `src/download_real.py`

---

# 프로젝트 목표

Reco Muon 정보를 입력으로 사용하여

- Signal
  - Z → μ⁺μ⁻

- Background
  - Z가 아닌 모든 이벤트

를 분류하는 딥러닝 모델을 구축한다.

최종 목표는 **Particle Transformer (ParT)** 를 구현하고 기존 모델과 성능을 비교하는 것이다.

---

# 현재 진행 상황

완료

- Truth matching
- Dataset 생성
- Train / Validation / Test 분리
- Feature 전처리
- Normalization
- Baseline MLP 구현
- Simple Transformer 구현
- ROC / AUC 평가

진행 예정

- Feature importance 분석
- Pairwise feature
- Pairwise Attention Bias
- Full Particle Transformer 구현

---

# 데이터 흐름

ROOT

↓

dataset.py

↓

merge_dataset.py

↓

train / valid / test dataset

↓

feature_transform.py

↓

normalization.py

↓

train.py

↓

evaluate.py

↓

ROC

---

# 현재 입력 feature

각 Muon마다

- Muon_pt
- Muon_eta
- Muon_phi
- Muon_pfRelIso04_all
- Muon_dxy
- Muon_dz

사용

최대 Muon 개수는

```
8
```

Padding + Mask를 사용한다.

---

# 현재 성능

| 모델 | AUC |
|------|------|
| Baseline | 0.9512 |
| Transformer (5 epoch) | 0.9735 |

Transformer가 Baseline보다 background rejection에서 큰 향상을 보였다.
