# 데이터 분석

## 사용한 샘플

Signal

DYJetsToLL

Background

TTbar

---

# Truth 정의

Signal

Reco Muon들이

같은 Z boson에서 나온

반대 전하(OS)

Muon pair를 만들면

Signal

그 외는 모두

Background

---

# Dataset 크기

DYJetsToLL에서

총 이벤트 = 14,744,826

Signal = 13,234,058

Background = 1,510,768


TTbar에서

총 이벤트 = 579,494

Signal = 0

Background = 579,494


Merged했을 때

총 이벤트 = 15,324,320

Signal 비율 = 86.36%

---

# Dataset 분리

Train = 70%

Validation = 15%

Test = 15%

Stratified split 사용 
(한쪽에 signal이 편향되지 않도록 signal과 background를 각각 분리 후 비율 계산 뒤 dataset에 넣는다)

---

# Feature 분포

Outlier 확인 완료

Feature transform 적용

- pt → log
- Isolation → log
- dxy → signed log
- dz → signed log

이후 Standard normalization 수행

---

# 결과

Feature transform 이후

모든 feature가 안정적인 분포를 가짐.

Transformer 학습이 정상적으로 진행됨.