# Optimal with PnP : Optimal start for VRF systems
* 본 프로젝트는 실내 온도를 예측하는 모델을 만들기 위한 전반적인 머신러닝/딥러닝 방법이 있음.
* 문서작성일 : 2022.04.24(화)
* 비고 : 백업용 파일은 참고용으로 사용

### 1. 데이터 전처리(Pre-processing)
* biotData.py : 데이터 베이스에서 필요한 데이터 호출
  * Update (22.04.23) : 분석 목적별로 저장 위치 업데이트(가상센서 목적용 별도로 저장)
  * 저장할 때는 메타데이터를 Full Sentence로 만들어서 데이터 저장 장소에 저장

### 2. 데이터 저장 장소(Data)
* 데이터 분석을 위한 데이터를 저장하는 장소

### 3. 특징추출 머신러닝(FeatureExtracting)
* FEATURE_SELECTION.py : 지도학습 기반 특징중요도 계산
* FEATURE_SELECTION_OPT.py : 지도 학습 기반 특징중요도 하이퍼파라미터 최적화 탑재 (수정중)
* DBSCAN.py : 비지도 학습 중에서 밀도를 기반으로하는 군집(Clustering) 기법 
  * Update(22.04.14) 다수의 유닛들이 있을 때 데이터 프로파일 의 특징을 분리
* device_cluster.py : 클러스터링 기법들을 모아둠.(백업)

### 4. Method
* DEEPGRAPH.py : 딥러닝 모델의 그래프를 만들고 예측 결과를 살펴본다.
  * Update (22.04.14) : gpu 사용 추가
  * Update (22.04.15) : AutoEncoder/Attention Layer 추가하였다.
  * Update (22.04.18) : 데이터의 index 명이 누락되는 문제를 보완, 결측값 발생시 모델 생성이 안되는 문제를 보완, 
  AutoEncoder 같은 경우에는 기존 Seq2seq 대비 성능이 좋다.(헌팅 문제는 해결 진행중), 
  Attention Layer는 현재 값이 예측을 못하는 상태이다.(변화 폭이 너무 큰 상태임.)
  * Update (22.04.18) : Attention Layer 온도를 너무 높거나 낮게 예측하는 문제 일부 수정
  현재 상태에서는 하이퍼파라미터 최적화를 진행하는 것이 예측성능 올리기에 가장 간단한 방법,시간 주기성 부분을 수정
  * Update 
* SIEMENS.py : 회귀식 기반의 에어컨 최적가동시간을 예측하는 모델
  * Update (22.04.14) : 파일 추가
* VSENSORS.py : 실외기 데이터를 사용한 가상센서
  * Update(22.04.23) : 냉매 질량 유량 가상센서, 전력 가상센서, 증발기 용량 가상센서, 응축기 용량 가상센서 추가
  * Update(22.04.24) : 열관류율 가상센서, 열교환기 출구 온도 가상센서
  * Update(22.04.25) : VSENS_CAPA_EVAP의 Try-Except에서 Except 값 부분 수정 
  
### 5. Results
* Results/{File}
* Ensemble : Feature importance 결과를 저장 장소
* Deepmodel : 딥러닝 모델을 학습시킨 결과와 예측 테스트한 결과를 저장 장소 
* Optimal : Optimal start 결과를 저장 장소
* VirtualSensor : 가상센서 저장 장소

### 6. Config
* load_config.py : Configuration file의 값을 불러오는 파일
* identification.json : identification 딕셔너리(비공개)
* mapping.json : 방의 코드가 매핑된 자료(비공개)