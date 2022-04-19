# Optimal with PnP : Optimal start for VRF systems
* 본 프로젝트는 실내 온도를 예측하는 모델을 만들기 위한 전반적인 머신러닝/딥러닝 방법이 있음.
* 문서작성일 : 2022.04.19(월)
* 비고 : 백업용 파일은 참고용으로 사용

### 1. 데이터 전처리(Pre-processing)
* biotData.py : 데이터 베이스에서 필요한 데이터 호출 
* 저장할 때는 메타데이터를 Full Sentence로 만들어서 데이터 저장 장소에 저장

### 2. 데이터 저장 장소(Data)
* 데이터 분석하기 위해 데이터를 저장하는 장소

### 3. 특징추출 머신러닝(FeatureExtracting)
* 머신러닝(지도학습,비지도학습) 분류 방법 알고리즘이 있다.
* FEATURE_SELECTION.py : 지도학습 기반 특징중요도 계산
* FEATURE_SELECTION_OPT.py : 지도 학습 기반 특징중요도 하이퍼파라미터 최적화 탑재
  * 상황 : 수정중이라 작동 안함. 
* DBSCAN.py : 비지도 학습 중에서 밀도를 기반으로하는 군집(Clustering) 기법 
  * Update(22.04.14) 다수의 유닛들이 있을 때 데이터 프로파일 의 특징을 분리
* device_cluster.py : 클러스터링 기법들을 모아둠.(백업)

### 4. Method
* DEEPGRAPH.py : 딥러닝 모델의 그래프를 만들고 예측 결과를 살펴본다.
    * Update (22.04.14) : gpu 사용 추가
    * Update (22.04.15) : AutoEncoder/Attention Layer 추가하였다.
    * Update (22.04.18) : 데이터의 index 명이 누락되는 문제를 보안하였다. 
    * 또한, 결측값 발생시 모델 생성이 안되는 문제를 보완하였다.
    * AutoEncoder 같은 경우에는 기존 Seq2seq 대비 성능이 좋다.(헌팅 문제는 해결 진행중)
    * Attention Layer는 현재 값이 예측을 못하는 상태이다.(변화 폭이 너무 큰 상태임.)
    * Update (22.04.18) : Attention Layer 온도를 너무 높거나 낮게 예측하는 문제 일부 해결
    * 현재 상태에서는 하이퍼파라미터 최적화를 진행하는 것이 예측성능 올리기에 가장 간단한 방법
    * 시간 주기성 부분을 수정
* SIEMENS.py : 회귀식 기반의 에어컨 최적가동시간을 예측하는 모델
    * Update (22.04.14) : 파일 추가
* VSENSORS.py : Pytorch를 사용하여 만든 간단한 회귀식 제작 방법

### 5. Results
* Results 파일은 분석 결과가 저장됨
* Ensemble : Feature importance 결과를 저장하는 디렉토리
* Deepmodel : 딥러닝 모델을 학습시킨 결과와 예측 테스트한 결과를 저장하는 디렉토리 
* Optimal : Optimal start 결과를 저장하는 디렉토리

### 6. Config
* load_config.py : 딕셔너리의 key를 사용하여 값을 불러오는 코드
* identification.json : 딕셔너리 형태의 접속 관련 파일
* mapping.json : 방의 코드가 매핑된 자료