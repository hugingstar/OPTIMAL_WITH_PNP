# Optimal start for VRF systems
* 본 프로젝트는 VRF systems에 대한 최적 시작 및 최적 정지 시간을 결정하는 기술임.
* 문서작성일 : 2022.04.14(월) 

### 1. Config
* load_config.py : 딕셔너리의 key를 사용하여 값을 불러오는 코드
* identification.json : 딕셔너리 형태의 접속 관련 파일
* mapping.json : 방의 코드가 매핑된 자료

### 2. Data(데이터 저장 장소)
* 데이터 분석하기 위해 데이터를 저장하는 장소(biotData.py에서 저장된다.)

### 3. FeatureExtracting(특징 추출 인공지능 기법)
* 지도학습 및 비지도학습과 같은 인공지능 기반의 분류 방법 알고리즘을 개발하는 장소이다.
* FEATURE_SELECTION.py : 지도학습 기반 특징중요도 계산
* FEATURE_SELECTION_OPT.py : 지도 학습 기반 특징중요도 하이퍼파라미터 최적화 탑재
* DBSCAN.py : 비지도 학습 중에서 밀도를 기반으로하는 군집(Clustering) 기법 (다수의 유닛들이 있을 때 데이터 프로파일 의 특징을 분리)
  * Update(22.04.14) : 
* device_cluster.py : 클러스터링 기법들을 모아둔 백업용 파일

### 3. Pre-processing(데이터 전처리)
* biotData.py : 데이터 베이스에서 필요한 데이터 호출하여 저장할 때 메타데이터 Full Sentence로 만들어서 정리한다.
* 백업용 파일은 참고용으로 사용

### 4. Method
* DEEPGRAPH.py : 딥러닝 모델의 그래프를 만들고 예측 결과를 살펴본다.
    * Update (22.04.14) : gpu 사용 추가
    * Update (22.04.15) : AutoEncoder/Attention Layer 추가하였다. 
    * AutoEncoder 같은 경우에는 기존 Seq2seq 대비 성능이 좋다. 
    * Attention Layer는 현재 값이 예측을 못하는 상태이다.(변화 폭이 너무 큰 상태임.)
* SIEMENS.py : 회귀식 기반의 에어컨 최적가동시간을 예측하는 모델
    * Update (22.04.14) : 파일 추가
* VSENSORS.py : Pytorch를 사용하여 만든 간단한 회귀식 제작 방법

### 5. Results
* Results 파일은 분석 결과가 저장됨
* Ensemble/Deeplearning 등 다양한 폴더가 생성될 것이다.