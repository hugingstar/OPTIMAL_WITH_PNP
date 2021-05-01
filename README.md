# Optimal start for VRF systems
    * 본 프로젝트는 VRF systems에 대한 최적 시작 및 최적 정지 시간을 결정하는 기술임.
    
### 1. Pre-processing(Preprocessing/pre_processing.py)
    * 전처리(Pre-processing) : 오리지널 데이터에는 수많은 정보가 있는데 그 중에 중요한 정보만을 남긴 서브셋을 생성.
    * 전처리도 세부적으로 나누어져야 하며, 1단계로는 장기간의 데이터 중 당일 하루의 데이터만을 남긴다.
    * 2단계로는 전원이 켜져서 오늘의 운전 데이터를 기반으로 다음날의 최적 가동시간을 예측하는 과정이 필요하므로, 
      추가적인 데이터 전처리가 필요함.(초기 온도 값, 나중 온도 값 등이 필요함.)

### 2. Optimal method
    * 예측 시간을 예측하는 방법은 세분화 될 수 있음.
    * 이 부분은 다음 Commit에 업데이트가 될 예정임.