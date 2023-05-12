# DKU
## AI 특론

### 모형 설명
- TabNet
  -  Deep Neural Network와 DecisionTree-based 모델의 장점을 계승한 정형 데이터 분석을 위한 딥러닝 모형
  -  기존 기계 학습 모델보다 우수한 성능을 보인다고 하기에는 어려움이 있지만, 성능보다는 딥러닝의 장점이 필요할 때 주로 사용
  -  특징
    1) 전처리가 거의 필요 없고 최적화 알고리즘으로 경사하강법을 사용하는 구조로 end-to-end 학습에 유연하게 적용 가능
    2) Sequential attention을 사용하여 feature selection의 이유 추적 가능 -> interpretability 확보(local & global)
  
  <img src=https://user-images.githubusercontent.com/59715960/234817143-c58d5125-1f07-49a5-af9d-1805c03a20ea.png />
  
  - Encoder  
<img src=https://user-images.githubusercontent.com/59715960/234817915-8102e9be-7526-4f6c-8a11-807eb9ec40c5.png width="600" height="300"/>
  
   - Step의 여러 단계를 거치면서 각 Step 마다 feature selection mask를 구하여 각 Step 별 핵심 feature 파악 가능(local)
   - 모든 Step의 mask를 합하여 입력받은 전체 데이터에 대해서도 feature의 중요도 파악 가능(global)
  
  - Result
    <img src=https://user-images.githubusercontent.com/59715960/235048302-64b58d87-aabb-4ac0-a349-17ff95f7c836.png width="400" height="300"/> 

### 요구사항
- Python version 3.7.x, 3.8.x, 3.9.x, 3.10.x
- Tensorflow 2.x
(pytorch-tabnet 라이브러리로 구현이 되었지만, tensorflow를 사용하여 implementation 수행)

### 실행 방법

### pseudocode
  - Sparse max
    <img width="200" alt="스크린샷 2023-05-12 오후 1 46 21" src="https://github.com/KR-ESWord/DKU/assets/59715960/dc63986d-0f35-4c96-969b-5811484d81f0">
