## 4장 선형 회귀로 이해하는 지도학습 심화 문제

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

import seaborn as sns

## 4.2. 자동차의 연비에 영향을 미치는 요소는 마력뿐만 아니라 총중량도 중요한 요소가
##      될 것이다. 다음은 P 자동차 회사의 차종과 마력뿐만 아니라 자동차의 총중량을
##      추가한 표이다.

# 데이터프레임 생성
data = pd.DataFrame()

# 차종 정보 추가
data.index = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
# 마력 정보 추가
data['power'] = [130, 250, 190, 300, 210, 220, 170]
# 총중량 정보 추가
data['weight'] = [1900, 2600, 2200, 2900, 2400, 2300, 2100]
# 연비 정보 추가
data['ratio'] = [16.3, 10.2, 11.1, 7.1, 12.1, 13.2, 14.2]

# 설명 변수 설정
X = data[['power', 'weight']]
# 종속 변수 설정
y = data['ratio']

## 4.2.1. 위의 자료를 바탕으로 적절한 선형 회귀 모델을 구현하라. 이 모델의 계수와 
##        절편, 예측 점수를 출력하라.

# 선형 회귀 모델 생성
model = LinearRegression()

# 학습
model.fit(X, y)

# 예측 점수
score = model.score(X, y)

# 계수 출력
print(f'계수\t: {model.coef_}')
# 절편 출력
print(f'절편\t: {model.intercept_}')
# 예측 점수 출력
print(f'예측 점수\t: {score}')

## 4.2.2. 위의 선형 회귀 모델을 바탕으로 270 마력의 신형 엔진을 가진 총중량 2500kg
##        의 자동차를 개발하려 한다. 이 자동차의 연비를 선형 회귀 모델에 적용하여 다음
##        과 같이 구해 보라.

# 테스트 데이터
X_test = pd.DataFrame({'power': [270],
                       'weight': [2500]})

# 테스트
pred = model.predict(X_test)

# 테스트 결과
result = f'{pred[0]:.2f}'

# 테스트 결과 출력
print(f'270 마력 자동차의 예상 연비: {result} km/l')

## 4.2.3. 마력과 총중량, 연비 사이의 상관관계를 쌍플롯(pairplot)으로 그려 보라. 
sns.pairplot(data)