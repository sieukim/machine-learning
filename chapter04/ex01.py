## 4장 선형 회귀로 이해하는 지도학습 심화 문제

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

## 4.1 다음은 P 자동차 회사의 차종과 마력, 그리고 평균연비를 나타내는 표이다.

# 데이터프레임 생성
data = pd.DataFrame()

# 차종 정보 추가
data.index = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
# 마력 정보 추가
data['power'] = [130, 250, 190, 300, 210, 220, 170]
# 연비 정보 추가
data['ratio'] = [16.3, 10.2, 11.1, 7.1, 12.1, 13.2, 14.2]

# 설명 변수 설정
X = data[['power']]
# 종속 변수 설정
y = data['ratio']

## 4.1.1. P 자동차 회사의 마력과 연비 사이에는 어떤 상관 관계가 있을까? 선형
##        회귀 분석을 통해서 선형 회귀 모델의 절편과 계수를 구하여라. 마지막으
##        로 이 선형 회귀 모델이 입력 마력 값에 대해 연비를 예측하는데 얼마나
##        적합한지 예측 점수를 출력해보라.

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

## 4.1.2. 위의 선형 회귀 모델을 바탕으로 270 마력의 신형 엔진을 가진 자동차를
##        개발하려 한다. 이 자동차의 연비를 선형 회귀 모델에 적용하여 다음과 같
##        이 구해 보자. 출력은 다음과 같이 소수점 둘째 자리까지 출력해 보자.

# 테스트 데이터
X_test = pd.DataFrame({'power': [270]})

# 테스트
pred = model.predict(X_test)

# 테스트 결과
result = f'{pred[0]:.2f}'

# 테스트 결과 출력
print(f'270 마력 자동차의 예상 연비: {result} km/l')