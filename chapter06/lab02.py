##   6장  다양한 머신러닝 기법들: 다항 회귀, 결정 트리, SVM 
## lab1. 다항 회귀의 회귀 함수를 그려 보자.

## 목표
## 직선으로 표현하기 어려운 데이터를 화면에 그려보고,
## 이 데이터를 설명하는 회귀 함수를 다항 회귀를 이용하여 찾아보자.
## 그리고 회귀 함수를 데이터의 독립 변수 범위 내에서 가시화하자.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

## 데이터 url
file = 'https://github.com/dknife/ML/raw/main/data/nonlinear.csv'
## 데이터 적재 
df = pd.read_csv(file)

## 데이터 분포 확인 
plt.scatter(df['x'], df['y'])

## 데이터 설정 
X = df['x'].to_numpy().reshape(-1, 1)
y = df['y'].to_numpy()

## 설명변수의 범위 (0 ~ 1)
domain_X = np.linspace(0, 1, 100).reshape(-1, 1)

## Polynomial Features 객체 생성
poly = PolynomialFeatures(degree=3)

## 설명변수를 3차 회귀를 위해 변환
X_poly = poly.fit_transform(X)
domain_X_poly = poly.fit_transform(domain_X)

## 선형 회귀 모델 객체 생성
model = LinearRegression()

## 모델 학습
model.fit(X_poly, y)

## 모델 예측
pred = model.predict(domain_X_poly)

## 3차 회귀 함수 확인 
plt.scatter(df['x'], df['y'])
plt.scatter(domain_X, pred, color='r')