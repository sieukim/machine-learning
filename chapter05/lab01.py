## 5장 분류와 군집화로 이해하는 지도 학습과 비지도 학습 LAB 5-1

## 실습 목표
## 붓꽃 데이터에 대하여 sklearn에서 제공하는 k-평균 군집화 알고리즘을
## 적용하자. 이때 target 정보는 이용하지 말고, 4개의 특성값만을 이용
## 하여 군집화된 데이터의 레이블 정보를 출력하면 다음 출력과 같이 [1,
## 1, 1, ..., 0, 0, ..., 2, 2, 2, ...]로 레이블링 될 수 있다.
## 이와 같이 출력되는 이유는 군집화 알고리즘에서는 각각의 군집에 대한 
## 레이블만을 출력할 뿐 어느 군집이 setosa(0), versicolor(1), virgi
## nica(2)에 속하는지에 대한 target 정보가 없기 때문이다. 이제 이 
## 정보를 바탕으로 다시 레이블링을 하여 new_label을 만들어보자. 마지막
## 으로, 이 군집화 결과값과 원래 iris 데이터의 target과의 차이를 비교
## 하여 정확도를 다음과 같이 출력하도록 하자.

import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 데이터셋 적재
dataset = load_iris()

# 설명변수 설정
X = dataset.data

# k 설정
k = len(dataset.target_names)

# K-means 모델 객체 생성
model = KMeans(n_clusters=k)

# 학습
model.fit(X)

# 예측 => 레이블 생성
labels = model.predict(X)
print(f'군집화 결과 labels: {labels}')

# !! lables는 클러스팅된 데이터에 대한 레이블로 기존 target과 같지 않을 수도 있다.
# 따라서 lables를 삼등분하여 넘파이 bincount() 함수에 넣는다. bincount()는 객체의 
# 원소 중 0부터 최대값 범위의 정수값을 오름차순으로 정리한 뒤 각 원소에 대한 빈도수
# 를 반환한다. 따라서 이 함수는 0, 1, 2 값의 빈도수를 구하는데 사용한다. 예를 들어,
# [2, 2, 2, 4, 4]에 bincount()를 실행할 경우 결과는 [0, 0, 3, 0, 2]가 된다.

count_0 = np.bincount(labels[:50])
count_1 = np.bincount(labels[50:100])
count_2 = np.bincount(labels[100:])

# !! 넘파이 argmax() 함수를 활용하여 각 배열에 가장 빈번한 값을 각각 0, 1, 2에 대
# 응시키는 딕셔너리를 생성한다.
d = {0: np.argmax(count_0),
     1: np.argmax(count_1),
     2: np.argmax(count_2)}

# !! labels의 값을 new_labels에 복사한 후, 반복문을 실행하여 재레이블링을 수행한다.
new_labels = np.copy(labels)

for old, new in d.items():
    new_labels[labels == old] = new
    
print(f'재레이블링 후 new_labels: {new_labels}')

# 군집화 결과와 기존 target과의 차이를 비교하여 정확도를 출력한다.
print(f'iris 데이터 군집화 결과 정확도: {accuracy_score(dataset.target, new_labels)}')