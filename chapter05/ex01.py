## 5장 분류와 군집화로 이해하는 지도 학습과 비지도 학습 심화문제

## 다음은 철수네 동물병원에 치료를 받은 개의 종류와 그 크기 데이터이다. 이 데이터를
## 바탕으로 k-NN 알고리즘을 적용해 보자.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

# 닥스훈트 길이와 높이 데이터
length_0 = [75, 77, 83, 81, 73, 99, 72, 83]
height_0 = [24, 29, 19, 32, 21, 22, 19, 34]

# 사모예드 길이와 높이 데이터
length_1 = [76, 78, 82, 88, 76, 83, 81, 89]
height_1 = [55, 58, 53, 54, 61, 52, 57, 64]

# 말티즈 길이와 높이 데이터
length_2 = [35, 39, 38, 41, 30, 57, 41, 35]
height_2 = [23, 26, 19, 30, 21, 24, 28, 20]


## 5.1. 위의 정보를 바탕으로 닥스훈트를 0, 사모예드를 1, 말티즈를 2로 레이블링하여
##      데이터와 레이블을 각각 생성하도록 하여라. 그리고 다음과 같이 각 견종별 데이
##      터를 2차원 배열로 만들어 출력하라. 

# 닥스훈트 데이터와 레이블
data_0 = np.column_stack((length_0, height_0))
label_0 = np.array([0.] * len(data_0)) 

# 사모예드 데이터와 레이블
data_1 = np.column_stack((length_1, height_1))
label_1 = np.array([1.] * len(data_1))

# 말티즈 데이터와 레이블
data_2 = np.column_stack((length_2, height_2))
label_2 = np.array([2.] * len(data_2))

# 각 견종별 데이터를 2차원 배열로 만들어 출력
print(f'닥스훈트(0): {data_0.tolist()}')
print(f'사모예드(1): {data_1.tolist()}')
print(f'말티즈(2): {data_2.tolist()}')


## 5.2. k 값이 3일 때, K-NN 분류기의 분류 결과 목표값과 예측결과를 혼동행렬로
##      표시하라.

# 모든 견종 데이터와 레이블
data = np.concatenate((data_0, data_1, data_2))
label = np.concatenate((label_0, label_1, label_2))

# 모든 견종 클래스
dogs = {0: '닥스훈트',
        1: '사모예드', 
        2: '말티즈'}

# k가 3인 k-nn 모델 객체 생성
model = KNeighborsClassifier(n_neighbors=3)

# 학습
model.fit(data, label)

# 예측
pred = model.predict(data)

# 목표값과 예측결과를 혼동 행렬로 출력
cm = confusion_matrix(y_true=label, y_pred=pred)
print(cm)


## 5.3. 다음과 같은 개의 길이, 높이 데이터 A, B, C, D에 대하여 각각 n_neighbors를 
##      3, 5, 7로 하여 아래와 같이 분류하고 그 분류 결과를 출력하여라. 

# n_neighbors 리스트
n_neighbors = [3, 5, 7]

# k-nn 모델 객체 리스트
models = []

for k in n_neighbors:
    # k가 k인 k-nn 모델 객체 생성
    model = KNeighborsClassifier(n_neighbors=k)
    # 학습
    model.fit(data, label)
    # 모델 객체 리스트에 추가
    models.append(model)

# 테스트 데이터 A, B, C, D에 대한 정보 리스트
test_data_length = [58, 80, 80, 75]
test_data_height = [30, 26, 41, 55]
test_data = np.column_stack((test_data_length, test_data_height))
test_data_names = ['A', 'B', 'C', 'D']

# 주어진 테스트 데이터 데이터에 대하여 모든 모델의 분류 결과를 출력하는 함수
def print_result(data, 
                 data_name, 
                 models=models, 
                 n_neighbors=n_neighbors):
    print(f'{data_name} 데이터 분류 결과')
    
    for i in range(len(models)):
        # 분류 결과
        result = models[i].predict(data)
        print(f'{data_name} {data}: n_neighbors가 {n_neighbors[i]}일 때: {dogs[result[0]]}')
    
    print('\n')
    
# 모든 테스트 데이터에 대하여 모든 모델의 분류 결과를 출력
for i in range(len(test_data)):
    print_result(test_data[i].reshape(-1, 2), test_data_names[i])
    

## 5.4. 5.3.의 결과로 보아 위의 데이터 중에서 k값에 영향을 받지 않는 데이터는 무엇이며,
##      그 이유는 무엇인지 서술하여 보아라.
##   -> 데이터 B와 D가 k값에 영향을 받지 않는다. 따라서 모든 모델에 대한 분류 결과가 
##      각각 항상 닥스훈트와 사모예드가 된다. 


## 5.5 5.3.번 데이터를 산포도 그래프로 그려서 다음과 같이 A, B, C, D 데이터를 나타내어 
##     보자. 이때 가로축은 개의 길이로 두고, 세로축은 높이로 두자. 

# 모든 강아지에 대한 길이와 높이 데이터
data_length = [length_0, length_1, length_2]
data_height = [height_0, height_1, height_2]

# 강아지 데이터 색상 리스트
data_colors = ['red', 'blue', 'green']

# 강아지 데이터 이름 리스트
data_names = ['Dachshund', 'Samoyed', 'Maltese']

# 강아지 데이터 마커 리스트
data_markers = ['o', '^', 's']

# 모든 강아지 데이터에 대한 scatter
for i in range(3):
  plt.scatter(data_length[i], data_height[i], c=data_colors[i], label=data_names[i], marker=data_markers[i])

# 테스트 데이터 색상 리스트
test_data_colors = ['purple', 'grey', 'skyblue', 'green']

# 모든 테스트 데이터에 대한 scatter
for i in range(len(test_data)):
  plt.scatter(test_data[i][0], test_data[i][1], c=test_data_colors[i], label=test_data_names[i], s=300)

plt.xlabel('Length')
plt.ylabel('Height')
plt.title('Dog Size')
plt.legend(loc='upper left')

plt.show()


## 5.6. 원래의 데이터와 A, B, C, D 데이터를 모두 포함한 dog_data를 만들자. 이때 데이터가
##      가진 모든 레이블을 삭제하고,  k-means 알고리즘을 적용하여 클러스터링을 수행하고, 
##      다음과 같이 k가 2, 3, 4일 때의 수행 결과를 시각화하도록 하자.

# 원래 데이터 + A, B, C, D
dog_data = np.concatenate((data, test_data))

# n_clusters 리스트
n_clusters = [2, 3, 4]

# k-means 모델 객체 리스트
models = []

# 레이블 리스트
labels = [] 

for k in n_clusters:
    # k가 k인 k-means 모델 객체 생성
    model = KMeans(n_clusters=k)
    # 학습
    model.fit(dog_data)
    # 예측 => 레이블 생성
    label = model.predict(dog_data)
    # 모델 객체 리스트에 추가
    models.append(model)
    # 레이블 리스트에 추가
    labels.append(label)
    
# matplotlib 활용 시각화 템플릿 함수
def show_plot(x,
              y,
              c,
              label=None,
              xlabel=None,
              ylabel=None,
              title=None,
              loc=None):
    plt.scatter(x, y, c=c, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    loc and plt.legend(loc=loc)
    plt.show()

# 2차원 리스트를 1차원 리스트로 변환하는 함수
convert = lambda x: [elem for row in x for elem in row]

# 강아지 데이터 길이와 높이 리스트
dog_length = convert(data_length) + test_data_length
dog_height = convert(data_height) + test_data_height

# 시각화
show_plot(x=dog_length,
          y=dog_height,
          c='blue',
          label='no labeled data',
          xlabel='Length',
          ylabel='Height',
          title='Dog data without label',
          loc='upper left')

# 강아지 데이터 색상 리스트 
dog_colors = np.array(['red', 'green', 'blue', 'purple'])

# 각 모델의 결과에 대해 시각화 수행
for i in range(len(labels)):
    show_plot(x=dog_data[:, 0],
              y=dog_data[:, 1],
              c=dog_colors[labels[i]],
              title=f'K-Means Clustering, k={n_clusters[i]}')
    
