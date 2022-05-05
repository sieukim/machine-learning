## 5장 분류와 군집화로 이해하는 지도 학습과 비지도 학습 미니 프로젝트 A1

## 목표
## 이미지에 있는 잡음을 제거하는 일을 k-NN을 이용해 해보려고 한다. 이것은 잡음이 있는 
## 이미지를 읽어 아래 그림과 같이 픽셀 개수만큼의 분류 결과를 내어 놓고, 이들을 모아
## 이미지를 구성하는 것이다. 이렇게 분류 결과를 여러개 내어 놓는 것을 다중 출력 분류라고
## 한다.

import matplotlib.pyplot as plt
import numpy as np


from sklearn.neighbors import KNeighborsClassifier

## 1. 미리 준비할 것 

# 이미지 읽기
from skimage.io import imread
#이미지 크기 변경
from skimage.transform import resize
# 이미지 증강
from keras.preprocessing.image import ImageDataGenerator

## 2. 이미지 읽어 들이기

# 이미지 주소
url = 'https://github.com/dknife/ML/raw/main/data/Proj1/40/'

# 이미지 사이즈, 채널
imgR, imgC, channel = 24, 24, 3

# 이미지 리스트
images = []

for i in range(40):
  # 이미지 파일 이름
  file = url + 'img{0:02d}.jpg'.format(i + 1)
  # 이미지 읽기
  img = imread(file)
  # 이미지 크기 변경
  img = resize(img, (imgR, imgC, channel))
  # 이미지 리스트에 추가
  images.append(img)

# 이미지 시각화 함수
def show_images(nRow, nCol, img):
  fig = plt.figure()
  fig, ax = plt.subplots(nRow, nCol, figsize=(nCol, nRow))
  for i in range(nRow):
    for j in range(nCol):
      if nRow <= 1:
        axis = ax[j]
      else:
        axis = ax[i, j]
      axis.get_xaxis().set_visible(False)
      axis.get_yaxis().set_visible(False)
      axis.imshow(img[i * nCol + j])

# 모든 이미지 시각화
show_images(4, 10, images)


## 3. 훈련용 데이터와 검증용 데이터 분리
##    - 기계학습은 데이터를 통해 훈련한 뒤에 얻은 동작 방법을 훈련에 사용한 적이 없는
##      새로운 데이터를 적용하여 일반화할 수 있다는 가정을 갖느다. 그런데 훈련 단계
##      에서 일반화 능력을 검증하지 않는다면, 일반화 능력이 확인되지 않은 모델을 배포
##      혹은 사용하게 된다. 따라서 훈련 과정에서는 가지고 있는 데이터의 일부만을 사용
##      하고, 일반화 능력은 검증용 데이터를 사용하여 검증한다. 이러한 개념에 따라 가
##      지고 있는 데이터를 훈련용 데이터와 검증용 데이터로 분리하여 활용한다. 

# 훈련용 데이터 
X = np.array(images[:30])
# 검증용 데이터
X_test = np.array(images[30:])

# 훈련용 데이터(이미지) 시각화
show_images(3, 10, X)
# 검증용 데이터(이미지) 시각화
show_images(1, 10, X_test)


## 4. 입력 데이터 준비
##    - 현재 데이터는 훈련 데이터와 검증 데이터 모두 잡음이 없는 깨끗한 이미지를 
##      레이블로 가지며, 훈련 과정을 통해 잡음이 섞인 입력 데이터를 이러한 깨끗
##      한 이미지 데이터로 변환하도록 해야 한다. 하지만 현재는 입력 데이터 또한
##      깨끗한 이미지 데이터이므로, 훈련을 위해 직접 잡음을 섞은 데이터로 변환하는
##      과정이 필요하다.
##
##    - 잡음은 넘파이를 이용하여 쉽게 만들 수 있다. 잡음의 형식은 배열과 같은 차
##      원을 가져야 하므로 (데이터 개수, 이미지 행 수, 이미지 열 수, 채널 수)의
##      차원을 가져야 한다. 생성한 잡음은 적절한 크기로 조절한 후 원래 이미지 배
##      열과 더함으로써 잡음이 섞인 이미지를 생성할 수 있다. 

# 잡음이 섞인 훈련용 데이터
X_noisy = X + np.random.randn(len(X), imgR, imgC, channel) * 0.1
X_noisy = np.clip(X_noisy, 0, 1)
# 잡음이 섞인 검증용 데이터
X_test_noisy = X_test + np.random.randn(len(X_test), imgR, imgC, channel) * 0.1
X_test_noisy = np.clip(X_test_noisy, 0, 1)

# 잡음이 섞인 훈련용 데이터(이미지) 시각화
show_images(3, 10, X)
# 잡음이 섞인 검증용 데이터(이미지) 시각화
show_images(1, 10, X_test)


## 5. 분류기 입출력 데이터 형식에 맞추어 훈련하기 
##    - K-NN 분류기는 이미지와 같은 2차원 이상의 데이터를 다루지 않고, 모든 입력 데이
##      터를 1차원 벡터 데이터로 취급한다. 따라서 입력 이미지 데이터 배열을 (이미지 수,
##      이미지 픽셀 수)의 형태로 변형해야 한다. 

# 깨끗한 훈련용 데이터 형식 변환
X_flat = np.array(X.reshape(-1, imgR * imgC * channel) * 255, 
                dtype=np.uint)

# 잡음이 섞인 훈련용 데이터 형식 변환
X_noisy_flat = X_noisy.reshape(-1, imgR * imgC * channel)

# k-nn 모델 객체 생성
model = KNeighborsClassifier()

# 학습(잡음 제거 모델)
model.fit(X_noisy_flat, X_flat)

# 예측
pred = model.predict(X_noisy_flat)
pred = pred.reshape(-1, imgR, imgC, channel)

# 잡음 제거 모델 예측 결과 시각화
show_images(3, 10, pred)


## 6. 데이터를 증강하여 훈련 효과 높이기
##    - 부족한 데이터 문제를 극복하기 위해 기존의 데이터를 변형하여 수를 늘린다. 

# 데이터 증강 횟수
n_augmentation = 100

# 증강된 데이터를 덧붙일 잡음이 섞인 훈련용 데이터
X_noisy_aug = X + np.random.rand(len(X), imgR, imgC, channel) * 0.2
y_label = np.array(X * 255, dtype=np.unit)
y = y_label

# 데이터 증강 수행
for _ in range(n_augmentation):
    # 증강을 위해 잡음을 섞은 데이터
    noisy_data = X + np.random.randn(len(X), imgR, imgC, channel) * 0.2
    # 데이터 덧붙이기
    X_noisy_aug = np.append(X_noisy_aug, noisy_data, axis=0)
    y = np.append(y, y_label, axis=0)

# 데이터 정리 (상한 - 1, 하한 - 0)
X_noisy_aug = np.clip(X_noisy_aug, 0, 1)

# 증강된 데이터 시각화
show_images(1, 10, X_noisy_aug[0:300:30])

# 증강된 데이터를 활용해 재훈련
X_noisy_aug_flat = X_noisy_aug.reshape(-1, imgR * imgC * channel)
y_flat = y.reshape(-1, imgR * imgC * channel)

model.fit(X_noisy_aug_flat, y_flat)

pred = model.predict(X_noisy_flat)
pred = pred.reshape(-1, imgR, imgC, channel)

# 재훈련 결과 시각화
show_images(3, 10, pred)


## 7. 검증 데이터로 일반화 능력 살피기

# 난수 정수 생성
rndidx = np.random.randint(0, 20)

# 깨끗한 이미지를 10개 추출하고, 표준편차 0.4의 강한 잡음을 섞기
data = X[rndidx:rndidx+10] + np.random.randn(10, imgR, imgC, channel)
data = np.clip(data, 0, 1)
data_flat = data.reshape(-1, imgR * imgC * channel)

# 예측 
pred = model.predict(data_flat)
pred = pred.reshape(-1, imgR, imgC, channel)
red = np.clip(pred, 0, 255)

# 예측 결과 시각화
show_images(1, 10, pred)


## 8. 데이터 증강으로 일반화 능력을 높여보자
##    - 일반화 능력을 높이기 위해서는 데이터를 증강해야 한다. 이번에는 이미지를 다양하게
##      증강시켜 볼 것이며, 이미지를 다양하게 변형하여 새로운 데이터를 만들어내는 클래스
##      ImageDataGenerator를 활용할 것이다. 이 클래스는 Iterator를 활용하여 이
##      미지를 계속해서 생성할 수 있다. Iterator는 flow() 함수에 원본 이미지를 넣어
##      생성할 수 있으며, next() 함수를 호출할 때마다 생성된 이미지 데이터를 반환한다.

# ImageDataGenerator 객체 생성
image_generator = ImageDataGenerator(rotation_range=30,     # 회전
                                     zoom_range=0.1,        # 확대
                                     shear_range=0.1,       # 축소
                                     width_shift_range=0.1, # 기울이기
                                     height_shift_range=0.1,# 기울이기
                                     horizontal_flip=True,  # 좌우반전
                                     vertical_flip=True)    # 상하반전

# 증강에 사용될 원본 이미지 데이터
y_aug = X.reshape(-1, imgR, imgC, channel)

# 증강에 사용될 원본 이미지 데이터 개수
nData = y_aug.shape[0]

# Iterator 생성
it = image_generator.flow(y_aug)

# 데이터 증강
X_aug = y_aug + np.random.randn(nData, imgR, imgC, channel) * 0.1

# 증강 횟수
n_augmentation = 500

# 증강 수행
for _ in range(n_augmentation):
    # 새로운 데이터
    new_y = it.next()
    # 잡음 섞인 새로운 데이터
    new_X = new_y + np.random.randn(nData, imgR, imgC, channel) * 0.1
    # 데이터 덧붙이기
    y_aug = np.append(y_aug, new_y)
    X_aug = np.append(X_aug, new_X)

# 차원 변경
y_aug = np.array(y_aug * 255, dtype=np.uint)
y_aug = y_aug.reshape(-1, imgR, imgC, channel)
X_aug = X_aug.reshape(-1, imgR, imgC, channel)

# 데이터 정리
y_aug = np.clip(y_aug, 0, 255)
X_aug = np.clip(X_aug, 0, 1)

# 증강된 데이터 시각화
show_images(3, 10, y_aug[30:])


## 9. 새로 학습하고 검증용 데이터 적용하기

# 차원 변경
X_aug_flat = X_aug.reshape(-1, imgR * imgC * channel)
y_aug_flat = y_aug.reshape(-1, imgR * imgC * channel)

# 학습
model.fit(X_aug_flat, y_aug_flat)

# 검증 데이터에 대한 예측
pred = model.predict(X_test_noisy.reshape(-1, imgR * imgC * channel))
pred = pred.reshape(-1, imgR, imgC, channel)

# 예측 결과 시각화
show_images(1, 10, pred)
