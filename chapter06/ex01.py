##  6장  다양한 머신러닝 기법들: 다항 회귀, 결정 트리, SVM 
## 6.9. 사이킷런의 결정 트리로 붓꽃 분류하기

## 데이터 적재
from sklearn.datasets import load_iris
iris = load_iris()

## 데이터 설정
X, y = iris.data, iris.target

## 결정 트리 객체 생성
from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier(max_depth=3)

## 데이터에 맞는 결정 트리 객체 생성
dec_tree.fit(X, y)

## 결정 트리 시각화 이미지 생성
from sklearn.tree import export_graphviz
export_graphviz(
    dec_tree,                             # 결정 트리
    out_file=("./dec_tree_for_iris.dot"), # 결과 파일 이름
    feature_names=iris.feature_names,     # 속성 이름 리스트
)

## dec_tree_for_iris.dot -> dec_tree_for_iris.jpg 로 변환 필요
## dot -Tjpg dec_tree_for_iris.dot -o dec_tree_for_iris.jpg

## 결정 트리 시각화 이미지 출력
import matplotlib.pyplot as plt
dec_tree_img = plt.imread('./dec_tree_for_iris.jpg')
plt.figure(num=None, figsize=(12, 8), dpi=80,
           facecolor='w', edgecolor='k')
plt.imshow(dec_tree_img)
