##   6장  다양한 머신러닝 기법들: 다항 회귀, 결정 트리, SVM 
## lab2. 꽃받침의 너비와 길이로 결정트리 만들기 

## 목표
## 붓꽃 데이터를 그대로 사용하여 결정 트리를 만들면, 꽃잎의 길이와 너비만이 분류 기준으로 사용된다.
## 꽃받침의 너비와 길이만 가지고 결정 트리를 만들어 보고, 전체 속성을 사용했을 때 왜 이 기준이 
## 선택되지 않았는지 생각해 보자. 

## -> 꽃받침 정보만을 이용하여 결정 트리를 만든 경우, 지나치게 많은 분할이 일어난다. 
##    각 노드의 지니 불순도도 빠르게 감소시키지 못 한다. 따라서 꽃잎 정보를 이용하여
##    결정 트리가 만들어진 것이다. 

## 데이터 적재
from sklearn.datasets import load_iris
iris = load_iris()

## 데이터 설정
X, y = iris.data[:, :2], iris.target

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
    # 꽃받침 정보 사용 
    feature_names=iris.feature_names[:2], # 속성 이름 리스트 
)

## dec_tree_for_iris.dot -> dec_tree_for_iris.jpg 로 변환 필요
## dot -Tjpg dec_tree_for_iris.dot -o dec_tree_for_iris.jpg

## 결정 트리 시각화 이미지 출력
import matplotlib.pyplot as plt
dec_tree_img = plt.imread('./dec_tree_for_iris.jpg')
plt.figure(num=None, figsize=(12, 8), dpi=80,
           facecolor='w', edgecolor='k')
plt.imshow(dec_tree_img)
