import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('/Users/mj/문서 - 강민지의 MacBook Pro/machine-learning-repo/machine_learning_base/data/train(1).csv')
test_df = pd.read_csv('/Users/mj/문서 - 강민지의 MacBook Pro/machine-learning-repo/machine_learning_base/data/test(1).csv')

print(train_df)

print(test_df)

"""
타이타닉데이터 합쳐서 분석하기
- Name이라는 피처가 단순하게 이름일 수 있지만, 결혼유무나 성별을 확인할 수 있는 하나의 피처로도 바라볼 수 있다.
- 이 피처를 통해 우리가 결혼이나 다른 피처를 추가해서 만들 수 있다.

- Name 어떤 식으로 가공할까?
- 문자열, 정규표현식 이용해서 피처를 가공할 수 있다.
- 정규표현식을 이용한 피처 가공 (Name 피처)
"""

train_df = train_df.drop(['Ticket','Cabin'], axis=1)
test_df = test_df.drop(['Ticket','Cabin'], axis=1)
combine = [train_df, test_df]

print(combine)

"""
Name 컬럼 전처리
- str 문법
- 문자열 데이터에 대한 전처리 작업 진행시 사용해
- 시리즈, 데이터프레임으로 잡고 .str.원하는문법 ( extract, 여러가지가 있다. findall)
- 정규표현식을 사용할 예정이니, (정규표현식을 넣을 예정)
"""

for df in combine:
    df['Name_re'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=True) #위의 말했던 str 문법과 정규표현식을 이용해서 전처리


# 실제 성별과 Name_re 만든 피처가 실제 잘 매핑이 되었는지 간단한 검증?
# 판다스에서 제공하는 crosstab() 피벗테이블의 개념 엑셀의 개념으로

pd.crosstab(train_df['Name_re'],train_df['Sex']) #다른 피처넣어보면서 어떤 관계인지는 확인해 보시면 좋다.

"""
- 그냥 그대로 다 가지고 간다. 나눈 것들을 그대로 옆에 컬럼을 붙이는 경우
- 피처의 값들의 범위가 너무 넓어지고 y값에 따른 구분할 수 있는 피처를 만드는 게 아니라 의미 없이 피처를 하나더 만드는 것은 -> 차원이 하나 더 추가 되는 것
- 차원이 하나 더 추가되면 데이터 상에서 멀어지고 복잡해진다.
- 성별과 생존의 유무가 확실히 보였다.
- 이 부분을 응용해서 진행
- 전체 다 사용하는 게 아니라, 구간화해서 묶어주는 작업 그룹화 작업의 개념, bining 그룹핑으로 데이터를 묶는다.
- 유사한 것들끼리, 이 기준은 우리는 Y값과 기존에 탐색했던 성별 컬럼을 보고 의사결정 진행
"""

# replace 파이썬 기초문법
for df in combine:
    df['Name_re'] = df['Name_re'].replace(['Lady','Countess','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    df['Name_re'] = df['Name_re'].replace('Mlle','Miss')
    df['Name_re'] = df['Name_re'].replace('Ms','Miss')
    df['Name_re'] = df['Name_re'].replace('Mme','Miss')


# 간단하게 groupby로 살펴보자
train_df[['Name_re','Survived']].groupby(['Name_re'],as_index=False).mean()

print(train_df)

"""
- Name_re 에서 나온 값들을 -> 수치로 변경하는 것
- 인코딩 작업
- 레이블인코딩, 원핫인코딩, 그외에 인코딩도 많지만 대표적으로 이 두 가지

- 레이블인코딩 : 1,2,3,4,5,6 해당 값에 순서를 매칭해서 변경해 준다. Mr 1 Mrs 2, Rare 3, Miss 4 등등
- 원핫인코딩 : 0,1로만 피처를 만들어 준다.
- Mr, Mrs, Rare, Miss 컬럼으로 만들어 준다.
- 이 컬럼에 대응하는 값이 0,1 인지를 인덱스기준으로 매핑한다.
"""

# 레이블인코딩으로 변환
name_re = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4,'Rare':5}

for df in combine:
    df['Name_re'] = df['Name_re'].map(name_re) #map함수와 판다스의 시리즈가 만나서 해당 값으로 변환해 준다.
    df['Name_re'] = df['Name_re'].fillna(0)

print(train_df)

"""
Sex 컬럼 전처리
원핫인코딩의 개념으로

- Sex 를 전처리하기 좋은 Male, Female -> 0,1 생존확률이 높은 사람은 여성이었고, 여성을 1, 남성을 0
"""

#전처리 후 남은 사용한 컬럼은 제거
train_df = train_df.drop(['Name','PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

for df in combine:
    df['Sex'] = df['Sex'].map({'female':1, 'male':0}).astype(int)

#남성과 여성도 원핫인코딩의 개념으로 인코딩작업 진행

train_df.isna().sum()

"""
age 컬럼 전처리
- Missing Value 대체하는 방법
- MICE
- 보간법
- 단순한 기초통계치로 대체하는 법
- KNN, ML 기법을 통해서 대체하는 법
"""

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', aspect =1.5)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)

"""
- 단순하게 Age 평균으로 대체하면 -> Median
- Pclass, Sex 따라서 2차원으로 나눈 후 -> 그 값들의 Median, Mean 대체를 한다.
"""

guess_ages = np.zeros((2,3)) 
# np.zeros 0값 넣는다. 일단 na값을 채워넣을 주머니 하나 만들었다.

## 결측치를 채울 코드는?
## pclass, 성별 2 두가지의 차원으로 결측치를 바라볼 것
## 두 가지의 피처로 값을 추출하면서 이 값의 median의 값을 -> guess_ages 넣어서 대체하면 된다.
## age는 22.1살 22.7살 없다. 반올림을 하는 작업을 진행해야 한다.

for df in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df = df[(df['Sex']==i) & (df['Pclass']==j+1)]['Age'].dropna()

            age_guess = guess_df.median()
            guess_ages[i,j] = (age_guess/0.5 + 0.5) * 0.5


    for i in range(0,2):
        for j in range(0,3):
            df.loc[(df.Age.isnull()) & (df.Sex ==i)&(df.Pclass==j+1), 'Age'] = guess_ages[i,j]
    df['Age'] = df['Age'].astype(int)

train_df.Age.isna().sum()

print(train_df.head())

"""
age 데이터 범위가 넓다.
왜 우리가 구간을 나누고 계속해서 전처리를 하는가?
- 판다스 문법을 통해 쉽게 구간을 나눌 수 있다.
- cut, qcut (나누는 개념)
"""

train_df.Age.value_counts()

max(train_df['Age'])
min(train_df['Age'])

# 나이에 대한 범주화 값을 나눠보자!
train_df['Agerange']=pd.cut(train_df['Age'],5)

# 단순하게 그냥 구간을 나누고 마음대로 할 수 있는 게 아니라 위와 동일하게 y값의 관계를 살펴보자!
train_df[['Agerange','Survived']].groupby(['Agerange'],as_index=False).mean().sort_values(by='Agerange',ascending=False)

"""
age -> 인코딩하는 작업으로 데이터 전처리
"""

for df in combine:
    df.loc[df['Age']<=16,'Age'] = 0
    df.loc[(df['Age']>16) & (df['Age']<=32), 'Age']=1
    df.loc[(df['Age']>32) & (df['Age']<=48), 'Age']=2
    df.loc[(df['Age']>48) & (df['Age']<=64), 'Age']=3
    df.loc[df['Age']>64,'Age'] =4

#위에 pd.cut 구간으로해서 데이터를 인코딩 진행 완료
train_df.Age.value_counts()

train_df = train_df.drop(['Agerange'],axis=1)

print(train_df)

"""
sibsp, parch
- 두 개 컬럼이 가족과 관련한 동승자와 관려한 컬럼
- 이 두 가지를 합쳐서 새로운 컬럼을 만들고 그 컬럼을 통해서 피처 가공한다.
- 이 두 가지를 합쳐서 -> 새로운 하나의 파생변수 FamilySize 만들기!
"""

print(combine)

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] +1


print(combine)


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] +1 # 가족과 관련된 컬럼 -> 패밀리 수치로 바꿈

combine[0][['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending=False)

"""
- 가족들과 함께 탑승, 혼자 탑승 이 두 가지의 경우로 나눠서 컬럼을 추가할 수 있다.
"""

print(combine)

for dataset in combine:
    dataset['IsAlone'] =0
    dataset.loc[dataset['FamilySize']==1 , 'IsAlone'] =1

combine[0][['IsAlone','Survived']].groupby(['IsAlone'], as_index=False).mean()

"""
- 두 가지 컬럼을 isAlone 만들어서 전처리 작업 완료
"""

print(combine[1])

# train_df에 우리가 만든 파생변수를 같이 넣자!
train_df=combine[0].drop(['Agerange','SibSp','Parch'],axis=1)
test_df= combine[1].drop(['SibSp','Parch'],axis=1)

print(combine[1])

# 학습할 데이터 셋으로 정리 다시하기!
combine = [train_df,test_df]

print(combine)

"""
Age 컬럼이랑 Pclass 가지고 새로운 변수로 만들기
- 나이와 타이타닉의 선박의 등급이랑 두 개의 갑을 곱해서 하나의 파생변수 만들었다.
"""

for dataset in combine:
    dataset['Age*Pclass'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Pclass', 'Age','Pclass']]

train_df.isna().sum()

freq_value=train_df['Embarked'].dropna().mode()[0] #최빈값으로 대체를 한다.

# Embarked 데이터 전처리

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_value) # 2개의 na값이 빈도가 가장 높았던 'S'로 대체가 된다.

train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)

# 결측값이 모두 다 처리가 되었다.
train_df.isna().sum()

# Embarked -> 인코딩 작업 진행

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1 , 'Q':2}).astype(int) #문자열 ->수치로

print(train_df)

# 테스트 데이터에도 na값이 fare이 있음
test_df.isna().sum()

## test fare 요금의 결측값 대체하는 법
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True) #na값 1개를 median으로 대체

"""
### Fare 요금도 동일하게 qcut, cut이용해서 전처리 작업하자!
"""

train_df['FareBand'] = pd.qcut(train_df['Fare'],4)

train_df[['FareBand','Survived']].groupby(['FareBand'],as_index=False).mean().sort_values(by='Survived',ascending=False)

train_df.FareBand.value_counts()

for dataset in combine:
    dataset.loc[dataset['Fare'] <=7.91, 'Fare']= 0
    dataset.loc[(dataset['Fare']>7.91) & (dataset['Fare']<=14.454),'Fare']=1
    dataset.loc[(dataset['Fare']>14.454) & (dataset['Fare']<=31),'Fare']=2
    dataset.loc[dataset['Fare'] >31, 'Fare']= 3
    dataset['Fare'] =dataset['Fare'].astype('int')

train_df = train_df.drop(['FareBand'],axis=1)
combine = [train_df, test_df]

print(train_df)


"""
모델링을 통해 성능을 비교해 보자!

성능의 비교는 전처리를 하기 전과 후로 비교하여 성능의 변화 값 확인!
- 둘 다 타이타닉데이터를 가지고 진행
- 원본: base , 전처리 하지 않은 원본데이터를 간단하게 가공예정
- 전처리: 우리가 3주동안 배웠던 전처리 데이터셋

- 데이터 분석 과정
- 데이터 수집 - > 데이터 전처리 -> 데이터 모델링 -> 데이터 모델링 평가 -> 적용, 전개

- 3주동안 배웠던 과정이 -> 데이터 전처리를 배웠다.
- 무엇을 모델링 하는지?
    - 타이타닉 데이터에 대한 생존을 예측
- 독립변수 / 종속변수, 피처들에 대한 이해도가 있어야 한다.
    - 타이타닉데이터의 생존을 예측 Survived y값, label 정답!
    - 지도학습 ( 정답이 있는 데이터 )
    - 생존율은 1,0으로 이뤄진 이진 데이터
    - 연속형, 범주형 (이진, 다중) 데이터 값에 따라 모델의 사용하는 방법이 달라진다.
    - 이진분류 -> 분류 모델링을 사용하는 것
        - 이진 분류모델링 - 로지스틱회귀, Decision Tree, RandomFroest, XGBoost, CatBoost, LightGBM 기타 등등..
        - 연속형 회귀 모델링 - 선형회귀, 다중회귀, 다항회귀 등등, 분류 모델들도 다 회귀로도 모델링이 가능하다.
        
- 모델을 사용해서 현실의 문제를 해결하기 위해서 -> 타이타닉 생존율 예측하기 위해서 필요한 기본 과정
    - 데이터를 학습시켜야 한다. 데이터를 준비해야 한다.
    - train, test, validation 데이터셋을 나눠야 한다.
    - model도 불러오거나, 직접 알고리즘을 코드로 구현하거나 등등
    - 성능을 평가해야 한다. ( 얼마나 생존율을 잘 예측하는가? )
        - 성능지표들을 알아야 한다. ( Accuracy, Recall, Precision, F1-score, AIC, BIC AUC등등.. )
        - 성능지표도 예측하는 정답에 데이터 타입에 따라 다르다.
            - 이진 분류, 다중 분류 classif -> Accuracy, Recall, Precision, F1-score, AIC, BIC AUC
            - 연속형의 경우는 -> MSE, RMSE 기타 잔차에 대한 평가 지표로 진행한다.
    - 성능을 통해 평가 후에 -> 일반화 가능한지를 검증
        - Test 데이터를 가지고 최종 결과물을 평가한 값으로 확인하는 것
    
- 모델링시 중요하게 봐야 하는 부분!
- 과소적합 -> 이 부분을 꼭 잘 체크해야 합니다.
- 과대적합 : 모델이 train 데이터에 너무 집착해서 학습하게 된 경우 -> train data 에서는 좋은 성능이 나오지만 -> test에서는 성능이 나쁘게 나오는
- 과소적합 : 모델이 너무 단순하거나, 데이터셋이 부족해서 -> train 성능이 잘 나오지 않고, test 높거나 이러한 상황들이 나오는 경우

- 과대적합, 과소적합에 대해서 -> 어떤 값을 보고 우리가 확인해야 하는지?
    - train, test 결과물을 통해 확인한다.
    - train, test로 비교했을 때 -> 둘의 성능 어떤 식으로 나오는 게 가장 이상적일까?
        - 둘 다 높게 나온다?
        - 둘 다 적당하게 나온다?
        - train 일단 높게 나와야 하고 -> 하지만 test 수능과 같이 한 번도 우리가 보지 못한 문제를 푸는 것
        - test의 점수는 조금 train보다는 낮다.
        - train도 높고 test도 높은 게 좋지만 -> train보다는 test가 성능이 떨어지는 것은 당연한 것
        - 둘 사이의 간극을 최대한 줄이는 그런 모델의 성능평가가 가장 좋다고 바라볼 수 있다.

- train/test로 데이터셋을 나눠야 한다.
- train/test로 데이터를 나눈 이유!
- 데이터셋을 가지고 모델링을 하고 끝이 아니라 -> 이 데이터셋을 가지고 학습한 모델을 가지고 일반화를 통해서 다른 데이터셋이 들어왔을 때 동일하게 예측을 한다. **일반화** 일반화성능이 높아야 한다.
"""

## 실제 코드로 모델링을 진행하자!
## train, test를 나누기 위해서
## x_train, x_test, y_train, y_test ->
## x, y -> x는 독립변수 y는 종속변수, x는 생존을 제외한 다른 피처, y는 생존, 정답
## train, test 학습, 실제 일반화 하기 위한 검증 train 학습하는 모델 test train으로 테스트하고 실제 성능이 얼마나 좋은지 평가하는 데이터셋

X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']

## test

X_test = test_df.drop('PassengerId',axis=1).copy()

## 데이터셋을 좀 더 확인해 보자! 몇 개?

X_train.shape, Y_train.shape, X_test.shape

"""
- train, test를 나눌 때 비율이나 이런 부분을 말씀 주셨는데
- sklearn. train_test_split 이 패키지로 간단하게 할 수 있다.
- 7:3 8:2 train 7: test :3 이런 식으로 나눌 수 있다. 정답은 아니지만 대부분 진행하는 데이터셋 나누는 기준
"""

# 모델을 불러와서 진행하자!
from sklearn.linear_model import LogisticRegression

# import 모델을 변수에 넣는다.
logreg= LogisticRegression()
## 학습을 하기 위해 fit method를 이용해서 데이터를 학습시켜야 한다.
logreg.fit(X_train, Y_train) # X_train, Y_train 모델에 학습 끝

# 예측값 -> test 데이터를 생존율 예측하는 정답을 받아야 한다.
# predict 사용해서 정답을 가지고 오면 된다.
# 미래값을 예측해야 하는 것이니 X_test test를 넣어야 한다.
Y_pred_test=logreg.predict(X_test)

##train, test를 비교해야 하는데
## test값으로는 성능을 비교할 수 없다 (지금 당장은 -> 정답이 없으니)
## train으로만 비교를 해보자!
## train -> 정답을 예측했을 때 정말 성능이 높게 나오는가? 수치적으로 확인한다
## accuracy 확인해 보자! -> 정확도 실제 생존을 생존으로 예측했는지? + 실제 생존이 아닌 것을 생존이 아닌 것으로 예측했는가? / 전체 모수

Y_pred_train=logreg.predict(X_train)

## train data의 정확도를 평가하기 위해서
round(logreg.score(X_train, Y_train) *100 ,3)

logreg.score(X_train, Y_train)

# train 데이터를 통해 예측한 y값을 -> 어디에 비교할까?
len(Y_pred_train)

print(Y_pred_train)

list(Y_train)

len(Y_pred_test)


or_train_df = pd.read_csv('/Users/mj/문서 - 강민지의 MacBook Pro/machine-learning-repo/machine_learning_base/data/train(1).csv')
or_test_df = pd.read_csv('/Users/mj/문서 - 강민지의 MacBook Pro/machine-learning-repo/machine_learning_base/data/test(1).csv')

print(or_train_df)
print(or_train_df)

