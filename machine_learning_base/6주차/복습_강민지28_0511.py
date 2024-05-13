import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('/Users/mj/Documents/Machine_Learning_Base/data/train(1).csv')
test_df = pd.read_csv('/Users/mj/Documents/Machine_Learning_Base/data/test(1).csv')

train_df

test_df

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

combine

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

train_df

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

train_df

"""
Sex 컬럼 전처리

## 원핫인코딩의 개념으로

- Sex 를 전처리하기 딱 좋은 Male, Female -> 0,1 생존확률이 높은 사람은 여성이었고, 여성을 1, 남성을 0
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

guess_ages = np.zeros((2,3)) # np.zeros 0값 넣는다. 일단 na값을 채워넣을 주머니 하나 만들었다.

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

train_df.head()

"""
## age 데이터 범위가 넓다.
## 왜 우리가 구간을 나누고 계속해서 전처리를 하는가?
- 판다스 문법을 통해 쉽게 구간을 나눌 수 있다.
- cut, qcut (나누는 개념)
"""

train_df.Age.value_counts()

max(train_df['Age'])

min(train_df['Age'])

#나이에 대한 범주화 값을 나눠보자!
train_df['Agerange']=pd.cut(train_df['Age'],5)

#단순하게 그냥 구간을 나누고 마음대로 할 수 있는 게 아니라 위와 동일하게 y값의 관계를 살펴보자!
train_df[['Agerange','Survived']].groupby(['Agerange'],as_index=False).mean().sort_values(by='Agerange',ascending=False)

"""### age ->인코딩하는 작업으로 데이터 전처리"""

for df in combine:
    df.loc[df['Age']<=16,'Age'] = 0
    df.loc[(df['Age']>16) & (df['Age']<=32), 'Age']=1
    df.loc[(df['Age']>32) & (df['Age']<=48), 'Age']=2
    df.loc[(df['Age']>48) & (df['Age']<=64), 'Age']=3
    df.loc[df['Age']>64,'Age'] =4

#위에 pd.cut 구간으로해서 데이터를 인코딩 진행 완료

train_df.Age.value_counts()

train_df = train_df.drop(['Agerange'],axis=1)

train_df

"""
## sibsp, parch
- 두 개 컬럼이 가족과 관련한 동승자와 관려한 컬럼
- 이 두 가지를 합쳐서 새로운 컬럼을 만들고 그 컬럼을 통해서 피처 가공한다.

- 이 두 가지를 합쳐서 -> 새로운 하나의 파생변수 FamilySize 만듦!
"""

combine

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] +1

combine

combine

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] +1 # 가족과 관련된 컬럼 -> 패밀리 수치로 바꿈

combine[0][['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending=False)

"""- 가족들과 함께 탑승, 혼자 탑승 이 두 가지의 경우로 나눠서 컬럼을 추가할 수 있다."""

combine

for dataset in combine:
    dataset['IsAlone'] =0
    dataset.loc[dataset['FamilySize']==1 , 'IsAlone'] =1

combine[0][['IsAlone','Survived']].groupby(['IsAlone'], as_index=False).mean()

"""- 두 가지 컬럼을 isAlone 만들어서 전처리 작업 완료"""

combine[1]

#train_df에 우리가 만든 파생변수를 같이 넣자!
train_df=combine[0].drop(['Agerange','SibSp','Parch'],axis=1)
test_df= combine[1].drop(['SibSp','Parch'],axis=1)

combine[1]

#학습할 데이터 셋으로 정리 다시하기!
combine = [train_df,test_df]

combine

"""## Age 컬럼이랑 Pclass 가지고 새로운 변수로 만들기
- 나이와 타이타닉의 선박의 등급이랑 두 개의 갑을 곱해서 하나의 파생변수 만들었다.
"""

for dataset in combine:
    dataset['Age*Pclass'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Pclass', 'Age','Pclass']]

train_df.isna().sum()

freq_value=train_df['Embarked'].dropna().mode()[0] #최빈값으로 대체를 한다.

#Embarked 데이터 전처리

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_value) # 2개의 na값이 빈도가 가장 높았던 'S'로 대체가 된다.

train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)

#결측값이 모두 다 처리가 되었다.
train_df.isna().sum()

# Embarked ->인코딩 작업 진행

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1 , 'Q':2}).astype(int) #문자열 ->수치로


train_df

#테스트 데이터에도 na값이 fare이 있음
test_df.isna().sum()

## test fare 요금의 결측값 대체하는 법
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True) #na값 1개를 median으로 대체

"""### Fare 요금도 동일하게 qcut, cut이용해서 전처리 작업하자!"""

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

train_df

