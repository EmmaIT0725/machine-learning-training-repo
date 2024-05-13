import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('/Users/mj/Documents/Machine_Learning/data/train(1).csv')
test_df = pd.read_csv('/Users/mj/Documents/Machine_Learning/data/test(1).csv')

print(train_df.head())
print()
print(test_df.head())

# 타이타닉 데이터 합쳐서 분석하기
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
print(combine)

# Name column을 전처리
'''
str 문법
문자열 데이터에 대한 전처리 작업 진행시 사용
'''
for df in combine:
    df['Name_re'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=True)
    # 'Name_re' 라는 열을 새로 생성
    # 위의 말했던 str 문법과 정규표현식을 이용해서 전처리
    '''
    df.Name.str.extract() 부분은 df 데이터프레임의 'Name' 열의 각 값에
    대해 정규 표현식(정규식)을 사용해 특정 패턴을 추출한다.
    정규 표현식 '  ([A-Za-z]+)\.' : 이 정규 표현식은 다음을 의미한다:
    (공백): 공백 문자 (스페이스)를 의미한다.
    [A-Za-z]+: 알파벳 대소문자(A-Z 또는 a-z)로 구성된 하나 이상의 문자(여러 문자)를 의미한다.
    \.: 마침표(.)를 의미한다. 마침표는 정규식에서 특별한 의미를 가지므로, 
    앞에 백슬래시(\)를 붙여 이스케이프 해야한다.
    추출 결과를 새로운 열 'Name_re'에 저장: 
    expand=True는 추출 결과를 데이터프레임 형태로 반환하도록 한다. 
    '''

print(df['Name_re'])
# 실제 성별과 Name_re 만든 feature가 실제 잘 매핑이 되었는지 간단한 검증은 어떻게 할 수 있을까?
# 판다스에서 제공하는 crosstab()은 피벗테이블의 개념 
print(pd.crosstab(train_df['Name_re'], train_df['Sex']))
print()
# 다른 피처 넣어보면서 어떤 관계인지 확인해 보면 좋다.
print(pd.crosstab(train_df['Name_re'], train_df['Pclass']))
print()
print(pd.crosstab(train_df['Name_re'], train_df['Embarked']))
print()

# replace 파이썬 기초문법

# inplace=True 작동안함
# for df in combine:
#     df['Name_re'].replace(['Lady','Countess','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare', inplace=True)
#     df['Name_re'].replace('Mlle','Miss', inplace=True)
#     df['Name_re'].replace('Ms','Miss', inplace=True)
#     df['Name_re'].replace('Mme','Miss', inplace=True)

for df in combine:
    df['Name_re'] = df['Name_re'].replace(['Lady','Countess','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    df['Name_re'] = df['Name_re'].replace('Mlle','Miss')
    df['Name_re'] = df['Name_re'].replace('Ms','Miss')
    df['Name_re'] = df['Name_re'].replace('Mme','Miss')

# 간단하게 groupby로 살펴보기
print(train_df[['Name_re', 'Survived']].groupby(['Name_re'], as_index=False).mean())

# 인코딩 작업하기
'''
Name_re 에서 나온 값들을 -> 수치로 변경하는 것

# 인코딩 작업

레이블인코딩, 원핫인코딩, 그외에 인코딩도 많지만 대표적으로 이 두 가지

- 레이블인코딩 : 1,2,3,4,5,6 해당 값에 순서를 매칭해서 변경해 준다. Mr 1 Mrs 2, Rare 3, Miss 4 등등

- 원핫인코딩 : 0,1로만 피처를 만들어 준다.

    Mr, Mrs, Rare, Miss 컬럼으로 만들어 준다.
    이 컬럼에 대응하는 값이 0,1 인지를 인덱스기준으로 매핑한다.
'''

# 레이블 인코딩으로 변환
name_re = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}

for df in combine:
    df['Name_re'] = df['Name_re'].map(name_re)  # map함수와 판다스의 시리즈가 만나서 해당 값으로 변환해 준다.
    df['Name_re'] = df['Name_re'].fillna(0)


# Sex 컬럼 전처리

# 원-핫 인코딩의 개념으로
# Sex를 전처리하기 딱 좋은 Male, Female -> 0, 1 : 생존 확률이 높았던 여성을 1, 남성을 0

# 전처리 후 남은 사용한 컬럼은 제거하기
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

for df in combine:
    df['Sex'] = df['Sex'].map({'female':1, 'male':0}).astype(int)

print(train_df.isna().sum())

# age 컬럼 전처리
'''
Missing Value 대체하는 방법
MICE
보간법
단순한 기초통계치로 대체하는 법
KNN, ML 기법을 통해서 대체하는 법
'''

grid = sns.FacetGrid(train_df, row='Embarked', col='Sex', aspect=1.5)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
plt.show()

guess_ages = np.zeros((3,4)) # np.zeros 0값 넣는다. 일단 na값을 채워넣을 주머니 만듦.

## 결측치를 채울 코드는?
## Pclass, 성별 2 두가지의 차원으로 결측치를 바라볼 것
## 두 가지의 피처로 값을 추출하면서 이 값의 median의 값을 -> guess_ages 넣어서 대체하면 된다.
## age는 22.1살 22.7살 없다. 반올림을 하는 작업을 진행해야 한다.

for df in combine:
    for i in range(0,3):
        for j in range(0,4):
            guess_df = df[(df['Sex']==i) & (df['Pclass']==j+1)]['Age'].dropna()

            age_guess = guess_df.median()
            guess_ages[i,j] = (age_guess/0.5 + 0.5) * 0.5


    for i in range(0,3):
        for j in range(0,4):
            df.loc[(df.Age.isnull()) & (df.Sex ==i)&(df.Pclass==j+1), 'Age'] = guess_ages[i,j]
    df['Age'] = df['Age'].astype(int)

print(train_df.Age.isna().sum())