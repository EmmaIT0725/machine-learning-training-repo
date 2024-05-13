# -*- coding: utf-8 -*-
"""복습과제_강민지28.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Oeib-UT5580d9coms9JuJ8oGbWA7vOLM
"""

'''
데이터 분석에 필요한 기초문법(2)

    query()
    sort_values()
    행,열 추출 loc,iloc [], [[]] (2주차에 진행)
    파생변수 만들기
    assign(), lambda
    groupby()와 함께 사용하는 agg()
    merge(), concat()
    str()
    시계열
    [] 불린인덱싱
    전체 다 사용하여 응용진행

'''

# query()

# 데이터를 추출하기 위한 조건
# 내가 원하는 데이터를 추출하여서 해당값을 가지고 오는 것

import seaborn as sns
data = sns.load_dataset('titanic')

data

## 문법
# data.query('요구조건')

# 자료형 에러
# data.query(sex == 'female')   # 전체를 ' '로 묶어주어야 한다.
data.query('sex == "female"')   # 문자형은 '' or "" 따옴표 달아주어야 한다.
data.query('''sex == "female"''')

data.query('age >= 50')

data.query('''fare >=50 and fare <= 100''')

'''
    3가지의 조건을 모두 충족시키는 데이터는?
    생존하고 여성이고 25살 미만
'''

data.query('survived == 1 and sex == "female" and age < 25')

# and, or 조건에 따라 값이 달라진다.
data.query('''sex == "female" or survived == 1 and age < 25''')

# data.query('''sex == "female" or survived == 1 and age < 25''')
'''
sex == "female"  이거나 survived == 1 and age < 25
즉, and 조건이 or보다 먼저이므로, survived == 1 and age < 25 먼저 평가됨
이후 sex == "female"이 평가된다.
즉, survived == 1 and age < 25 이거나 sex == "female"인 데이터가 모두 도출
'''

# query에 변수도 넣을 수 있는가?

data_age_mean = data['age'].mean()
data_age_mean

data.query('age >= data_age_mean')
# 오류가 난다. >> 변수로는 들어가지 않음을 확인할 수 있음

data.query('sex == "female" and age <= 25')[['survived', 'sex', 'age']]
# 두 개 이상의 column 담을 때는 [[]] 사용하기

data.query('sex == "female" and age <= 25')[['survived', 'sex', 'age']].age

data.query('sex == "female" and age <= 25')[['survived', 'sex', 'age']].age.mean()

# 필수과제1 >> 필수과제1_3주차_강민지28.ipynb로 제출

## sort(정렬)

# 값을 정렬하는 개념
# 오름차순, 내림차순 개념
# data.sort_values('age')

# 오름차순이 디폴트 값으로 정렬
data.sort_values('age')

# ascending 차순을 변경하는 파라미터
# 내림차순으로 변경, ascending = False
data.sort_values('age', ascending = False)

# 두 개 이상의 기준으로 내림, 오름 차순으로 정렬하면?
# 두 개 이상이면 [ ] 도입하기
data.sort_values(['age', 'fare'])
# 1 순위 : age, 2 순위 : fare

# 차순을 지정할 수 있다.
data.sort_values(['age', 'fare'], ascending = [False, True])

# 어쨌든지 앞의 것이 우선! 앞의 것에 대한 차순이 끝이나면 그 다음으로 넘어감

## 추가 변수를 만들어보자!
'''
# 새로운 파생변수를 만들기!
# 기존 변수를 이용해서 새로운 변수를 만드는 것
# 기존 변수를 합치거나 둘 이상을, 기존 변수에 새로운 값을 사칙연산을 통해 계산하거나 등등

# 요금에 단위를 맞추기 위해 100을 다 곱하자!
# 새로운 변수에 값을 넣어야 한다.
# 변수를 만들어야 한다.
# 변수를 만들기 위해서는 데이터프레임에서 변수를 만들어야 한다.
'''

# 파생변수 1차로 만들기
data['fare_100'] = 0 # 초깃값 세팅 python 문법

data['fare_100'] = data.fare * 100 # 변수를 만들어서 새로운 값을 넣을 수 있다.

data

'''
python 문법 중 우리가 배웠던 if문랑 for문 이용해서 전처리를 진행해 보자!

    Q1. 남성과 여성의 fare 평균을 새로운 변수로 만들고 싶다!
    내가 원하는 컬럼은 fare에 대한 남성과 여성에 따른 평균값이 다른 fare 컬럼의 값이 매핑된다.

'''

# query문은 ()
# loc은 []
male_fare_mean = data.query('sex == "male"')['fare'].mean()
female_fare_mean = data.query('sex == "female"')['fare'].mean()

data.loc[1]['sex']
data.loc[1].sex

data['male_fare_mean'] = 0
data['female_fare_mean'] = 0

# 각각의 행을 판단해서 도출할 경우 loc[i] 붙여주기
# len(data) = 891
for i in range(len(data)):
    # if문을 이용해서 남성과 여성의 값을 비교하고, 파생변수를 만들어서 그곳에 남성과 여성의 평균 fare값을 넣는 변수를 만든다.
    if data['sex'].loc[i] == 'female':
        data['female_fare_mean'].loc[i] = female_fare_mean
        # female_fare_mean = data.query('sex == "female"')['fare'].mean()
        # loc[[i]]으로 행 지정해줘야한다.
    elif data['sex'].loc[i] == 'male':
        data['male_fare_mean'].loc[i] = male_fare_mean

data

data[['sex', 'fare', 'male_fare_mean', 'female_fare_mean']]

# 필수과제2 >> 필수과제2_3주차_강민지28.ipynb로 제출

# 파생변수를 만들 수 있는 문법
## query는 () 내부에 변수 넣으니까 오류발생

# - assign(내가 원하는 요구조건을 넣으면 된다.) 파생변수를 할당하는 함수
# - lambda

data

# assign(원하는 요구 조건, 넘파이를 이용해서 where을 이용할 예정)
import numpy as np

data.assign(age_level = np.where(data['age'] >=60, 'old', 'young'))
# 반영되지는 않음

# age_level는 파생변수
# dataframe에 할당됨
# assign(파생변수명 = np.where(요구조건, 파생변수값, 파생변수 나머지 값))

data

# 마찬가지로 assign도 () 사용
data.assign(fare_level = np.where(data['fare'] >= 100, 'expensive', 'cheap'))

# np.where은 해당조건의 인덱스를 반환한다.
np.where(data['sex'] == 'female')

# np.where()은 해당 조건의 인덱스를 반환한다.
np.where(data['sex'] == 'female')

data.assign(sex_symbol = np.where(data['sex'] == 'female', 'F', 'M'))
# 정리하면 sex_symbol라는 파생변수를 column으로 받아서 \
# np.where을 만족하는 인덱스에 해당 파생변수 값('F' / 'M')을 \
# 할당하는 것이 assign 함수의 역할

'''
    실제값을 바꿔주려면 두 가지 방법이 있다.
    변수처럼 사용해서 새로운 파생변수로 만들거나
    inplace= True 원본데이터에 반영을 해달라고 지정을 해야 한다.
'''

data.assign(city_symbol = np.where(data['embark_town'] == 'Southampton', 'S', 'NotS'))

# np.where()은 () 안에 있는 조건을 만족하는 값의 인덱스를 할당하는 함수
# 다만, data.assign(파생변수 = np.where(조건, \
#                   '조건을 만족하는 경우의 파생변수 값',
#                           '조건을 만족하지 않는 경우의 파생변수 값'))
# 으로 표현할 수 있는데, 2가지 종류 이외의 더 다양한 종류의 파생변수 값을 넣고 싶다면...?
# np.where을 계속 추가한다.
# 최종 np.where 찍고, 최종 나머지 값을 넣어주면 된다.
data.assign(city_symbol = np.where(data['embark_town'] == 'Southampton', 'S',
                                   np.where(data['embark_town'] == 'Cherbourg', 'C',
                                            np.where(data['embark_town'] == 'Queenstown', 'Q', 'Unknown'))))

# 새로운 객체를 선언해서 완전히 저장
data1 = data.assign(pclass_level = np.where(data['pclass'] == 1, 'Top',\
    np.where(data['pclass'] == 2, 'Middle', 'Bottom')))

data1

data

# lambda 이용해서도 가능
'''
lambda x는 파이썬에서 익명 함수(anonymous function)를 정의하는 방법 중 하나
한 줄로 간결하게 함수를 정의할 수 있다.
'''
double = lambda x: x*2

double(5)

fare_100 = lambda x: x['fare']*100
# 함수처럼 작동

fare_100(data)

# assign lambda 함수도 쉽게 넣어서 파생변수를 만들 수 있다.

data.assign(fare_100 = lambda x: x['fare']*100)
# 해당 조건을 만족하는 파생변수를 column으로 받아서 dataframe 내에\
# 새로운 column을 만든다.

data.assign(fare_mean = lambda x: x['fare'].mean(),\
    age_double = lambda x: x['age']*2)

# 해당 조건을 만족하는 값이 fare_orin이라는 column에 삽입되는 것
# 두 개이상 파생변수 만들 때는 아래 로직으로 , 찍고 원하는 파생변수를 동일한 로직으로 이어가면 된다.

## groupby()
# 집단들에 대한 그룹을 묶어서 요약통계치 등을 보는 문법

# groupby('묶고자 하는 column 명')['값을 보려고 하는 column'].통계치()
data.groupby('pclass')['survived'].sum()
# sum은 0/1로 되어있어서 합을 통해 생존자를 파악할 수 있다.

data.describe()

# 여러 개의 요약통계치를 확인하기
'''
.agg는 데이터프레임(DataFrame)에서 그룹별로 집계(aggregate) 함수를 적용하는 메서드.
주로 데이터프레임을 그룹화한 후에 (groupby) \
    각 그룹에 대해 특정한 집계 함수를 적용할 때 사용.
'''
data.groupby('sex').agg(fare_mean = ('fare', 'mean'),\
                        fare_median = ('fare', 'median'),\
                            fare_std = ('fare', 'std'))

data.groupby('pclass').agg(survived = ('survived', 'sum'),
                            fare_mean = ('fare', 'mean'),
                            age_mean = ('age', 'mean'))

# groupby와 agg 함수 사용하는 방법 익혀두기
data.groupby('embark_town').agg(survived = ('survived', 'sum'),
                                age_mean = ('age', 'mean'),
                                fare_mean = ('fare', 'mean'))

