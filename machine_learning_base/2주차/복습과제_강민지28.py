# -*- coding: utf-8 -*-
"""복습과제_강민지28.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OVjFI3zNewkoDDA44F76quOozKQaFH_q
"""

# seaborn 패키지를 불러서 가지고 오기!
import seaborn as sns

data = sns.load_dataset('titanic')

data

# Pandas 기초문법

import pandas as pd

# 기본적인 판다스의 다양한 문법
# 판다스는 dataframe, series 두 가지 메인이 나뉜다.
# 2차원은 df, 1차원 series

type(data)  # 데이터 프레임
type(data['embarked'])  # 하나의 열만을 추출한 경우: Seires
type(data.pclass)
type(data[['age', 'fare']]) # 두 개의 열 이상부터는 다시 데이터 프레임

# 2차원은 dataframe, 1차원은 Series

# shape
# 행과 열을 반환
data.shape

# columns
data.columns    # 컬럼만 추출하고 싶은 경우

# 컬럼만 추출하고 싶은 경우
data_col = list(data.columns)
data_col

# info 데이터타입의 다양한 정보를 얻을 수 있다.
data.info()

# 데이터 타입만 가져올 경우
data.dtypes

# 수치형 데이터 기초 통계
data.describe()

# 수치형 데이터 외 데이터까지 모두 확인 가능
data.describe(include='all')

# 데이터를 살펴볼 수 있는 다양한 문법
# head, tail
data.head()     # default 값은 앞의 5개 추출

data.tail()     # default 값은 뒤의 5개 추출

data.head(20)       # 안의 숫자를 통해 추출하고자 하는 행의 수 조정 가능

## 행과 열로 이루어진 데이터 프레임
# 열 기준으로 추출해보기
# df[컬럼]
# df.컬럼
# 문자열이나 공백이 있으면 df.age 이런식으로 하면 에러남 >> 이럴 때는 df[age]로 사용

data['survived']    # 데이터 컬럼 하나만 추출

# 여러 개의 값이 들어가면 리스트에서 [[]] 대괄호 두 개 넣기
data[['survived', 'age']]

# 변수처럼 내가 원하는 데이터를 넣을 수 있다.
# 즉, 객체처럼 이용 가능
data_sp = data[['survived', 'age', 'fare']]
data_sp

data

data[['sex', 'age', 'embarked']]

data_sae = data[['sex', 'age', 'embarked']]
data_sae.head()

## 행 데이터 추출하기!
# 인덱스 기준 문법 이해하기

# loc : 행 인덱스 기준으로 추출(이름으로 추출이 가능),
#       변수의 개념으로 추출도 가능하다! ( -1 끝 값 출력 불가능)
#       약속한 인덱스 기준으로만 추출 가능

# iloc : 행 인덱스 기준으로 추출(추출한 인덱스 기준)
#        모든 번호로 추출이 가능 (-1 끝 값 출력 가능)
#        약속한 인덱스 뿐만 아니라 새롭게 설정한 인덱스 기준으로도 추출가능
# data.iloc[1:40, ['age', 'survived']] 이렇게 접근 불가능
# >> 컬럼명이 아닌, 번호로 추출
# data.iloc[1:40, [3, 8, -1]]

# 리스트 생각하면 index [] 동일
# loc[index]
# iloc[index]

data

data.loc[0] # 첫 행의 모든 값을 추출

data.iloc[0]

# data.loc[-1] >> 오류 발생
# 즉, '-1' 값으로 끝 행의 값 추출 불가

# loc에서 맨 끝 값을 만드는 경우
# 891 값을 불러오기 (가장 끝 값)

# shape >> (행, 열) 의 튜플값이 나온다.
# data.shape[0]: 행을 불러오는 것

num_row = data.shape[0] # 891
num_row
lst_idx = num_row - 1   # 새로운 변수 선언

data.loc[lst_idx]       # loc은 맨 끝 값을 이렇게 변수로 넣어서도 가능

data.iloc[-1]

quarter_num = 600
data.loc[quarter_num]

data.iloc[quarter_num]

## 하나의 데이터만 추출하는 것이 아니라 여러 행을 추출하고 싶다!

# [[inedex1, index2, index3,..., indexn]]

data.loc[[1, 10, 20, 30]]

data.iloc[[1, 10, 20, 30]]

idx = 1
data.loc[[idx, idx+10, idx+20, idx+30]]

idx = 1
data.iloc[[idx, idx+10, idx+20, idx+30]]

'''
    range 슬라이싱을 가지고 -> 동일하게 적용가능.
    loc, iloc 슬라이싱을 통해서 데이터 범위 range 지정가능.
    [:] 전체
    [시작점:끝점:증가폭] range함수와 동일하게 추출할 수 있다!
'''

data.loc[::2]

data.iloc[::3]

data_sp_idx = data.loc[3:50:4]

data_sp_idx

# data_sp_idx.loc[2]
# loc은 지정한 index에 대해서만 추출 가능

data_sp_idx.iloc[2]
# iloc은 새롭게 지정한 index에 대해서도 추출 가능

# 슬라이싱과 컬럼을 통해 접근하기
data.loc[1:50, ['age']]

data.loc[:30, ['age', 'survived']]
# 1차원 [] 두 개 컬럼이상이면 그대로 추가하면 된다.

# data.iloc[1:40, ['age', 'survived']]

data.iloc[1:40, [3, 8, -1]]
# # iloc은 넘버로만 지정해서 접근해야 한다

data.iloc[1:50, [3, 6, 10]]
data.iloc[1:50, [3, 6, 10]]['age']
data.iloc[1:50, [3, 6, 10]]['age'].loc[10]

# data2 = data.iloc[51:100, ['age', 'survived', 'fare']]
data2 = data.loc[51:100, ['age', 'survived', 'fare']]

data3 = data2['fare']
data3.loc[51]

# 인덱스 5번 째 값중에서 sex와 embarked를 추출한다!
data.loc[5][['sex', 'embarked']]

data[['sex', 'embarked']].iloc[5]

# Q.여기서 iloc 대신 loc을 쓰면 에러가 뜨는데 왜그런건가요? tt.loc[1:50,[3,4]]
# data.loc[1:50,[3,4]] >> loc의 슬라이싱 + 배열에서, 컬럼명을 숫자가 아닌 이름으로 사용해야함.
data.iloc[1:50,[3,4]]

## range 이용하여 변수화 하여 데이터 추출
rn = range(0,100,3)
rn

rn = list(range(0,100,3))
rn

# 인덱스처럼 넣을 수 있다.
data.loc[rn]

data.iloc[rn]   # 슬라이싱 + 배열 아닌 곳에서는 문자열 넣어도 무방

rn_col = list(range(0,10,2))
rn_col

data.iloc[:,rn_col] # rn_col은 열(column) 추출로 사용
# [ , ] 컴마 기준 행/열
# 컬럼을 range로 list 넣은 후 원하는 컬럼만 데이터 추출 가능
# iloc 은 새로운 index 가능

## 기본적인 통계치를 구하기!

# fare 요금이니 통계치로 보기에 적정
# 평균, median, sum 값?
data['fare'].mean()

data.fare.sum()

data.fare.count()

data.fare.median()

data.fare.sum()/data.fare.count()

## 한 뎁스씩 더 들어가서 값을 구하자!
# pclass별로 평균 fare 얼마입니까?
# 엑셀에서 피벗 개념으로 값 계산할 수 있다.
# groupby 그룹으루 묶어서 계산하는 것
# groupby('그룹을 묶을 컬럼')['통계치를 볼 컬럼'].원하는 통계치()
data.groupby('pclass')['fare'].mean()

# sex별로 평균 age얼마입니까?
data.groupby('sex')['age'].mean()

data.groupby('sex')['fare'].mean()

data.groupby('embark_town')['fare'].mean()

# 재밌는 것!
data.groupby('sex')['survived'].sum() # 전체 생존자 중에서 여성이 233명 생존, 남성이 109명 생존

data.groupby('sex')['fare'].mean()

data.groupby('sex')[['fare','age']].mean()

data

data.groupby('embarked')['who'].count()

data.groupby('adult_male')['fare'].mean()

