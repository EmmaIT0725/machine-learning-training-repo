## 다양한 데이터 전처리 문법
'''
- apply
- multi_index
- pivot_table
- transpose
- melt
'''

### apply
"""
- 데이터프레임의 전체 데이터 값을 함수나 전처리하기 위해서는 apply 적용시켜야 한다.
- 전체 데이터프레임값을 변환하거나 함수를 사용할 때 적용한다.
"""

import seaborn as sns
import pandas as pd

df = sns.load_dataset('titanic')

# 평균값을 전처리 하고 싶다.
# 요금에 대해서 전처리

df['fare'].mean()

# apply 기존 메서드 외에 튜닝이 가능 --> 내가 원하는 함수들을 만들어서 적용시킬 수 있다.

df.groupby('class')['fare'].apply(lambda x : x.mean())

"""
### 나만의 함수를 만들어서 직접 전처리
- 나이를 정규화해서 class도 정규화 해서 score를 하나 만들어서 전체 스코어를 출력하는 함수를 만들자!
"""

def cal_risk_score(row):
    age_score = row['age']/80
    class_score = (4-row['pclass'])/3
    return age_score + class_score

print(df)

#위의 만든 함수를 전체 데이터프레임 적용
df['risk_score']=df.apply(cal_risk_score, axis=1)

"""
### multi_index
- 다중인덱스
- 인덱스 1개가 아니라 2개 이상인 경우
"""

print(df)

## class , sex  두 개를 묶어서 보고싶다.
multi_idx=df.set_index(['class','sex','embarked']).groupby(level=[0,1,2]).mean()

"""
- 멀티인덱스 접근하는 방법
- loc
"""

print(multi_idx)

# 멀티 인덱스로 다중으로 잡힌 상태에서
# 인덱스에 있는 값으로 loc으로 접근
# 인덱싱 된 값 기준으로 -> loc안에다 해당 값을 넣어서 원하는 값을 추출할 수 있다.
multi_idx.loc[('First','female')]

print(multi_idx)

# 인덱스를 슬라이싱하여서 값으로 추출할 수 있다.
multi_idx.loc['Second':'Third']

"""
- xs('내가 원하는 인덱스 값을 적고', 해당 인덱스 값의 인덱스를 적어주면 된다.)
"""

multi_idx.xs('female',level='sex')

"""
### pivot_table
- values
- index
- columns
- aggfunc

- 위의 인자값으로 넣으면 된다.
- 내가 원하는 데이터의 요약치를 빠르게 확인할 때 사용할 수 있는 문법
"""

df.pivot_table(values= 'survived', index='pclass',columns='sex', aggfunc='sum')

df.pivot_table(values= ['fare','age'], index='pclass',columns='survived', aggfunc='mean')

"""
### transpose
- 축반전, 인덱스와 컬럼이 반전이 된다.
"""

print(df.transpose())

"""
### melt
- 깔끔하게 데이터를 전처리 해야 하는 경우 사용한다.
- 컬럼자체에서 값이 있는 경우나, 컬럼의 역할을 하지 못하는 경우, 속성의 attribute의 가정을 위반하는 데이터인 경우
- 깔끔하게 전처리 할 수 있는 melt를 사용하면 쉽게 데이터를 정리할 수 있다.
---

- id_vars : 그대로 유지할 값을 기준을 정하는 것
- value_vars : 되돌리기 할 열을 나타내는 것 id_vars로 지정하지 않은 모든 열이 피벗 되어 되돌리 대상으로 지정
- var_name : value_vars 되돌린 값이 새로운 컬럼안에 들어가면 해당 컬럼 값을 이름 정할 때 지정하는 명, 기본값이 variable
- value_name : var_name의 열의 값을 나타내는 새로운 열의 이름 문자
"""

print(df)

pew=pd.read_csv('pew.csv')

print(pew)

# id_vars
pd.melt(pew,id_vars='religion',var_name='income',value_name='count')

bill=pd.read_csv('billboard.csv')

print(bill)

pd.melt(bill, id_vars=['year','artist','track','time','date.entered'])

pd.melt(bill, id_vars=['year','artist','track','time','date.entered'])

ebola=pd.read_csv('country_timeseries.csv')

ebola_melt=pd.melt(ebola,id_vars=['Date','Day'])

ebola_melt.variable.str.split('_')

print(ebola_melt)

## 나눠진 2개의 값을 다시 컬럼을 넣어보자!

ebola_melt['case']=ebola_melt.variable.str.split('_').str.get(0)
ebola_melt['country']=ebola_melt.variable.str.split('_').str.get(1)

# 컬럼에 의미가 있던 값들을 -> melt를 이용해서 피봇 값으로 변환 -> 그 값에서 2개의 의미를 가지고 있는 값들을 다시 2개를 나눠서 각각 하나의 컬럼에 담는 과정

print(ebola)
print(ebola_melt)
print(df)