# -*- coding: utf-8 -*-
"""필수과제1_강민지28.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1P69CmN78DwOeQjJCF0werb_4o9SgqQyd

# 필수과제1
- NA값이 있는 데이터를 공유할 예정
- 해당 NA값들을 위의 나온 방법을 가지고 실제 결측치를 대체해 주세요.
- 결측치를 대체한 후에 groupby를 통한 다른 피처들과의 관계를 비교하면서 값들이 어떤 식으로 대체되었는지를 정리해 주세요.

### 필수과제 1-1
- 4개의 피처가 있는데, 1개 피처만 NA값이 있어서 -> 이걸 위의 방법으로 대체한 후에, 다른 컬럼들과의 관계를 통해서 어떤 식으로 값들이 변화하는지

### 필수과제 1-2
- 기존의 원본데이터와, NA값을 대체한 데이터들의 차이를 비교해서 정리해 주세요.


- 필수과제 데이터를 드리면서, 원본데이터를 드릴 예정, 여러분들이 직접 NA값의 구간을 만들어서 원본데이터에서, NA값을 만들고 그 후에 이제 비교분석을 하셔야 합니다.

- 추가적으로 시각화를 통해서 도메인에 따른 결측치 처리를 분석하면서 공유하셔도 됩니다.

###
- 주석으로 설명을 못하면 과제를 한 게 아니다.
- 본인의 언어로 정리해서 과제를 잘 해주세요.

[ 필수과제1 데이터셋 설명 ]
###### 꼭 읽어주세요!
###### 사이킷런 제공하는 캘리포니아 집값 데이터 셋 불러오는 코드
from sklearn.datasets import fetch_california_housing
###### 캘리포니아 주택 가격 데이터셋 로드
housing = fetch_california_housing()
###### 데이터프레임 생성
df_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
df_housing['Target'] = housing.target  # 목표 변수 추가

###### 결측값 컬럼
MedInc 소득 컬럼에 대해 결측치를 만들고 과제 진행해 주세요.

###### 피처 설명
MedInc: 해당 지역의 중간 소득. 이 값은 수천 달러 단위로 표현됩니다. 중간 소득이 높을수록 해당 지역의 주민들은 더 많은 돈을 벌고 있다는 것을 의미합니다.

HouseAge: 해당 지역의 중앙값 주택 연령. 이는 지역에 있는 주택들 중간의 연령을 나타냅니다.

AveRooms: 지역의 평균 방 갯수. 이 값은 해당 지역의 모든 주택의 방 수를 평균낸 것입니다.

AveBedrms: 지역의 평균 침실 갯수. 이 값은 해당 지역의 모든 주택의 침실 수를 평균낸 것입니다.

Population: 해당 지역의 인구. 이는 해당 지역에 살고 있는 사람들의 수를 나타냅니다.

AveOccup: 평균 주택 점유율. 이는 한 주택에 평균적으로 거주하는 사람의 수를 나타냅니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from fancyimpute import IterativeImputer
# 사이킷런 제공하는 캘리포니아 집값 데이터 셋 불러오는 코드
from sklearn.datasets import fetch_california_housing

# 캘리포니아 주택 가격 데이터셋 로드
housing = fetch_california_housing()
# 데이터프레임 생성
df_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
df_housing['Target'] = housing.target  # 목표 변수 추가

df_housing

# 결측값 컬럼
# MedInc 소득 컬럼에 대해 결측치를 만들고 과제 진행해 주세요.

df_housing['MedInc']

df_housing.value_counts('MedInc')

df_housing.loc[5:20, 'MedInc'] = np.NaN
df_housing.loc[150:200, 'MedInc'] = np.NaN
df_housing.loc[450:600, 'MedInc'] = np.NaN
df_housing.loc[790:1050, 'MedInc'] = np.NaN
df_housing.loc[2500:5279, 'MedInc'] = np.NaN
df_housing.loc[10798:15798, 'MedInc'] = np.NaN
df_housing.loc[20000:20600, 'MedInc'] = np.NaN

# 결측치 대체
# 1. 1차 선형보간법, interpolate
df_linear = df_housing.interpolate(method='linear') # 1차 선형보간법

# 2. 2차 선형보간법, interpolate
df_quaderatic = df_housing.interpolate(method='quadratic')  # 2차 선형보간법

# 3. 평균대치법
imputer_mean = SimpleImputer(strategy='mean')
# sklearn 제공하는 simpleimputer로 평균대치
df_mean = imputer_mean.fit_transform(df_housing)
df_mean = pd.DataFrame(df_mean, columns= df_housing.columns)
# ** columns= 은 전체 열을 반영해야 한다.
# columns 담길 때 복습과제에서 data를 df라는 DataFrame으로 만들어줬기 때문에
# 2차원이 되어 column명이 List로 들어감
'''
만약, 코드를 아래와 같이 작성시,
df_mean = pd.DataFrame(df_mean, columns= ['MedInc'])
그러나 코드에서는 'MedInc' 열만을 선택하여 대체하고 있는데,
다른 열들에 대해서는 대체를 수행하지 않고 있다.
이는 원하는 대체 작업을 다른 열에도 적용하지 않는 한계로 인해 발생할 수 있다.
'''

# 4. 0값으로 대체
# fillna() 이 안에 원하는 값을 넣을 수 있다.
df_zero = df_housing.fillna(0)

# 5. KNN 방법
imputer_knn = KNNImputer(n_neighbors=3)
df_knn = imputer_knn.fit_transform(df_housing)
df_knn = pd.DataFrame(df_knn, columns=df_housing.columns)
# df_knn = pd.DataFrame(df_knn, columns=['MedInc'])

# 6. MICE 다중대치법
mice_imputer = IterativeImputer()   # 베이지안회귀로 결측치 보간
df_mice = mice_imputer.fit_transform(df_housing)
df_mice = pd.DataFrame(df_mice, columns=df_housing.columns)
# df_mice = pd.DataFrame(df_mice, columns=['MedInc'])

print(df_housing.loc[450:600, 'MedInc'])
print(df_linear.loc[450:600, 'MedInc'])
print(df_quaderatic.loc[450:600, 'MedInc'])

print(df_mean.loc[450:600, 'MedInc'])
print(df_zero.loc[450:600, 'MedInc'])
print(df_knn.loc[450:600, 'MedInc'])
print(df_mice.loc[450:600, 'MedInc'])

# 결측치를 대체한 후에 groupby를 통한 다른 피처들과의 관계 비교하기

# 결측치가 들어간 행
# df_housing.loc[5:20, 'MedInc']
# df_housing.loc[150:200, 'MedInc']
# df_housing.loc[450:600, 'MedInc']
# df_housing.loc[790:1050, 'MedInc']
# df_housing.loc[2500:5279, 'MedInc']
# df_housing.loc[10798:15798, 'MedInc']
# df_housing.loc[20000:20600, 'MedInc']

print(df_housing.groupby(['AveRooms', 'AveRooms'])['MedInc'].mean())
print()
print(df_housing.groupby(['HouseAge', 'AveOccup'])['MedInc'].mean())
# NaN 있는 부분으로 인해 4번째 줄의 평균이 1.0938임을 볼 수 있는데
# 아래의 df_linear은 결측치가 채워져서 4번째 줄의 평균이 다르게 나옴을 확인할 수 있다.

print(df_linear.groupby(['AveRooms', 'AveRooms'])['MedInc'].mean())
print()
print(df_linear.groupby(['HouseAge', 'AveOccup'])['MedInc'].mean())
# 위의 df_housing의 3번째 값이 NaN이었는데, 현재 df_linear의 결측치 값은 보간되었다.
# 따라서 결측치 값이 3.477810으로 보간 되었고 이후의 평균이 df_housing과 다르게 나옴을 알 수 있다.

print(df_linear.groupby(['AveRooms', 'AveRooms'])['MedInc'].sum())

print(df_quaderatic.groupby(['AveRooms', 'AveRooms'])['MedInc'].mean())
print()
print(df_quaderatic.groupby(['HouseAge', 'AveOccup'])['MedInc'].mean())
# df_quaderatic을 groupby로 그룹화한 후 MedInc의 평균을 살펴보면 '-'값이 들어있음을 알 수 있다.

print(df_quaderatic.groupby(['AveRooms', 'AveRooms'])['MedInc'].sum())

print(df_mean.groupby(['AveRooms', 'AveRooms'])['MedInc'].mean())
print()
print(df_mean.groupby(['HouseAge', 'AveOccup'])['MedInc'].mean())

print(df_mean.groupby(['AveRooms', 'AveRooms'])['MedInc'].sum())

print(df_zero.groupby(['AveRooms', 'AveRooms'])['MedInc'].mean())
print()
print(df_zero.groupby(['HouseAge', 'AveOccup'])['MedInc'].mean())

print(df_zero.groupby(['AveRooms', 'AveRooms'])['MedInc'].sum())

print(df_knn.groupby(['AveRooms', 'AveRooms'])['MedInc'].mean())
print()
print(df_knn.groupby(['HouseAge', 'AveOccup'])['MedInc'].mean())

print(df_knn.groupby(['AveRooms', 'AveRooms'])['MedInc'].sum())

print(df_mice.groupby(['AveRooms', 'AveRooms'])['MedInc'].mean())
print()
print(df_mice.groupby(['HouseAge', 'AveOccup'])['MedInc'].mean())

print(df_mice.groupby(['AveRooms', 'AveRooms'])['MedInc'].sum())

