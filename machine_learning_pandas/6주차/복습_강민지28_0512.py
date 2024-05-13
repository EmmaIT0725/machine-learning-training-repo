import numpy as np
import pandas as pd
import seaborn as sns

# 이상치 탐지 방법
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# 이상치 탐지방법
from statsmodels.tsa.seasonal import seasonal_decompose

# 데이터 생성
data = np.random.randn(100) * 20 + 50
data = np.append(data, [150, -50, 200])  # 명확한 이상치 추가

# 데이터를 DataFrame으로 변환
df = pd.DataFrame(data, columns=['Data'])

# IQR 을 가지고 이상치를 탐지하기!

# Q1, Q3 계산

Q1 = df['Data'].quantile(0.25)
Q3 = df['Data'].quantile(0.75)

# IOR 계산

IQR = Q3- Q1

# 이상치의 경계 계산
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

# 이상치 식별
outliers = df[(df['Data'] < lower_bound) | (df['Data']> upper_bound)]

outliers

print(Q1, Q3, IQR,lower_bound,upper_bound, outliers)

# IsolationForest를 이용한 이상치 탐지

# 중심점 3개인 데이터 생성
data, _ = make_blobs(n_samples = 300, centers=3, random_state=111)

# 이상치에 대한 것들 추가

outliers = np.random.uniform(low=-10, high=10, size=(20,2))

# 원본데이터 이상치 데이터 합치기

data =np.vstack([data, outliers])

# Isolation Forest로 모델 생성 및 학습
model = IsolationForest(n_estimators = 100,contamination=0.5, random_state=111)
# comtamination
# 이상치가 차지하는 비율을 나타내는 것 0과 1사이의 실수로 표현 
# 기본은 'auto' -> 데이터셋에 따라 알아서 이상치 탐지
# 값이 높아질수록 더 많은 데이터셋이 이상치로 간주 - False Positive 증가하는 경우
# 값이 낮은 경우는 이상치 판단되는 데이터셋 포인트 수가 줄어든다. - False Negative 증가하는 경우

# 모델이 직접 예측
predictions = model.fit_predict(data)

# 데이터 시각화 진행

plt.figure(figsize=(10,6))
plt.scatter(data[:,0], data[:,1], c =predictions, cmap='Paired', marker='o', s=30, edgecolor='k')

print(predictions)

model

# 시계열 데이터를 통해 롤링함수 이용해서 간단하게 이상치 탐지하는 방법

# 데이터 생성
np.random.seed(111)
data= np.random.randn(100).cumsum() + 100
# 이상치를 좀 추가하기
data[20] +=50
data[70] -=30
data[50] +=15

# 이동평균 표준편차 계산으로 이상치 탐지

window_size = 5
moving_average = pd.Series(data).rolling(window = window_size).mean()
moving_std = pd.Series(data).rolling(window = window_size).std()

# 이상치 탐지 기준 설정 (평균 +- 1.5 *표준편차)

upper_bound = moving_average + 1.5 * moving_std
lower_bound = moving_average - 1.5 * moving_std

# 데이터프레임을 생성해서 확인하기
df = pd.DataFrame({'Data':data, 'Moving Average':moving_average, 'upper_bound':upper_bound,'lower_bound':lower_bound})

# 이상치 탐지
outliers =df[(df['Data'] > df['upper_bound'])| (df['Data'] < df['lower_bound'])]


# 시각화로 확인해 보기
plt.figure(figsize=(12,9))
plt.plot(df['Data'], label='Data', color='gray', marker='o', markersize=4, linestyle='-')
plt.plot(df['Moving Average'], color='red',label='Moving average')
plt.fill_between(df.index, df['upper_bound'], df['lower_bound'], color='gray', alpha=0.2, label='Confidence Interval')
plt.scatter(outliers.index, outliers['Data'], color='magenta', label='Outliers', s=200, edgecolor='black',zorder=5)

outliers

print(data, moving_average, moving_std)

# decompose 패키지 이용

# 시계열 분해
result = seasonal_decompose(data, model ='additive',period=4)
result.plot()

# 잔차를 통한 이상치 탐지
residuals = result.resid
outliers = residuals[abs(residuals) > 2 * residuals.std()]

