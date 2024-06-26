# -*- coding: utf-8 -*-
"""필수과제1_강민지_28.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OspAvXTyTWURuwgsQG962lTQR0k6cIwE
"""

'''
필수과제 1

    성별에 따라, 클래스에 따라서 값을 매핑하는 반복문 만들자
    남, 여 pclass ,1,2,3 으로 나뉘어져 있다.
    남 1 = m1
    남 2 = m2
    남 3 = m3

    여 1 = f1
    여 2 = f2
    여 3 = f3

    파생변수를 만들어 주세요!

    파생변수?
    기존에 있는 변수들을 가지고 조합해서 만드는 경우
    기존 변수에 사칙연산을 통해서 만들 수도 있고
    아예 새로운 식으로 만들 수 있다. (기존변수 또는 파생변수를 이용해서도)

'''

import seaborn as sns
import pandas as pd
import numpy as np

dt = sns.load_dataset('titanic')

dt

# 값을 loc[i] 이용해서 한 행씩 받을 것
# 파생변수 미리 설정해서 값 초기화
dt['sex_p'] = 0

for i in range(len(dt)):
    if(dt['sex'].loc[i] == 'male' and dt['pclass'].loc[i] == 1):
        dt['sex_p'].loc[i] = 'm1'
    elif(dt['sex'].loc[i] == 'male' and dt['pclass'].loc[i] == 2):
        dt['sex_p'].loc[i] = 'm2'
    elif(dt['sex'].loc[i] == 'male' and dt['pclass'].loc[i] == 3):
        dt['sex_p'].loc[i] = 'm3'
    elif(dt['sex'].loc[i] == 'female' and dt['pclass'].loc[i] == 1):
        dt['sex_p'].loc[i] = 'f1'
    elif(dt['sex'].loc[i] == 'female' and dt['pclass'].loc[i] == 2):
        dt['sex_p'].loc[i] = 'f2'
    elif(dt['sex'].loc[i] == 'female' and dt['pclass'].loc[i] == 3):
        dt['sex_p'].loc[i] = 'f3'

dt

# 마지막 것은 elif 대신 else 사용해도 됨
# 값을 loc[i] 이용해서 한 행씩 받을 것
# 파생변수 미리 설정해서 값 초기화
dt['sex_p'] = 0

for i in range(len(dt)):
    if(dt['sex'].loc[i] == 'male' and dt['pclass'].loc[i] == 1):
        dt['sex_p'].loc[i] = 'm1'
    elif(dt['sex'].loc[i] == 'male' and dt['pclass'].loc[i] == 2):
        dt['sex_p'].loc[i] = 'm2'
    elif(dt['sex'].loc[i] == 'male' and dt['pclass'].loc[i] == 3):
        dt['sex_p'].loc[i] = 'm3'
    elif(dt['sex'].loc[i] == 'female' and dt['pclass'].loc[i] == 1):
        dt['sex_p'].loc[i] = 'f1'
    elif(dt['sex'].loc[i] == 'female' and dt['pclass'].loc[i] == 2):
        dt['sex_p'].loc[i] = 'f2'
    else:
        dt['sex_p'].loc[i] = 'f3'

dt

