---
title: "ML02_sklean_Analytical Model(회귀분석)"
date: 2021-08-14
comments: true
categories: ML PyCode
---

###  Analytical Model

#### 회귀 분석(regression analysis)

- sklearn 라이브러리에서 제공하는 보스턴(boston) 데이터를 활용

- seaborn 모듈을 사용하여 데이터 시각화

- sklearn에서 제공하는 회귀 분석 모듈을 사용해 보스턴 주택가격을 맞추는 머신러닝 진행

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_boston

# boston 데이타셋 로드
boston = load_boston()

# boston 데이타셋 DataFrame 변환 
bostonDF = pd.DataFrame(boston.data , columns = boston.feature_names)

# boston dataset의 target array는 주택 가격임. 이를 PRICE 컬럼으로 DataFrame에 추가함. 
bostonDF['PRICE'] = boston.target
print('Boston 데이타셋 크기 :',bostonDF.shape)
bostonDF.head()
```

Boston 데이타셋 크기 : (506, 14)

|      |    CRIM |   ZN | INDUS | CHAS |   NOX |    RM |  AGE |    DIS |  RAD |   TAX | PTRATIO |      B | LSTAT | PRICE |
| ---: | ------: | ---: | ----: | ---: | ----: | ----: | ---: | -----: | ---: | ----: | ------: | -----: | ----: | ----: |
|    0 | 0.00632 | 18.0 |  2.31 |  0.0 | 0.538 | 6.575 | 65.2 | 4.0900 |  1.0 | 296.0 |    15.3 | 396.90 |  4.98 |  24.0 |
|    1 | 0.02731 |  0.0 |  7.07 |  0.0 | 0.469 | 6.421 | 78.9 | 4.9671 |  2.0 | 242.0 |    17.8 | 396.90 |  9.14 |  21.6 |
|    2 | 0.02729 |  0.0 |  7.07 |  0.0 | 0.469 | 7.185 | 61.1 | 4.9671 |  2.0 | 242.0 |    17.8 | 392.83 |  4.03 |  34.7 |
|    3 | 0.03237 |  0.0 |  2.18 |  0.0 | 0.458 | 6.998 | 45.8 | 6.0622 |  3.0 | 222.0 |    18.7 | 394.63 |  2.94 |  33.4 |
|    4 | 0.06905 |  0.0 |  2.18 |  0.0 | 0.458 | 7.147 | 54.2 | 6.0622 |  3.0 | 222.0 |    18.7 | 396.90 |  5.33 |  36.2 |



```python
bostonDF = bostonDF[['RM', 'ZN', 'INDUS', 'NOX', 'AGE', 'PTRATIO', 'LSTAT', 'RAD', 'PRICE']]
bostonDF.head()
```

|      |    RM |   ZN | INDUS |   NOX |  AGE | PTRATIO | LSTAT |  RAD | PRICE |
| ---: | ----: | ---: | ----: | ----: | ---: | ------: | ----: | ---: | ----: |
|    0 | 6.575 | 18.0 |  2.31 | 0.538 | 65.2 |    15.3 |  4.98 |  1.0 |  24.0 |
|    1 | 6.421 |  0.0 |  7.07 | 0.469 | 78.9 |    17.8 |  9.14 |  2.0 |  21.6 |
|    2 | 7.185 |  0.0 |  7.07 | 0.469 | 61.1 |    17.8 |  4.03 |  2.0 |  34.7 |
|    3 | 6.998 |  0.0 |  2.18 | 0.458 | 45.8 |    18.7 |  2.94 |  3.0 |  33.4 |
|    4 | 7.147 |  0.0 |  2.18 | 0.458 | 54.2 |    18.7 |  5.33 |  3.0 |  36.2 |

예측에 사용할 독립변수 열과 종속변수(PRICE)열만 추출



```python
sns.pairplot(bostonDF)
plt.show()
```

![회귀분석0](https://user-images.githubusercontent.com/86189842/129445848-77a07b67-def5-4c17-96d7-bff73dce869c.png)독립변수와 종속변수간의 상관관계 시각화



```python
sns.jointplot(x='LSTAT', y='PRICE',kind='reg',data=bostonDF)
```

![회귀분석1](https://user-images.githubusercontent.com/86189842/129445851-8f880d6c-06c5-424c-874f-bb21219795e5.png)명확한 선형의 상관관계를 나타내는 LSTAT변수



```python
X=bostonDF[['LSTAT']] #독립변수X
y=bostonDF['PRICE'] #종속변수y

#train data 와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,                  #독립변수
                                                    y,                  #종속변수
                                                    test_size=0.3,      #검증30%
                                                    random_state=10)    #랜덤 추출
print('train data 개수: ', len(X_train))
print('test data 개수: ', len(X_test))
```

train data 개수:  354   test data 개수:  152



**1. 단순회귀분석(simple regression) **

```python
# sklearn 라이브러리에서 선형회귀분석 모듈 가져오기
from sklearn.linear_model import LinearRegression

#단순회귀분석 모형 객체 생성
lr = LinearRegression()

#train data를 가지고 모형학습
lr.fit(X_train, y_train)
# 학습을 마친 모형에 test data를 적용하여 결정계수(R-제곱) 계산
r_square = lr.score(X_test, y_test)
r_square
```

0.5135

```python
# 모형에 전체 X데이터를 입력하여 예측한 값 y_hat을 실제 값 y와 비교
y_hat = lr.predict(X_test)
df_y = pd.DataFrame({'y_hat':y_hat, 'y':y_test})
df_y['차이']=df_y['y_hat']-df_y['y']
df_y
```

|      |     y_hat |    y |      차이 |
| ---: | --------: | ---: | --------: |
|  305 | 25.732453 | 28.4 | -2.667547 |
|  193 | 29.364446 | 31.1 | -1.735554 |
|   65 | 29.699707 | 23.5 |  6.199707 |
|  349 | 28.563545 | 26.6 |  1.963545 |
|  151 | 21.681385 | 19.6 |  2.081385 |
|  ... |       ... |  ... |       ... |
|   56 | 28.675299 | 24.7 |  3.975299 |
|   37 | 25.881458 | 21.0 |  4.881458 |
|   66 | 24.512476 | 19.4 |  5.112476 |
|  427 | 20.526597 | 10.9 |  9.626597 |
|   12 | 19.418374 | 21.7 | -2.281626 |

```python
plt.figure(figsize=(10,5))
ax1 = sns.kdeplot(y_test, label='y_test')
ax2 = sns.kdeplot(y_hat, label='y_hat', ax=ax1)
plt.legend()
plt.show()
```

- ![회귀분석2](https://user-images.githubusercontent.com/86189842/129445852-515ab10c-e27a-4872-af7d-5cd2697b57d2.png)단순회귀분석 결과 추세는 비슷하게 예측했으나 정확도가 상당히 떨어짐을 알 수 있음



**2. 다항회귀분석(polynomial regression)**

```python
from sklearn.preprocessing import PolynomialFeatures
# 비선형 회귀분석을 위해 제곱값을 가지는 2차항 생성
poly=PolynomialFeatures(degree=2)
X_train_poly=poly.fit_transform(X_train)
# 2차항으로 변환시킨 X_train, 기존 y_train데이터를 통해 학습
pr = LinearRegression()
pr.fit(X_train_poly, y_train)

# X_test 데이터를 2차항으로 변형
X_test_poly=poly.fit_transform(X_test)

# 학습을 마친 모형에 test data를 적용하여 결정계수(R-제곱)계산
r_square = pr.score(X_test_poly, y_test)
r_square
```

0.6198

- 결정계수가 유의미하게 증가했음을 알 수 있음(약 21%)

```python
y_hat = pr.predict(X_test_poly)
df_y_test=pd.DataFrame({'y_hat':y_hat,'y':y_test})
df_y_test['차이']=df_y_test['y_hat']-df_y_test['y']
df_y_test
```

| y_hat |         y | 차이 |           |
| ----: | --------: | ---: | --------- |
|   305 | 25.289089 | 28.4 | -3.110911 |
|   193 | 31.879294 | 31.1 | 0.779294  |
|    65 | 32.553535 | 23.5 | 9.053535  |
|   349 | 30.313701 | 26.6 | 3.713701  |
|   151 | 19.480945 | 19.6 | -0.119055 |
|   ... |       ... |  ... | ...       |
|    56 | 30.528339 | 24.7 | 5.828339  |
|    37 | 25.533737 | 21.0 | 4.533737  |
|    66 | 23.368804 | 19.4 | 3.968804  |
|   427 | 18.123214 | 10.9 | 7.223214  |
|    12 | 16.944518 | 21.7 | -4.755482 |

152 rows × 3 columns

```python
plt.figure(figsize=(10,5))
ax1 = sns.kdeplot(y_test, label='y_test')
ax2 = sns.kdeplot(y_hat, label='y_hat', ax=ax1)
plt.legend()
plt.show()
```

- ![회귀분석3](https://user-images.githubusercontent.com/86189842/129445850-11135a44-0661-4f54-8d9a-f60739be6745.png)다항회귀분석 결과 정확도는 단순회귀분석에 비해 올라갔으나 추세선은 오히려 부정확해진 것을 알 수 있음

  

**3. 다중회귀분석(multiple regression)**

```python
# 8개의 독립변수를 변수X 에 저장하여 다중회귀분석
X=bostonDF[['RM','ZN','INDUS','NOX','AGE','PTRATIO','LSTAT','RAD']] #독립 변수 X1,X2,X3...
y=bostonDF['PRICE'] #종속 변수 y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,                  #독립변수
                                                    y,                  #종속변수
                                                    test_size=0.3,      #검증30%
                                                    random_state=10)    #랜덤 추출

print('train data 모양: ', X_train.shape)
print('test data 모양: ', X_test.shape)
```

train data 모양:  (354, 8) test data 모양:  (152, 8)

```python
# sklearn 라이브러리에서 선형회귀분석 모듈 가져오기
from sklearn.linear_model import LinearRegression

#단순회귀분석 모형 객체 생성
lr = LinearRegression()

#train data를 가지고 모형학습
lr.fit(X_train, y_train)

# 학습을 마친 모형에 test data를 적용하여 결정계수(R-제곱) 계산
r_square = lr.score(X_test, y_test)
r_square
```

0.6778

- 결정계수가 다항회귀분석보다 증가했음(약 10%)

```python
y_hat=lr.predict(X_test)
df_y_test=pd.DataFrame({'y_hat':y_hat,'y':y_test})
df_y_test['차이']=df_y_test['y_hat']-df_y_test['y']
df_y_test
```

|      |     y_hat |    y |      차이 |
| ---: | --------: | ---: | --------: |
|  305 | 26.459896 | 28.4 | -1.940104 |
|  193 | 30.962438 | 31.1 | -0.137562 |
|   65 | 29.488793 | 23.5 |  5.988793 |
|  349 | 27.718619 | 26.6 |  1.118619 |
|  151 | 20.675571 | 19.6 |  1.075571 |
|  ... |       ... |  ... |       ... |
|   56 | 28.711512 | 24.7 |  4.011512 |
|   37 | 21.583340 | 21.0 |  0.583340 |
|   66 | 24.624773 | 19.4 |  5.224773 |
|  427 | 18.812364 | 10.9 |  7.912364 |
|   12 | 20.842642 | 21.7 | -0.857358 |

152 rows × 3 columns

```python
plt.figure(figsize=(10,5))
ax1 = sns.kdeplot(y_test, label='y_test')
ax2 = sns.kdeplot(y_hat, label='y_hat', ax=ax1)
plt.legend()
plt.show()
```

- ![회귀분석4](https://user-images.githubusercontent.com/86189842/129445855-9dc8e57c-5a90-4ac2-9c94-f811d229de3c.png)시각화 결과 정확도와 추세선 모두 가장 정확한 결과가 나옴

