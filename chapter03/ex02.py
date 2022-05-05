## 3장 구현을 위한 도구 심화 문제

import numpy as np
import pandas as pd

## 3.2. 다음은 P 자동차 회사의 차종별 마력, 총중량,
## 그리고 연비를 나타낸 표이다.

## 3.2.1. 이 표를 데이터 프레임으로 만드는 코드를 작성하라.
name_series = pd.Series(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
horse_power_series = pd.Series([130, 250, 190, 300, 210, 220, 170])
weight_series = pd.Series([1900, 2600, 2200, 2900, 2400, 2300, 2200])
efficiency_series = pd.Series([16.3, 10.2, 11.1, 7.1, 12.1, 13.2, 14.2])

df = pd.DataFrame({'name': name_series,
                   'horse_power': horse_power_series,
                   'weight': weight_series,
                   'efficiency': efficiency_series,
                   })

## 3.2.2. 이 데이터 프레임의 name 열을 인덱스로 지정하는 코드를 작성하라.
df.set_index('name', inplace=True)

## 3.3.3. 마력과 연비를 고려하여 차를 선택하려고 한다. 마력과 연비를
##        곱한 값이 가장 큰 차종을 찾아 출력하는 코드를 작성하라.
df['performance'] = df['horse_power'] * df['efficiency']
best_performance_car = df['performance'].idxmax()
print(f'마력과 연비를 곱한 값 중 최대값을 가진 차종: {best_performance_car}')
