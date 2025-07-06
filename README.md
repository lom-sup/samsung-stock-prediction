# LSTM을 이용한 삼성전자 주가 예측 모델 생성

<br/>

> 이 프로젝트는 삼성전자의 주가를 다변량 시계열 데이터를 바탕으로 예측하는 **인공 신경망 모델**을 구현하는 것을 목표로 한다.  <br/>
> Pytorch 강의에서 배운 기초 내용을 토대로 시계열 예측에 적합한 **RNN, 그 중에서도 LSTM**을 이용하여 모델을 생성하며, S&P 500 지수, 원/달러 환율 등 **주요 글로벌 경제 지표를 독립변수**로, **삼성전자의 종가를 종속변수**로 설계하였다.
>
> 


<br/>


---


## 프로젝트 개요

<br/>

- **주요 목표**: 다변량 시계열 데이터를 바탕으로 삼성전자 종가를 예측
- **사용 모델**: PyTorch 기반 LSTM
- **비교 모델**: 다중선형회귀모델
- **프로젝트 기간**: 2023년 10월 ~ 2023년 12월
- **데이터 출처**: [Investing.com](https://www.investing.com/)


<br/>


---

## 사용된 경제 지표

<br/>


| Feature                     | 설명                         |
|----------------------------|------------------------------|
| samsung_closeprice         | 삼성전자 종가 (예측 대상)    |
| snp_closeprice             | S&P 500 지수                |
| wondollar_closeprice       | 원/달러 환율                |
| copper_closeprice          | 구리 선물 가격               |
| nasdaq_closeprice          | 나스닥 지수                 |
| nasdaq_volume              | 나스닥 거래량               |
| kospi_closeprice           | 코스피 지수                 |
| kospi_volatility           | 코스피 변동률 (%)           |


<br/>


---

## 데이터 전처리

<br/>

- 날짜 데이터 datetime 변환 및 인덱싱
- 문자열 → float 형식 변환 (쉼표/기호/단위 기호 제거)
- 결측값 처리: 선형보간 + forward/backward fill
- `StandardScaler`를 활용한 전체 컬럼 정규화 (종속 변수 포함)

<br/>

---

## 모델 (PyTorch LSTM)

<br/>

```python
class XavierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers):
        ...
```

- 입력 크기: 7 (경제 지표 수)
- window 크기: `sequence_length = 30`
- 가중치 처리: **Xavier 초기화**
- 출력층: Linear(hidden_size * seq_len → 1)


<br/>


---

## 결과

<br/>

| 모델 | MSE | 하이퍼파라미터 |
|--------|-----|--------|
| Model_1 | 0.281 | dropout=0, lr=1e-3 |
| Model_2 | 0.352 | dropout=0.25, lr=1e-4 |
| 다중선형회귀 | **0.107** |  |


<br/>


---

## 평가 및 한계

<br/>

- LSTM은 시계열 패턴을 학습하였으나, 일반화에 어려움이 있었음
- 비교 모델인 다중선형회귀가 오히려 더 낮은 MSE를 기록함
- Model_1은 과적합 경향, Model_2는 과소적합 경향을 보임

<br/>

---

## 개선 방향

<br/>

- 데이터 양 증가 (6년 이상)
- 가중치 규제 (L2 regularization 등)
- 모델 복잡도 조정 (hidden size, layer 수)

<br/>

---

## 파일 명세


<br/>


| 파일 | 설명 |
|--------|--------|
| `lstm(pytorch)_prediction_model.ipynb` | PyTorch LSTM 모델링 코드 |
| `multilinear_prediction_model(control_group).ipynb` | 대조군 다중선형회귀모델 코드 |
