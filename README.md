# 시계열 및 머신러닝 교재

저자 : 한희준 ([@heejoonhan](https://github.com/heejoonhan)) <br>

## 안내

본 교재에서 사용하는 모든 데이터와 코드는 아래 링크를 통해 다운로드 받아 사용할 수 있습니다.

[전체 코드 및 데이터](https://github.com/heejoonhan/Time-Series-and-Machine-Learning-Textbook/tree/main/%EC%BD%94%EB%93%9C%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0)

다운로드 받은 이후 아래 "코드 실행 주의사항"을 참고하여 실습을 진행하시면 됩니다. 교재 각 목차에 해당되는 코드를 다운로드 받기 위해서는 아래 "목차 및 해당 파트 코드"의 링크를 참고해주시기 바랍니다.

## 코드 실행 주의사항

코드를 원활하게 실행하기 위해서는 데이터의 위치 혹은 작업 공간을 올바르게 설정해야 합니다.

**데이터 세팅**

본 교재에서 사용되는 데이터는 모두 아래 링크에 저장되어 있습니다. 아래 'Data' 폴더를 다운로드 받아 실습을 진행할 수 있습니다. <br>

[데이터](https://github.com/heejoonhan/Time-Series-and-Machine-Learning-Textbook/tree/main/%EC%BD%94%EB%93%9C%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0/Data)

다운로드 받은 데이터 폴더 혹은 데이터를 작업공간에 위치시켜주시기 바랍니다.

**작업 공간 설정**

아래 코드는 R script가 저장된 폴더를 작업공간으로 설정해주는 코드입니다. 다음과 같이 작업공간을 설정할 경우, 작업공간 안에 사용할 데이터를 위치시켜야 합니다.

```CLI
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
```

아래와 같이 "./Data/" 폴더를 작업공간으로 설정하여 사용해야 할 데이터를 일일이 옮기지 않아도 데이터를 불러올 수 있습니다.

```CLI
setwd("..../Data/")
```

## 목차 및 해당 파트 코드
1. [시계열의 정상성 및 시계열 분석의 기초](https://github.com/heejoonhan/Time-Series-and-Machine-Learning-Textbook/tree/main/%EC%BD%94%EB%93%9C%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0/Ch1%20and%20Ch2) <br>

2. [시계열 예측 절차 및 평가 방법](https://github.com/heejoonhan/Time-Series-and-Machine-Learning-Textbook/tree/main/%EC%BD%94%EB%93%9C%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0/Ch1%20and%20Ch2) <br>

3. 머신러닝 소개 및 지도학습 <br>

4. [선형 회귀 모형과 국소 추정](https://github.com/heejoonhan/Time-Series-and-Machine-Learning-Textbook/tree/main/%EC%BD%94%EB%93%9C%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0/Ch4%20to%20Ch6) <br>

5. [의사결정나무 기반 방법](https://github.com/heejoonhan/Time-Series-and-Machine-Learning-Textbook/tree/main/%EC%BD%94%EB%93%9C%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0/Ch4%20to%20Ch6) <br>

6. [인공신경망 기반 모형](https://github.com/heejoonhan/Time-Series-and-Machine-Learning-Textbook/tree/main/%EC%BD%94%EB%93%9C%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0/Ch4%20to%20Ch6) <br>

7. [사례 연구 : 미국 인플레이션 예측 및 한국 주식시장 변동성 예측](https://github.com/heejoonhan/Time-Series-and-Machine-Learning-Textbook/tree/main/%EC%BD%94%EB%93%9C%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0/Ch7%20US%20inflation) <br>

8. [벡터자기회귀모형](https://github.com/heejoonhan/Time-Series-and-Machine-Learning-Textbook/tree/main/%EC%BD%94%EB%93%9C%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0/Ch8%20and%20Ch9) <br>

9. [조건부 분산과 변동성 모형](https://github.com/heejoonhan/Time-Series-and-Machine-Learning-Textbook/tree/main/%EC%BD%94%EB%93%9C%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0/Ch8%20and%20Ch9) <br>

부록 1. [R 설치 및 사용](https://github.com/heejoonhan/Time-Series-and-Machine-Learning-Textbook/tree/main/%EC%BD%94%EB%93%9C%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0/%EB%B6%80%EB%A1%9DA%20%20R%20%EA%B8%B0%EB%B3%B8) <br>

부록 2. [Python 설치 및 사용](https://github.com/heejoonhan/Time-Series-and-Machine-Learning-Textbook/tree/main/%EC%BD%94%EB%93%9C%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0/%EB%B6%80%EB%A1%9D%20Python%20%EA%B8%B0%EB%B3%B8) <br>
