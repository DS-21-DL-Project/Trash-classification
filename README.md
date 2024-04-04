![001](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/e55d08c5-d7c2-40f6-9eb1-17f23f966679)

---

## Introduction
<blockquote> &nbsp;제주의 클린 하우스 운영과 같이 쓰레기 분리 수거를 실시하는 지역에서는 </br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;분리 배출 전, 폐기물의 정확한 분류가 중요한 문제로 대두되고 있습니다. </br> &nbsp;한국 전역에서는 생활 폐기물 발생량이 증가하는 추세이며 분리 배출 오분류로 인한 경제적 손실 또한 증가하고 있습니다. </blockquote> </br>

### 이 프로젝트는 ...
#### &nbsp;&nbsp;&nbsp; Yolo의 고도의 정확성과 신속성을 활용해 생활 폐기물을 식별하고 분류함으로써 </br> 
#### &nbsp;&nbsp;&nbsp; 재활용 가능한 자원을 효율적으로 회수하고 환경 오염을 줄이는 데 기여하고자 합니다. </br>
#### &nbsp;&nbsp;&nbsp; 이를 통해 지구 환경을 보호하고 지속 가능한 발전을 이끌어내는 사회적 가치를 창출하고자 합니다. </br></br>

---

## Contents
- [1. 프로젝트 소개](#1-프로젝트-소개)
  * [배경](#배경)
  * [프로젝트 개요](#프로젝트-개요)
- [2. 데이터 수집](#2-데이터-수집)
  * [진행 과정](#진행-과정)
  * [캐글 데이터](#캐글-데이터)
  * [직접 수집 데이터](#직접-수집-데이터)
- [3.데이터 분석 & 시각화](#3-데이터-전처리)
- [4. 모델링](#4-모델링)
  * [모델 활용](#모델-활용)
- [5. 프로젝트 한계 및 과제](#5-프로젝트-한계-및-과제)
  * [한계점](#한계점)
  * [향후 과제](#향후-과제)

---

## 1. 프로젝트 소개
### 배경
- 제주에서는 가정의 올바른 생활 폐기물 분리 배출로 재활용되는 자원의 양을 높이기 위해 클린 하우스를 운영 중이지만 </br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;분리 배출 전 폐기물 분류 기준에 대해 모호한 경우 발생
- 생활 폐기물 발생량은 연도별 증가하고 있으며, 전국 평균 발생량 또한 증가 추세
- 분리 수거 오분류로 인한 경제적 손실은 2018년 한국환경공단 추정 약 4,000억 원에서 2022년 환경부 추정 약 8,000억원까지 증가
- 이에 분리 배출의 정확도 향상을 위해 생활 폐기물 데이터를 수집 및 분석해 객체를 탐지 및 분류시켜 </br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;가정의 분리 배출에 대한 기준 확립을 도와 지구 환경을 보호하고 지속 가능한 발전에 기여하고자 함 </br></br>


### 프로젝트 개요
1. 프로젝트명: **EcoSort Helper : Yolo를 활용한 분리 배출 도우미**
2. 수행자: 강수정, 강호진
3. 수행 기간: 1개월 (2024.3.04 \~ 4.03)
4. 목표: YOLO를 활용한 생활 폐기물 데이터 "객체 탐지 및 분류"  </br></br>

<h3  dir="auto"><a id="user-content--tech-stack-" class="anchor" aria-hidden="true" tabindex="-1" href="#-tech-stack-"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a> Tech Stack </h3>

&nbsp; &nbsp; Language & Library <p align="justify">&nbsp; &nbsp; &nbsp;<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"/> &nbsp; <img src="https://img.shields.io/badge/ultralytics-150458?style=for-the-badge&logo=ultralytics&logoColor=white"/> &nbsp; <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/> &nbsp; <img src="https://img.shields.io/badge/opencv-5AC710?style=for-the-badge&logo=opencv&logoColor=white"/> </br>
&nbsp; &nbsp; Other <p align="justify">&nbsp; &nbsp; &nbsp;<img src="https://img.shields.io/badge/Roboflow-A100FF?style=for-the-badge&logo=roboflow&logoColor=white"> &nbsp; <img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252"/> &nbsp; <img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white"/> &nbsp; <img src="https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white"/>

---

## 2. 데이터 수집
### 진행 과정
캐글에서 제공하는 'Garbage Image Dataset'를 활용하였고, </br> 
한국의 생활 폐기물에 대한 데이터가 부족하였기에 자택, 동네 공원, 쓰레기 처리장 등 직접 데이터를 수집하였습니다. 

### 캐글 데이터
![002](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/d301fc49-6189-4c4f-ac40-e7b33971d3f3)
 
- 일반쓰레기&nbsp; /&nbsp; 플라스틱&nbsp; /&nbsp; 종이&nbsp; /&nbsp; 고철&nbsp; /&nbsp; 유리
  
### 직접 수집 데이터
![003](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/564e2e5e-b3c4-4e06-b5da-90f669b2323f)

- 스티로폼&nbsp; /&nbsp; 복합 쓰레기&nbsp; /&nbsp; 배경&nbsp; /&nbsp; 비닐류&nbsp; /&nbsp; 일반 쓰레기&nbsp; /&nbsp; 고철류(한국 버전)&nbsp; /&nbsp; 플라스틱(라벨 없는)

---
## 3. 데이터 전처리

환경부에서 고시하는 분리 배출 기준에 따라 생활 폐기물 데이터를 다음 7개의 기준으로 나누었습니다. </br>
- 종이류
- 플라스틱류
- 캔*고철류
- 유리류
- 비닐류
- 스티로폼
- 일반쓰레기

</br></br>

✅ **라벨링 작업_Roboflow**

![image](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/586e8499-e37e-479d-81d7-54c8695861c2)



- **Roboflow:** 데이터 셋 생성 및 전처리 작업, 증강 작업

---

## 4. 모델링

모델링 절차에 대해서 설명 드리겠습니다.

### 모델링 절차

생활 폐기물 데이터를 YOLO v8 Nano에 여러차례 학습시키며 큰 영향 준 분기점 표시해 보았는데요

모델링 절차에 대한 설명은 아래 그림과 같은 순서 대로 설명해 드리도록 하겠습니다.

![제목을 입력해주세요_-005 (1)](https://github.com/DS-21-DL-project/youquiz/assets/83691399/bc218c68-9a6a-4ec1-944e-ea524c58b0a5)

</br></br>
#### 1차 시도

![012](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/e900c0bb-6a89-4473-8a02-25bedf8b0575)


1차시도 당시에는 kaggle에서 얻은 데이터들 위주로 로보플로우로 라벨링하여 yolo 도큐먼트에 있는 방법데로 학습을 시켰는데요

그렇다 보니 학습 방법이나 증강기법에는 신경쓰지 않고 학습을 진행 하였습니다.

특이사항으로 kaggle에서 얻은 데이터가 해외 기준의 폐기물 데이터다 보니 국내 기준과 상이하여 데이터의 분포가 상당히 불균형 한것을 알 수 있습니다.

특히 스티로폼 데이터가 10개도 안되는 상태로 진행하여 아직까지는 많이 부족한 모델이였습니다.

</br></br>

#### 4차 시도

![013](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/5171bebd-b377-4bad-b972-b8b0287cde52)


4차시도 당시에는 부족한 데이터들을 추가 적으로 라벨링을 진행하였고 

학습방법에서 제 노트북 하드웨어가 버틸수 있는 최대 값으로 배치사이즈 24에 이미지 640으로 진행하였습니다.

여기까지 보면 왜 4차 시도가 분기점이지? 싶으실텐데요

![014](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/241ccaa3-ba67-4fec-bc67-57d659e887f6)

4차 시도 당시에 추가한 데이터들의 대부분이 이 사진처럼 한 화면에 수많은 객체가 들어있는 사진들이였습니다.

이러한 사진들은 객체의 겹칩이나 부분적으로 가려짐, 조명과 그림자 등의 요인으로 인해 무지막지하게 많은 데이터를 학습시키지 않은 이상 정확도를 떨어뜨릴수 있다는 사실을 알게되었습니다,

오른쪽의 차트를 보시면 1차시도 당시와 비교했을때 yolo에서 가장 중요한 지표중의 하나인 mAP50이 1차 시도 당시 보다 많이 떨어졌음을 알 수 있었습니다.

따라서 이후 학습 시에는 이러한 데이터들을 대부분 배재 하고 한 화면에 많아도 10개 정도의 객체를 가지고있는 데이터들 위주로 학습을 진행 하였습니다.

</br></br>

#### 11차 시도

![015](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/20177a9b-6598-4578-9b8f-7aaebdcd5183)

4차시도에서 얻은 방향성을 토대로 데이터 수집을 계속 하였고 하이퍼파라미터를 추가하여 정적인 데이터에 대해서는 어느 정도의 정확도를 나타내는 모델을 완성 하였습니다

하지만 저희가 원하는 모델은 사용자가 쓰레기를 보여줫을 때 실시간으로 대답을 내놓는 모델이 였기 때문에 동영상데이터에 모델을 학습 시켜 보았습니다

![016](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/a048c220-478d-4487-aaf2-d4acea87b954)

동영상 데이터에 학습시켜본 결과 3가지 정도의 문제가 발생 하였는데 

첫번째로 배경 데이터를 객체로 인식하는 문제였습니다.
데이터를 라벨링할 당시 배경 데이터에 대해서는 일절 학습을 시키지 않았는데 이것으로 인해 배경 데이터를 오탐하는 문제가 발생 하였고

두번째로 동영상 데이터는 영상으로 프레임단위로 잘라 분석을 진행하는데 프레임단위로 자르다 보니 객체가 흐리지게 되고  이것을 yolo가 분석하게되면 정확도가 많이 떨어지는 문제가 발생하였습니다.

세번째로는 일반쓰레기 데이터에 대한 정확성입니다.
일반쓰레기는 범주가 상당히 넓어 데이터가 지금까지의 모델로는 정확도가 5~60% 대에서 머물러있었습니다.

이러한 문제를 해결하고자 첫번째로 로보플로우에서 제공하는 증강기법의 흐림 기능을 사용하여 흐린 사진을 추가적으로 만들어 동영상 데이터에 강해지도록 하였고 배경이미지 또한 상당수 추가하여 배경에도 강해지도록 하였습니다

또 일반 쓰레기에 대해서는 프로젝트 마감일까지 계속해서 수집하여 추가적으로 보완하였습니다

</br></br>


#### 최종 모델
![017](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/b8e24ea0-9eb4-4096-8407-6d63a772cf03)

앞에서 말씀드린 문제점들을 최대한 보완하였고 수집한 데이터들의 분포 또한 초기와 비교하였을때 균일해 졌습니다

또 하이퍼파라미터 튜닝을 사용하여 저희 모델에 최적의 값의 파라미터를 구해 모델을 보완하였습니다

![018](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/e1508a09-36e7-4988-a48c-f5c4960758ed)

하이퍼파라미터는 10번의 에포크를 100번씩 아담w 옵티마이저를 사용 하여 진행하였습니다

조금더 시도를 해보고 싶었습니다만 하드웨어의 한계로 10번의 에포크를 100번 하는데만 15시간이 걸려 시간상 추가적인 시도가 어렵다 판단하여 여기서 구한 파라미터값을 최종 모델에 적용하였습니다



</br></br>

## 5. 성능평가

### mAP50, mAP50 - 90

![019](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/58f2ae4c-6568-4a0a-b5b0-5b41bfdaac57)

yolo 모델의 성능을 평가 할때 가장 중요한 지표는 mAP 50과 mAP 50-90인데 지표를 보시면 최종 모델로 갈수록 모델이 개선 되고있음을 확인 할 수 있었고 앞에서도 말씀 드린 4차 시도 당시의 지표가 상당히 떨어졌던 것을 확인 할수있었습니다.

#### 참고

mAP 50 : 객체 탐지 모델이 실제 객체와 IoU가 50% 이상일 때의 정확도 평균
mAP 50-90 : IoU가 50% - 90% 일 때, 이 범위 내 각각 IoU 값에 대한 모델의 정확도의 평균 값 
IoU : 모델이 예측한 객체의 경계 상자와 실제 객체의 경계 상자가 얼마나 잘 겹치는지를 나타내는 지표



### Precision(정밀도), Recall(재현율)

![020](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/854a8d5c-b725-4706-bb7c-cdda086ed5f9)

정밀도와 재현율에 대해서도 마찬가지로 최종 모델로 갈수록 성능이 올라갔음을 확인할수 있었습니다.

### 정확도

![021](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/5071bd3b-0fec-41db-8eec-81d1d06d51a9)

이전까지의 모델에서는 일반쓰레기에 대한 정확도가 50~60사이에 머물러 있엇는데 비해 최종 모델에서는 정확도가 많이 올라간것을 확인할수 있었습니다

### 데모 영상 1
[![Video Label](http://img.youtube.com/vi/shorts/9rdbIPDRP-0/0.jpg)](https://youtu.be/shorts/9rdbIPDRP-0)

[![Video Label](http://img.youtube.com/vi/shorts/JA3J8qMjWiY/0.jpg)](https://youtu.be/shorts/JA3J8qMjWiY)

Roboflow에서 제공해주는 모델을 스마트폰에서 실행하여 테스트해 보았습니다.

### 데모 영상 2

[![Video Label](http://img.youtube.com/vi/CVucEqj2OW4/0.jpg)](https://youtu.be/CVucEqj2OW4)

배경 데이터를 추가하기 전인 11차 시도와 최종 모델의 차이를 비교해 보았습니다.

## 6. 프로젝트 한계 및 과제
### 한계점
- 모델을 충분히 학습시키고 더 좋은 성과를 내기에는 데이터가 조금 부족했었습니다. 만약 시간과 데이터가 더 주어진다면, 더 좋은 모델을 만들어 정확한 예측을 할 수 있을 거라 생각합니다. 
- 직업군 분류 작업이 잘 진행되지 않았던 거 같습니다. 직접 크롤링을 하여 얻은 데이터를 3명이서 분류를 진행하다 보니 기준이 저마다 다르거나 애매한 경우가 있어 직업군별 분류가 좋지 못한 결과가 나온거 같습니다.
- 최대한 정보를 모아 데이터를 이해해 모델을 학습시켰지만, 아쉽게도 버려야 하는 컬럼들이 존재했습니다. 활용하지 못한 변수도 포함해 적용해 본다면, 더 나은 성과를 내지 않았을까 생각됩니다.
- 예측했을 때의 그래프만 보았을 때는 좋은 결과를 나타냈지만 실제로 적용해 보았을 때 엄청난 과적합이 발생하여 실제로 활용하기에는 어려움이 있다는 결론이 나와 많이 아쉬웠습니다.

### 향후 과제
- 현재는 229화까지의 데이터 셋을 사용하여 진행하였지만 몇 년 뒤에 유퀴즈 프로그램이 남아있다면 더 많은 데이터를 사용하여 더 좋은 모델을 만들 수 있지 않았을까 싶습니다.
- 하이퍼파라메터를 다루는 방법에 대해 미숙했습니다. 아직 하이퍼파라메터의 기능이 무엇인지를 정확이 인지하지 못해 모든 경우를 돌려야 하는 점이 최적의 값을 찾는 데 오랜 시간을 걸리게 하였고 일부 모델에서는 몇몇 조건을 제외하여 진행하였습니다.
