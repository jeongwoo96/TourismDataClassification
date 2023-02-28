# TourismDataClassification

Main Project 1


<img src="https://user-images.githubusercontent.com/112039781/211071653-040a1845-a12a-43e8-bda9-67120644afce.png" width="600 " height="480">


목록 | 파일명 | 설명 |
------------|------|-------|
DataAugmentation.ipynb | [DataAugmentation.ipynb](https://github.com/jeongwoo96/TourismDataClassification/blob/master/DataAugmentation.ipynb)| 데이터 증강 코드 |
Image_RegNetY120.ipynb | [Image_RegNetY120.ipynb](https://github.com/jeongwoo96/TourismDataClassification/blob/master/Image_RegNetY120.ipynb)| 이미지 모델 |
Text_KlueRoBERTa.ipynb | [Text_KlueRoBERTa.ipynb](https://github.com/jeongwoo96/TourismDataClassification/blob/master/Text_KlueRoBERTa.ipynb)| 텍스트 딥러닝 모델 |
Text_MachineLearning.ipynb |[Text_MachineLearning.ipynb](https://github.com/jeongwoo96/TourismDataClassification/blob/master/Text_MachineLearning.ipynb)| 텍스트 머신러닝 모델 |
adjective.csv | [adjective.csv](https://github.com/jeongwoo96/TourismDataClassification/blob/master/adjective.csv)| 증강용 형용사 사전 |
adverb.csv | [adverb.csv](https://github.com/jeongwoo96/TourismDataClassification/blob/master/adverb.csv)| 증강용 부사 사전 |
nsmc_stopwords.txt | [nsmc_stopwords.txt](https://github.com/jeongwoo96/TourismDataClassification/blob/master/nsmc_stopwords.txt)| 불용어 사전 |
sim_data.csv | [sim_data.csv](https://github.com/jeongwoo96/TourismDataClassification/blob/master/sim_data.csv)| 유의어 사전 |


##

## 2022 관광데이터 AI 경진대회 (DACON)


### 목차

<img src="https://user-images.githubusercontent.com/112039781/213433493-62a62c78-265e-40df-942d-b10616c031f3.png" width="680 " height="500">


### 1. 프로젝트 기획

<img src="https://user-images.githubusercontent.com/112039781/213434106-97db3d93-8e5b-4872-95e0-930268081e78.png" width="680 " height="500">


관광지 **이미지**와 **설명문**을 입력으로 넣어 어떤 관광지인지 **카테고리 예측**(소분류)


카테고리 분류를 인공지능의 힘으로 자동화 한다면, 더 적은 공공의 예산으로 더 많은 POI 데이터 생성 가능 !

##


<img src="https://user-images.githubusercontent.com/112039781/213434191-fa0719f7-e35f-40cd-b1e7-28e74c15ff42.png" width="680 " height="500">
<img src="https://user-images.githubusercontent.com/112039781/213434303-50a70e00-0c68-41b3-a07a-e84c7fba659b.png" width="680 " height="500">
<img src="https://user-images.githubusercontent.com/112039781/213434391-308be6c5-f58f-43c6-8828-688c36f3c018.png" width="680 " height="500">
<img src="https://user-images.githubusercontent.com/112039781/213434420-42714459-a861-4fa0-b76c-a5ad2e563b06.png" width="680 " height="500">


### 2. 전처리

<img src="https://user-images.githubusercontent.com/112039781/213434785-42280e53-66c1-4ccf-aee1-0f5711ad7009.png" width="680 " height="500">
<img src="https://user-images.githubusercontent.com/112039781/213434880-9329ca15-84f5-4b3f-8991-8fcebccb845a.png" width="680 " height="500">
<img src="https://user-images.githubusercontent.com/112039781/213434945-3611ae9d-18b2-4b04-8b5e-bff7d2da3464.png" width="680 " height="500">
<img src="https://user-images.githubusercontent.com/112039781/213435016-2c61663b-9557-4f5a-b0bc-8b745a3172ec.png" width="680 " height="500">


- 형용사 증강만 했을 때 성능이 향상되었던 이유


 -한 문장 내의 명사 갯수가 많기 때문에, 명사 앞에 랜덤으로 형용사를 삽입하는 경우 문장의 의미를 크게 손상시키지 않으면서, 유사도를 줄인채 증강을 할 수 있었기 때문으로 추정
 
 
 -부사 증강의 경우 문장 내에 동사가 적어 증강된 문장이 유사도가 너무 높고, 유의어나 Back Translation은 원래 문장의 의미를 훼손할 수 있으므로 효과가 적다고 추정


##


<img src="https://user-images.githubusercontent.com/112039781/213435083-e87c4bca-7772-4034-b17a-c2054c25594a.png" width="680 " height="500">
<img src="https://user-images.githubusercontent.com/112039781/213435142-dd2d9d5b-de8f-4b42-9474-24e3a22fe61d.png" width="680 " height="500">
<img src="https://user-images.githubusercontent.com/112039781/213435205-3b91d667-56b5-4e2a-83aa-d8b039ed74e8.png" width="680 " height="500">


### 3. 모델 학습

<img src="https://user-images.githubusercontent.com/112039781/213435859-5b4ce416-3598-4a36-8b5b-e10b0180f48d.png" width="680 " height="500">

- Support Vector Machine이 단일 ML 모델 중 가장 성능이 높았던 이유 


SVM은 범주를 예측하는데 사용이 가능하며, 오류 데이터의 영향이 적다. 또한 과적합 되는 경우가 적기 때문에 단일 모델 중에서 관광 데이터 분류에 적합했다고 판단

##


<img src="https://user-images.githubusercontent.com/112039781/213435958-1a095e4a-3a6d-4540-a631-9d73d6e3ea79.png" width="680 " height="500">

- RegNetY120 선정 이유

기존의 다양한 Networks 설정과 비교해서 성능이 뛰어나며, GPU환경에서 빠른 속도를 보여준다.

120층 모델을 사용한 것은 사용 가능한 GPU자원으로 감당 가능한 것이 120층이 최대였기 때문이다.

##


<img src="https://user-images.githubusercontent.com/112039781/213436019-90c1f733-813a-49a3-9178-75f4ce7fee58.png" width="680 " height="500">

- Klue/RoBERTa 선정 이유

Klue에서 학습시킨 BERT 계열 모델


한국에서 pre-trained가 매우 잘 되어 있어 여러 task에 fine-tuning하기 적합


모델을 더 많은 데이터로 오래 그리고 더 큰 batch로 학습


더 긴 문장들에 대해 학습


Mask를 Dynamic하게 바꿔줌(epoch마다 중복되지 않도록)


##


<img src="https://user-images.githubusercontent.com/112039781/213436081-2df81bae-be89-4117-9272-7ac6299fda60.png" width="680 " height="500">


- 멀티모달은 HJOK님의 코드 참조

[[New Baseline] Roberta + ViT (1fold public score 0.8284)](https://dacon.io/competitions/official/235978/codeshare/6861?page=1&dtype=recent)


Image Features와 Text Features를 횡방향으로 연결하는 것이 종방향으로 연결하는 것보다 좋은 성능을 보임


##


<img src="https://user-images.githubusercontent.com/112039781/213436123-b673dc60-f7c3-4fbe-9af3-f6abdd0988be.png" width="680 " height="500">


### 4. 모델 핸들링 & 앙상블

<img src="https://user-images.githubusercontent.com/112039781/213436610-a7f847fc-d87c-40ac-9662-1acd3444a395.png" width="680 " height="500">
<img src="https://user-images.githubusercontent.com/112039781/213436808-fc529f8e-87c5-498e-817b-284b268f3189.png" width="680 " height="500">


### 5. 결과 및 피드백

<img src="https://user-images.githubusercontent.com/112039781/213437387-14dd8a41-8fd4-421a-a016-36cb4a687c32.png" width="680 " height="500">
<img src="https://user-images.githubusercontent.com/112039781/213437447-bdfcb642-df6a-407a-8aee-6fd7309b0e5c.png" width="680 " height="500">
<img src="https://user-images.githubusercontent.com/112039781/213437511-94af757b-95bb-4473-ad35-863d2979c76b.png" width="680 " height="500">
<img src="https://user-images.githubusercontent.com/112039781/213437571-839af2e8-b72c-4462-8723-f72a71217ed3.png" width="680 " height="500">

