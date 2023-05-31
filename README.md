# KLUE - RE(Relation Extraction)
## 👋 팀원 소개
<div align="center">
<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/lectura7942"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/81620001?v=4"/></a>
            <br />
            <strong>권지은</strong>
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/JLake310"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/86578246?v=4"/></a>
            <br />
            <strong>김재연</strong>
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/hoooolllly"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/126573689?v=4"/></a>
            <br />
            <strong>박영준</strong>
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Da-Hye-JUNG"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/96599427?v=4"/></a>
            <br />
            <strong>정다혜</strong>
        </td>
            <td align="center" width="150px">
            <a href="https://github.com/yunjinchoidev"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/89494907?v=4"/></a>
            <br />
            <strong>최윤진</strong>
        </td>
    </tr>
    <tr height="50px">
        <td align="center">
            <a href="https://github.com/lectura7942">:octocat: GitHub</a>
        </td>
        <td align="center">
            <a href="https://github.com/JLake310">:octocat: GitHub</a>
        <td align="center">
            <a href="https://github.com/hoooolllly">:octocat: GitHub</a>
        </td>
        <td align="center">
            <a href="https://github.com/Da-Hye-JUNG">:octocat: GitHub</a>
        </td>
            <td align="center">
            <a href="https://github.com/yunjinchoidev">:octocat: GitHub</a>
        </td>
    </tr>
</table>
</div>

</br>

## 📌 담당 역할
- 권지은 - 데이터 전처리, 모델 선정을 위한 비교 실험, TAPT, studio-ousia/mluke-large 모델 튜닝
- 김재연 - 베이스라인 리팩토링, entity embedding layer 및 LSTM classifier를 활용한 Custom 모델 제작, CoRE 논문 구현
- 박영준 - 데이터 전처리, entity marker 추가, 데이터 증강, 모델학습
- 정다혜 - EDA, 데이터 전처리, Subject entity type 반영 실험, Ensemble 실험 및 결과 분석
- 최윤진 - 데이터 분석, 논문 조사, 모델 학습, 베이스라인 개선, 협업 환경 세팅

</br>

## 📃 Task 개요
문장 속에서 단어 간에 관계성을 파악하는 것은 의미나 의도를 해석함에 있어서 많은 도움을 준다.

![Untitled](https://github.com/boostcampaitech5/level2_klue-nlp-07/assets/86578246/0880c62f-ffd7-47a1-b077-5da1945a8219)

그림의 예시와 같이 요약된 정보를 사용해 QA 시스템 구축과 활용이 가능하며, 이외에도 요약된 언어 정보를 바탕으로 효율적인 시스템 및 서비스 구성이 가능하다.

**관계 추출(Relation Extraction)** 은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요하다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있다.

문장, 단어에 대한 정보를 통해, 문장 속에서 단어 사이의 관계를 추론하는 모델을 학습해야 한다. 이를 통해 인공지능 모델이 단어들의 속성과 관계를 파악하며 개념을 학습할 수 있다.

</br>

## 📊 EDA
- Labels
![1](https://github.com/boostcampaitech5/level2_klue-nlp-07/assets/86578246/c7230f58-8c3a-4918-b45a-7e55ae48c426)

- 레이블 별 데이터 개수 히스토그램을 그려보았을 때 no relation 레이블이 가장 많고, 레이블별 불균형이 심하다는 점을 확인했다. 
따라서 validation 과 k-fold 를 구현할 때 train 과 validation 의 학습 데이터 셋의 라벨 분포가 같도록 Stratified 방법을 이용했다.
![Untitled](https://github.com/boostcampaitech5/level2_klue-nlp-07/assets/86578246/50d4c5e7-9937-4866-8e3a-0f824aa9020b)

- Subject entity type은 비슷한 비율을 차지하는 반면, Object entity type에서 불균형을 확인했다.
<img width="1336" alt="image" src="https://github.com/boostcampaitech5/level2_klue-nlp-07/assets/86578246/44452592-dd2f-41d4-ad4c-dd7d6ba745ef">

- 토큰화 후 문장 길이가 30~50에 몰려있다는 걸 확인했다.
- test, train 문장 길이 비율이 비슷함을 확인했다.

## 📚 Preprocess
### 〈〉, 《》를 '로 통일
| seed | 기존 f1 | 통일 f1 |
| --- | --- | --- |
| 5 | 84.401 | 84.634 |
| 11 | 84.543 | 84.523 |
| 42 | 84.567 | 84.305 |
| 평균 | 84.504 | 84.487 |

### 이상데이터 처리
- EDA과정에서 이상 데이터가 있는 것을 확인했다.
- 이상 데이터의 기준은 다음과 같다.
    - Subject entity의 type과 label의 첫번째가 다른 경우
    - Subject entity의 type에 조직(ORG)이 사람(PER)으로 테깅된 경우와 그 반대 경우

### Entity marker 추가
![image](https://github.com/boostcampaitech5/level2_klue-nlp-07/assets/86578246/980e36b1-7479-4721-90f7-802304d4c362)
> 출처:[An Improved Baseline for Sentence-level Relation Extraction](https://arxiv.org/abs/2102.01373)

위 논문에 소개된 기법을 차용하여 모델에게 엔티티의 위치를 알려줄 수 있는 방안을 모색하였다.
- Entity marker 예시
  - 기존 : 비틀즈 [SEP] 조지 해리슨 [SEP] 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.
  - Method1: 〈Something〉는 #^PER^조지 해리슨#이 쓰고 @*ORG*비틀즈@가 1969년 앨범 《Abbey Road》에 담은 노래다.
  - Method2: 〈Something〉는 #^person^조지 해리슨#이 쓰고 @*organization*비틀즈@가 1969년 앨범 《Abbey Road》에 담은 노래다.
- 
| Method | micro_f1 | auprc |
| --- | --- | --- |
| 기존 | 66.2409 | 68.2921 |
| Method 1 | 68.2769  | 69.7847 |
| Method 2 | 68.6772 | 72.1613 |

### 데이터 증강
EDA를 통해 데이터의 불균형이 심하다는 것을 확인했다. 또한 아래 그래프를 보면 validation dataset의 검증 결과를 봤을 때 가장 많았던 no relation은 prob값에 많이 등장한 것을 볼 수 있었고 반대로 데이터가 거의 없었던 per: siblings은 prob값에 항상 0에 가깝게 나타난 것을 볼 수 있다. 이러한 불균형과 편향을 해결하고자 데이터 증강을 진행했다.
<img width="1313" alt="image" src="https://github.com/boostcampaitech5/level2_klue-nlp-07/assets/86578246/be442b9b-e759-451d-8918-1e8586aeef67">

[한국어 상호참조해결을 위한 BERT 기반 데이터 증강 기법](https://koreascience.kr/article/CFKO202030060835857.pdf)을 참고하여 MLM을 이용해 데이터의 문맥에 영향을 끼치지 않는 단어로 치환하여 증강하는 방법으로 레벨 예측을 잘 못하던 데이터를 증강시키면 불균형과 편향이 해결될 것이다.
기존 결과의 accuracy가 0.8 이하인 라벨들에 대해서 증강을 진행하였고, 엔티티를 제외한 나머지 토큰들을 10퍼센트 확률로 mask토큰으로 바꾼 후 MLM을 이용하여 채워 넣는 방식으로 증강 후 성능을 확인하였다.
- 실험 결과

| Method | micro_f1 | auprc |
| --- | --- | --- |
| 기존 | 72.1022 | 76.6324 |
| 증강 후 | 71.8970 | 76.5956 |

## 🗄️ Model
- Model test 1: 학습률 3e-5로 7 에폭씩 파인튜닝한 후 micro f1 점수이다. mixed precision을 사용했다.

| 모델 | 배치 크기 16 | 배치 크기 32 |
| --- | --- | --- |
| studio-ousia/mluke-large | **84.629** | 83.563 |
| xlm-roberta-large | 84.008 | **84.167** |
| kykim/bert-kor-base | 83.205 | 83.405 |
| klue/bert-base | 83.119 | 83.059 |
| sentence-transformers/xlm-r-large-en-ko-nli-ststb | 82.814 | 82.874 |
| kykim/funnel-kor-base | 82.556 | 82.781 |
| snunlp/KR-ELECTRA-discriminator | 82.483 | 81.832 |
| bert-base-multilingual-uncased | 81.167 | 81.319 |
| beomi/KcELECTRA-base-v2022 | 81.145 | 79.932 |
| skt/kogpt2-base-v2 | 78.979 | 79.637 |

- Model test 2: 학습률 5e-5로 7 에폭씩 파인튜닝한 후 micro f1 점수이다.

| 모델 | 배치 크기 16 | 배치 크기 32 |
| --- | --- | --- |
| klue/roberta-large | **84.956** | **84.925** |
| monologg/koelectra-base-v3-discriminator | 82.953 | 81.650 |
| monologg/kobert | 57.763 | 57.249 |

### Entity Embedding Layer 추가
1. entity_loc_ids를 추가하여 엔티티들의 위치 추가 
    
    토큰화 단계에서 문장을 토큰화한 다음에, 엔티티의 토큰들의 위치를 찾아 subject 토큰들은 1로, object 토큰들은 2로 표시했다.
    
    예시)
    
    ```
    {
        'input_ids': tensor([[    0, 24380, 12242, 12951,  2386,  2189,     2, 11214,     2, 24380,
             12242, 12951,  2386,  2189,  2259, 11214,  5993,  1761,  2194,  4443,
              2079, 19230,  2628, 27135,  4713,  2138,  3670,  2205,  2507,  2062,
                 2]]), 
        'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0]]), 
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1]]),
        'entity_loc_ids': tensor([[0, 1, 1, 1, 1, 1, 0, 2, 0, 1, 1, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    }
    ```
    
2. Custom embedding layer 추가를 활용한 Custom model 제작
  
    RobertaForSequenceClassification 모델을 Roberta model과 Classifier로 분리한 후,
    
    Roberta model 내부에 커스텀 임베딩을 추가하였고,
    
    커스텀 임베딩 내부에 entity location embedding을 추가했다.
    
- 실험 결과

| entity embedding | micro f1 score | auprc |
| --- | --- | --- |
| X | 69.8393 | 72.7379 |
| O | **71.0866** | **76.2210** |


### Focal Loss
라벨의 불균형을 해결하기 위해 도입하였다.
- 실험 결과
![Untitled (2)](https://github.com/boostcampaitech5/level2_klue-nlp-07/assets/86578246/560c6e98-00e2-49ed-9c5a-8112aa9ea2d6)

### Learning Rate Scheduler
3k step 가량에서 보통 최고 성능이 나오고 그 뒤로는 학습이 안정적으로 진행이 되지 않기 때문에 도입하였다.
- 실험 결과
<img width="615" alt="image" src="https://github.com/boostcampaitech5/level2_klue-nlp-07/assets/86578246/81e4d364-16be-42e4-aefd-0798c8b85d1a">

| scheduler | micro f1 score | auprc |
| --- | --- | --- |
| linear | 71.0407 | 76.9576 |
| exponential | **71.3977** | **76.4372** |

### Label Constraints
모델이 텍스트 문맥이 아닌, 엔티티 간의 attention으로 관계를 유추한다. → 엔티티 편향

이를 해결하기 위해 [CoRE](https://arxiv.org/pdf/2205.03784.pdf) 논문에 사용된 기법들을 적용하였다.

CoRE 논문에서 활용한 주요한 편향 해결 방법은 아래와 같다.

```python
new_preds = (prob + lamb_1 * prob_mask_1 + lamb_2 * prob_mask_2 + label_constraint).argmax(1)
```

<img width="395" alt="Untitled (3)" src="https://github.com/boostcampaitech5/level2_klue-nlp-07/assets/86578246/a7741532-1ed2-4719-b0ef-5cff5047cdb0">


여기서의 metric function은 micro f1 score, [a,b]는 [-2, 2]를 사용했다.

> prob : 모델에서 나온 확률</br>
prob_mask_1 : 엔티티로만 추측한 확률</br>
prob_mask_2 : no_relation의 확률만 1로 두고 나머진 0으로 패딩한 확률</br>
label_constraint : 엔티티 type 별로 가질 수 있는 관계를 제한해둔 리스트</br>

| Label constraint | micro f1 score | auprc |
| --- | --- | --- |
| X | 73.9659 | 77.8363 |
| O | 74.2457 | 78.0171 |

### Task Adaptive Pretraining(TAPT)
[Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/pdf/2004.10964.pdf) 논문을 참고하여 대회 데이터셋에 사전 학습 모델을 적응시키면 파인튜닝 시 예측을 더 잘할 것이라는 예측 하에 도입한 기법이다.

| 모델 | perplexity | RE f1 | RE auprc |
| --- | --- | --- | --- |
| klue/roberta-large | 331.737 | 71.1024 | 74.2235 |
| klue/roberta-large + TAPT | **4.169** | **72.6778** | 72.7815 |
| studio-ousia/mluke-large | 4.089 | 85.447 | - |
| studio-ousia/mluke-large + TAPT | 3.588 | 85.709 | - |

### LSTM 레이어를 분류기로 활용
Dense layer와 tanh 레이어를 사용하는 기존의 classificationhead에 비해 LSTM의 장기 의존성, 단기 메모리는 시퀀스 분류 성능을 높여줄 것이라는 가정 하에 도입한 기법이다.
- 구현 방법
    1. RobertaForSequenceClassification 모델을 RobertaModel과 Classifier로 분리
    2. Classifier를 LSTM layer를 추가한 분류기로 대체 → CustomLSTMClassificationHead
    3. LSTM을 양방향 context를 파악할 수 있는 bi-LSTM으로 바꾸며 분류 실험 진행
    
![LSTM](https://github.com/boostcampaitech5/level2_klue-nlp-07/assets/86578246/fce5b72e-c80b-4fb4-b2b5-938ed476c0e3)

- 실험 결과

| Label constraint | micro f1 score | auprc |
| --- | --- | --- |
| 기존 | 71.3977 | 76.4372 |
| LSTM | **71.8922** | 75.4229 |
| bi-LSTM | 71.7605 | **76.4502** |

### Subject Entity Type 반영
EDA과정에서 30개의 label이 Subject entity type(ORG,PER)에 따라 결정됨을 확인했다. 여기서 모델이 Subject entity type에 맞지 않는 label로 예측함을 발견하여, 이를 해결하기 위해 도입하였다.

결과 Prob값에서 Subject entity의 type이 아닌 label로 예측한 Prob들을 모두 0으로 바꾼 뒤 다시 총 합이 1이 되도록 보정하였다.

- 실험 결과

| Subject entity type 반영 | micro f1 score | auprc |
| --- | --- | --- |
| X | 73.9659 | 77.8363 |
| O | **74.2457** | **78.0171** |

## 📌 Final Model
- 단일 모델로 제일 성능이 높게 나온 것은 아래 옵션을 적용한 **klue/roberta-large** 였다.
    - 손실함수:  focal loss
    - lr scheduler : exponential
    - LSTM 레이어 추가
    - 학습률 : 1e-5
    - 배치 사이즈: 32
    - Entity Embeding layer 추가
    - TAPT
    - label constraint
- 편향을 최소화해주기 위하여 다음 8가지 결괏값에 **레이블값 결정 확률을 모두 더하고 이를 평균해서 이들 중 확률이 가장 높은 레이블 값을 최종 보팅 결괏값으로 선정하는 소프트 보팅 앙상블**을 적용하여 최종 제출했다.
    - klue/roberta-large + focal loss + exponential lr scheduler + LSTM 레이어 + Entity Embeding layer + label constraint
    - klue/roberta-large + focal loss + exponential lr scheduler + Bi-LSTM 레이어 + Entity Embeding layer
    - klue/roberta-large + focal loss + exponential lr scheduler + LSTM 레이어 + Entity Embeding layer
    - klue/roberta-large + TAPT + focal loss + exponential lr scheduler + embedding layer
    - klue/roberta-large + TAPT + cross entropy
    - klue/roberta-large + focal loss + exponential lr scheduler + Entity Embeding layer + 데이터 증강
    - xlm-roberta-large + focal loss + exponential lr scheduler + LSTM 레이어 + Entity Embeding layer
    - studio-ousia/mluke-large

|  | public | private |
| --- | --- | --- |
| micro-f1 | 75.2196 | 74.7872 |
| auprc | 80.8363 | 82.8973 |
| 순위 | 8 | 4 |

![Untitled (4)](https://github.com/boostcampaitech5/level2_klue-nlp-07/assets/86578246/48675f58-9ece-423c-9ba6-7a16921d73af)
