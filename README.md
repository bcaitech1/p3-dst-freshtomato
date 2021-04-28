## Baseline model of BoostCamp2021 P-Stage DST


Open-vocab based DST model인 [TRADE](https://arxiv.org/abs/1905.08743)의 한국어 구현체입니다. (5강, 6강 내용 참고) <br>

- 기존의 GloVe, Char Embedding 대신 `monologg/koelectra-base-v3-discriminator`의 `token_embeddings`을pretrained Subword Embedding으로 사용합니다.
- 메모리를 아끼기 위해 Token Embedding (768) => Hidden Dimension (400)으로의 Projection layer가 들어 있습니다.
- 빠른 학습을 위해 `Parallel Decoding`이 구현되어 있습니다.


### 1. 필요한 라이브러리 설치

`pip install -r requirements.txt`

### 2. 모델 학습

`SM_CHANNEL_TRAIN=data/train_dataset SM_MODEL_DIR=[model saving dir] python train.py` <br>
학습된 모델은 epoch 별로 `SM_MODEL_DIR/model-{epoch}.bin` 으로 저장됩니다.<br>
추론에 필요한 부가 정보인 configuration들도 같은 경로에 저장됩니다.<br>
Best Checkpoint Path가 학습 마지막에 표기됩니다.<br>

### 3. 추론하기

`SM_CHANNEL_EVAL=data/eval_dataset/public SM_CHANNEL_MODEL=[Model Checkpoint Path] SM_OUTPUT_DATA_DIR=[Output path] python inference.py`

### 4. 제출하기

3번 스텝 `inference.py`에서 `SM_OUTPUT_DATA_DIR`에 저장된 `predictions.json`을 제출합니다.


### wandb 적용하기
1. train.py파일을 수행하면 다음과 같은 화면에서 2번 선택

  ![image](https://user-images.githubusercontent.com/46676700/116401727-89628d80-a866-11eb-9069-5c7a947741ab.png)


2. API key를 받을 수 있는 링크로 들어가 (그림 2번째 줄) 공유 계정으로 로그인

  ![image](https://user-images.githubusercontent.com/46676700/116401752-91223200-a866-11eb-80e7-78af8acb2049.png)

3. 아래와 같이 key값을 terminal 창에 복사 붙여 넣기


    <img src="https://user-images.githubusercontent.com/46676700/116401797-9f704e00-a866-11eb-91b3-1cb509c19c88.png" width="40%">

    - 다음과 같이 수행됨
  
    ![image](https://user-images.githubusercontent.com/46676700/116401807-a26b3e80-a866-11eb-93ee-7a7e0b510a8b.png)


4. wandb 홈페이지에서 다음과 같이 만들어진 project를 확인할 수 있음

  ![image](https://user-images.githubusercontent.com/46676700/116401826-a6975c00-a866-11eb-806b-21e6cc6c5492.png)

  ![image](https://user-images.githubusercontent.com/46676700/116401835-a9924c80-a866-11eb-9b67-a918fb258b52.png)
