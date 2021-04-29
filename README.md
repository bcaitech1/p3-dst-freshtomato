# 💋Dialogue State Tracking
## Task Description
- Task: 주어진 목적 지향형 대화의 Dialogue State를 예측합니다.
- Metric: Joint Goal Accuracy, Slot Accuracy, Slot F1 Score

## Command Line Interface
### Train Phase
```python
>>> python code/train.py --project_name [PROJECT_NAME]
```

### Inference Phase
```python
SM_CHANNEL_EVAL=data/eval_dataset/public SM_CHANNEL_MODEL=[Model Checkpoint Path] SM_OUTPUT_DATA_DIR=[Output path] python inference.py
```

### Application: wandb
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

## TEAM FreshTomato🍅
- [고지형](https://github.com/iloveslowfood), [김진현](https://github.com/KimJinHye0n), [배아라](https://github.com/arabae), [손재훈](https://github.com/MoonJaeHoon), [최유라](https://github.com/Yuuraa), [최준구](https://github.com/soupbab)