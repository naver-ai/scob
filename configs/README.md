# Config files

## 1. 개요

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config=configs/textread_test.yaml --train.max_epochs=100 --train.optimizer.lr_schedule.params.warmup_steps=10000
```

위와 같이 명령어를 작성하고 실행할 경우, 아래 4단계의 과정을 거쳐서 파일/command line에 작성한 설정 값들이 적용됩니다. 
아래 내용들에 대한 코드는 `utils/config_manage.py:76`의 `__load()` method에 작성되어 있습니다.

#### 1) `configs/default.yaml`에 작성되어 있는 값 불러오기
  - 기본적으로 `configs/default.yaml` 파일에 작성되어 있는 값을 불러옵니다.
  - `configs/default.yaml`에는 (코드를 통해 생성되는 key를 제외하고는) 등장할 수 있는 모든 key에 대한 값들이 작성되어 있습니다.
  - 현재 `configs/default.yaml`에 작성되어 있는 값들은, 실제 사용을 위해서 작성되어있기보다는, 보고 참고하시라는 용도로 다양한 설정들의 조합이 작성되어 있습니다 (e.g. `loss_func` 등을 다양하게 사용함). 따라서 어떤 key에 대한 사용 예를 찾고자 한다면 `configs/default.yaml`에서 찾으시면 됩니다.

#### 2) `train.py` 에서 `ConfigManage()` class 생성시 argument로 전달되는 config file에 작성되어 있는 값 불러오기
  - 이 부분은 기능 구현은 되어있지만, `train.py:19`을 보시면 알 수 있듯이 특별히 넘겨주고 있는 config file이 없습니다.
  - 따라서 무시하고 넘어가셔도 됩니다.

### 3) 명령어에서 argument로 전달한 항목들 중, `--config`로 전달된 config file에 작성되어 있는 값 불러오기
  - 이 부분은 위의 명령어 예시에서 `--config=configs/finetune/local_finetune.yaml` 부분을 의미합니다.
  - 세번째 단계에서 `configs/finetune/local_finetune.yaml`에 작성되어 있는 값을 불러옵니다 (덮어쓰기).

### 4) 명령어에서 argument로 전달한 항목들 중, `--config`를 제외한 나머지 값들 불러오기
  - 이 부분은 위의 명령어 예시에서 `--train.max_epochs=100 --train.optimizer.lr_schedule.params.warmup_steps=10000` 부분을 의미합니다.
  - 각각의 항목들을 불러옵니다 (덮어쓰기).

## 2. 특징
  - `configs/default.yaml`에는 (코드를 통해 생성되는 key를 제외하고는) 등장할 수 있는 모든 key에 대한 값들이 작성되어 있습니다.
  - 현재 `configs/default.yaml`에 작성되어 있는 값들은, 실제 사용을 위해서 작성되어있기보다는, 보고 참고하시라는 용도로 다양한 설정들의 조합이 작성되어 있습니다 (e.g. `loss_func` 등을 다양하게 사용함). 따라서 어떤 key에 대한 사용 예를 찾고자 한다면 `configs/default.yaml`에서 찾으시면 됩니다.

