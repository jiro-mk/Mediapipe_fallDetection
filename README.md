# Mediapipe_fallDetection

MediaPipeで姿勢推定をし、立っている状態(standing)と転んでいる状態(fall)を推定します。

はじめにMediapipeで姿勢推定をし、各アクションのキーポイントを保存します。

LSTMを用いて前30フレームの姿勢推定のキーポイントからstandingとfallのアクションを学習します。


# データセット

・KFALL Dataset
・UR Fall Detection Dataset

# DEMO

静止画を用いた姿勢推定結果

![image](https://user-images.githubusercontent.com/93971055/155551785-a3e5e396-b629-4d05-902c-a0b31d0592df.png)


![image](https://user-images.githubusercontent.com/93971055/155551886-d6efec69-b342-482b-8111-0cee91959efe.png)



# Requirement

mediapipe                    0.8.9.1

opencv-python                4.5.5.62   

Tensorflow                   2.3.0 or Later

tf-nightly                   2.5.0.dev or later 



# Usage


```bash
git clone git@github.com:jiro-mk/Mediapipe_fallDetection.git
cd Mediapipe_fallDetection

#学習データ作成 
python app1.py 

# LSTMモデルの学習
python trainLSTM.py

# テストの実行
python infer.py

```


# Author

jiro-mk
