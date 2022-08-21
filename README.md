# Mediapipe_fallDetection

MediaPipeで姿勢推定をし、立っている状態(standing)と転んでいる状態(fall)を推定します。

はじめにMediapipeで姿勢推定をし、各アクションのキーポイントを保存します。

LSTMを用いて前30フレームの姿勢推定のキーポイントからstandingとfallのアクションを学習します。


# データセット

・KFALL Dataset
・UR Fall Detection Dataset

# DEMO

姿勢、アクションの推定結果

![tes1](https://user-images.githubusercontent.com/93971055/185788659-7b7a1b0f-0eb3-4274-bb27-20b7ea270938.gif)

![tes0](https://user-images.githubusercontent.com/93971055/185788664-6d0883e1-7e56-4309-8bd7-261e60c97047.gif)



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
