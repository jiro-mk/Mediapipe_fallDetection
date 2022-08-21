# Mediapipe_fallDetection

MediaPipeで姿勢推定をし、立っている状態(standing)と転んでいる状態(fall)を推定します。

はじめにMediapipeで姿勢推定をし、各アクションのキーポイントを保存します。

LSTMを用いて前30フレームの姿勢推定のキーポイントからstandingとfallのアクションを学習します。



# データセット

・KFALL Dataset

・UR Fall Detection Dataset



# DEMO

姿勢、アクションの推定結果

![tes1](https://user-images.githubusercontent.com/93971055/185789050-0bd1b8be-1a0c-47aa-ae07-a641634327db.gif)

![tes0](https://user-images.githubusercontent.com/93971055/185789072-58a3f6c6-42a9-4bf0-be2f-2c812902cae8.gif)




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
