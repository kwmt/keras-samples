## Kerasをインストールするには

バックエンドエンジンとしてTensorFlowをインストール
https://keras.io/#installation
```
pip install tensorflow 
```

Kerasをインストール

```
pip install keras
```


## Kerasの簡単な説明

Sequentialモデルを使って層を積んでいくことができる。

```
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
```

addメソッドで追加することもできる

```
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
```

## サンプルを触ってみる

ゼロから作るDeep LearningでやったのMNISTのデータセットからのクラス分類を、Kerasを使ってやってみる。
そのサンプルがある
https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py


`model.summary()`は次のような情報を表示してくれる

```
2020-03-22 00:39:49.724183: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-22 00:39:49.742672: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fee75cec610 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-03-22 00:39:49.742691: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 512)               401920    
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656    
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5130      
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
```

`model.fit`はモデルを訓練します。
https://keras.io/ja/models/sequential/#fit
```
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
```