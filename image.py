
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import optimizers
from keras.callbacks import Callback
import matplotlib.pyplot as plt


train_data_path = 'C:\\Users\\6d09\\Desktop\\dog or cat\\detaset\\train'#トレーニングデータの保存場所を記述
test_data_path = 'C:\\Users\\6d09\\Desktop\\dog or cat\\detaset\\test'#テストデータの保存場所を記述

#学習データの作成（ニューラルネットワークの入力として適切な形式の画像データが生成）
train_datagen = ImageDataGenerator(rescale=1/255)#画像のピクセルを0～255の範囲から0～1の範囲に変換するスケーリング
#評価データの作成
test_datagen = ImageDataGenerator(rescale=1/255)

#学習画像(generator)を生成する
train_generator = train_datagen.flow_from_directory(#画像を読みとる
                    directory=train_data_path,#ディレクトリにパスを渡す
                    target_size=(150,150),#処理時の画像サイズ
                    batch_size=32,#バッチサイズ（１ステップで学習する画像の枚数）一般的に一般的に使われる中程度の値
                    
                    class_mode='binary'#猫と犬の２種類なのでバイナリー
                    )

test_generator = test_datagen.flow_from_directory(
                    directory=train_data_path,
                    target_size=(150,150),
                    batch_size=32,
                    class_mode='binary'
                    )
print(train_generator.class_indices)#dogsとcatsというクラスに分けられる


model = Sequential()#シーケンシャルモデルの定義（順番に.addでモデルの追加ができる）
#層を組み合わせて複数の階層からなるネットワークを構築し、画像の特徴を階層的に抽出して分類や認識を行う

model.add(Conv2D(#畳み込み層。元の画像にフィルタをかけて特徴を抽出する
                 filters=64,kernel_size=(5,5), #5×5のフィルタを64枚使って畳み込みする
                 input_shape=(150, 150,3)))#縦150横150ピクセルのカラー画像
model.add(Activation('relu'))#Relu関数（活性化関数）を使用してモデルの表現力を増す。
model.add(MaxPooling2D(pool_size=(2,2)))#プーリング層。畳み込み層のデータを集約して量を減らす

model.add(Conv2D(filters=128, kernel_size=(5,5)))#２回目の畳み込み層
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(5,5)))#３回目の畳み込み層
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#過学習の防止、学習時にランダムにユニットを無効化しモデルのパターン化を防ぐ
model.add(Dropout(0.5))

model.add(Flatten())#入力を１次元配列に変換し全結合層に適した形式にデータを変換
model.add(Dense(256))#全結合層（ユニット数）
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))#シグモイド関数（ｓ字のシグモイド曲線を描く。入力を０か１に近い確率値に変換）

sgd = optimizers.SGD(lr=0.1)#確率的勾配降下法（学習率）ランダムなデータで計算
#モデルをコンパイル（損失関数、最適化関数、評価関数）
model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['accuracy'])
#コンパイルしたモデルの学習開始（大量にあるデータを学習させる為fit_generatorを使用）

class AccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("\nAccuracy: {:.4f}".format(logs['accuracy']))
        print("Validation Accuracy: {:.4f}".format(logs['val_accuracy']))

# コールバックのインスタンスを作成
accuracy_callback = AccuracyCallback()

# model.fit_generatorのcallbacks引数に追加
history = model.fit_generator(train_generator,#訓練用データのジェネレーター。画像データとラベルのバッチを生成するために使用
                              epochs=1,#訓練するエポック数を指定します
                              verbose=1,#訓練の進捗状況を表示するかどうかを指定します
                              validation_data=test_generator,#モデルが過学習していないかを確認するためデータジェネレーター。
                              steps_per_epoch=4000/32,#1エポック(125回）ごとの訓練ステップ数を指定。4000/32は、訓練データの総数をバッチサイズで割った値
                              validation_steps=4000/32,#1エポックごとの検証ステップ数を指定。4000/32は、訓練データの総数をバッチサイズで割った値
                              callbacks=[accuracy_callback])#訓練中の精度を表示するためのコールバック関数です。

#モデルをmodel.h5に保存
model.save('model.h5')

loss = history.history['loss']#学習用データの正解と予測の差分のグラフを作成（小さいほど正しい）
val_loss = history.history['val_loss']#検証用データの損失（小さいほど正解）

learning_count = len(loss) + 1

plt.plot(range(1, learning_count),loss,marker='+',label='loss')#学習用グラフを作成
plt.plot(range(1, learning_count),val_loss,marker='.',label='val_loss')#検証用データのグラフを作成
plt.legend(loc = 'best', fontsize=10)#凡例をグラフに被らないように表示
plt.xlabel('learning_count')#ｘ軸のラベル（学習回数）
plt.ylabel('loss')#ｙ軸のラベル（損失関数）
plt.show()#画面に表示

accuracy = history.history['accuracy']#学習用データの正しく予測できた画像の割合
val_accuracy = history.history['val_accuracy']#検証用データの正しく予測できた画像の割合

plt.plot(range(1, learning_count),accuracy,marker='+',label='accuracy')#学習用のグラフを作成
plt.plot(range(1, learning_count),val_accuracy,marker='.',label='val_accuracy')#検証用のグラフを作成
plt.legend(loc = 'best', fontsize=10)#凡例をグラフに被らないように表示
plt.xlabel('learning_count')#ｘ軸のラベル（学習回数）
plt.ylabel('accuracy')#ｙ軸のラベル（予測正解率）
plt.show()#画面に表示


