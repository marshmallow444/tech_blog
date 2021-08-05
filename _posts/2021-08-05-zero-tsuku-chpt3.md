---
layout: post
title: "【ゼロから作るDeepLearning】Google Colabでコードを動かす "
tags: 機械学習 ゼロつく
---

「ゼロから作るDeepLearning」のサンプルコードをGoogle Colaboratoryで動かしてみている。  
うまく動かなかったところをメモしておく。

### 3.6.1 MNISTデータセット
#### p.73  
【問題点】
MNISTデータのダウンロードが出来ない

【原因】
データのあるサーバからエラーが返されているらしい [^1]  

【対策】
データをダウンロードする代わりに、keras等からMNISTデータを取得する [^2]

```
from keras.datasets import mnist
(x_train, t_train), (x_test, t_test) = mnist.load_data()

# それぞれのデータの形状を出力
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
```

【備考】 

+ Google Drive上の別ファイルをインポートするには、Google Driveをマウントする必要がある
```
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```
上記コードを実行後、ブラウザ上で認証手続きを行うとマウントできる  
インポートは以下の要領で行う  
```
import sys
sys.path.append('/content/drive/My Drive/ColabNotebooks/path/to/dataset/')
from mnist import load_mnist
 ```

+ `x_train`や`x_test`のサイズに注意
    + サンプルコードでは768個の要素からなる1次元配列として取得されるが、kerasで取得したデータは28*28の2次元配列のまま取得される


#### p.74〜75

【問題点】  
MNIST画像の表示ができない

【原因】  
Google Colaboratory上だと`PIL`の`Image`の`show()`がうまく動かない

【対策】  
代わりに`IPython.display()`を使う [^3]

```
from PIL import Image
from IPython.display import display

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    display(pil_img)
```

[^1]: https://stackoverflow.com/questions/66577151/http-error-when-trying-to-download-mnist-data  
[^2]: https://colab.research.google.com/drive/1xckYBNOaRYojHrJVy6-D8O57bcjW8v4P#scrollTo=jnMMc6ivvI_a  
[^3]: https://qiita.com/kaityo256/items/ce34f412ceec1b72755d