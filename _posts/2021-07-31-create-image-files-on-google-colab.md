---
layout: post
title: "Google Colabで画像ファイルを書き出す"
---

Google ColaboratoryにてGoogle Drive上に画像を書き出してみたくなったので、そのやり方を調べてみた。  
なお ここで書き出す画像データは、kerasから取得したMNISTデータを使用している。  

## 画像ファイルを書き出す方法

1. ドライブのマウントを行う
    ```
    from google.colab import auth
    auth.authenticate_user()

    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from oauth2client.client import GoogleCredentials

    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    ```
1. 画像保存処理を行う関数を定義する
    ```
    from PIL import Image
    import IPython
    import sys
    import os

    file_name = 'test.jpg'
    dir_name = '/content/drive/My Drive/Colab Notebooks/path/to/the/folder/' # ←適宜書き換える

    sys.path.append(dir_name)
    os.chdir(dir_name) # ディレクトリを移動

    def img_save(img):
        pil_img = Image.fromarray(np.uint8(img))
        pil_img.save(file_name)
    
        #UPLOADする
        f = drive.CreateFile({'title': file_name, 'mimeType': 'image/png'})
        # f.SetContentFile(file_name)    # ←実行するとエラーに。なくても書き出せた
        f.Upload()
    ```
1. 上記関数を使って画像を書き出す
    ```
    from keras.datasets import mnist

    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    img  = x_train[0]
    label = t_train[0]

    img_save(img)
    ```

## 参考

+ [Python(Colab)でウェブ上の画像をドライブに保存する](https://qiita.com/plumfield56/items/c960d36f9224a68a4405)
+ [google colab で google driveを使う](https://skattun.hatenablog.jp/entry/2019/04/30/233526)
+ [Google Colaboratory でFileNotFoundErrorが出る。](https://teratail.com/questions/224318)