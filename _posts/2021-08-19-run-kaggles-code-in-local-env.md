---
layout: post
title: "KaggleのNotebookをローカルで編集する"
tags: Kaggle
---

KaggleのKernel上でコードを動かしてみたら実行に時間がかかったので、ローカルで実行してみた。  
手順を忘れないようメモしておく。  
一旦ソースをダウンロードしてローカルで編集し、完成したらアップロードしてSave&Submitする、という流れ。  

## 環境構築  

mac OS(10.15.7) + VSCodeを使用している

+ [VSCode](https://azure.microsoft.com/ja-jp/products/visual-studio-code/)と[jupyterのExtension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)をインストールしておく  
+ 必要なライブラリをインストールしておく  
(参考：[LightGBMをインストール](https://qiita.com/m__k/items/5f905cf5d20e875961b5))  

## ローカルでの実行手順

+ Kaggleから、編集したいNotebookをダウンロードする
    + Notebookの左上`File`メニュー→`Download notebook`を選択
+ (Competitionページの`Data`タブから、必要なデータもダウンロードしておく)
+ ダウンロードした`.ipynb`ファイルをVSCodeで開き、編集・実行する

## ソースのアップロード手順

+ 新規Notebookを作成する
    + 既存のNotebookへ上書きする場合は、そのNotebookを開く
+ 左上`File`メニュー→`Upload Notebook`を選択
    + <u>Uploadすると、Notebookの内容は全て上書きされてしまうので要注意</u>

アップロード後は、通常通りSaveとSubmitを行う。  

## 備考  

+ [ローカルのjupyterかkaggleのKernelかを判定する方法](https://www.currypurin.com/entry/2019/06/10/120225)を使うと、環境によって処理を変えられる
+ ローカルで動かすのにかかった時間は、Kaggle Kernel上で動かした時の半分程度になった
    + まだ一つのソースしか動かしていないので、他のソースでも同様の速度になるのかは不明