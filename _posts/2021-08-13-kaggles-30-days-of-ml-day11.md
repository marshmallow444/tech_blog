---
layout: post
title: "Kaggle's 30 Days of ML - Day11"
tags: Kaggle 機械学習
---

Kaggleの初心者向けプログラム「30 Days of ML」に挑戦中。  

## Day11の課題

1. the Intro to ML Courseの[Lesson 7のチュートリアル](https://www.kaggle.com/alexisbcook/machine-learning-competitions)を読む  
1. the Intro to ML Courseの[Lesson 7のexercise](https://www.kaggle.com/kernels/fork/1259198)を実施する  

## the Intro to ML CourseのLesson 7の内容

+ Competitionに参加する

#### 覚えておきたいと思った点

+ Competitionへの参加の仕方
    1. 用意された訓練用データ(`train.csv`など)を使って学習する
    1. 用意されたテスト用データ(`test.csv`など)を使って予測する
    1. 予測結果をcsvファイルへ書き出す
    1. 書き出した予測結果のファイルを提出する
+ 予測結果ファイルの提出の仕方
    1. Competitionのページで`Join Competition`ボタンをクリックしておく
    1. Notebookの`Save Version`ボタンをクリックする
    1. `Save and Run All`オプションを選択し、`Save`ボタンをクリックする
    1. Notebook右上の`Save Version`ボタンの右にある、数字のボタンをクリックする
    1. `Version History`にて、提出したいバージョンの右側にある`...`をクリックする
    1. `Submit to Competition` > `Submit`を選択する
    1. もしくは、`Open in Viewer` > `Output`タブ(画面右側) > `Submit` > `Submit`を選択する
+ featuresを増やした時に`NaN`が含まれる箇所は`fillna()`関数で置換することで対応できた