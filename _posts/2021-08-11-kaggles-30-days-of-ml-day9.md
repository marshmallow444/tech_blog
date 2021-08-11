---
layout: post
title: "Kaggle's 30 Days of ML - Day9"
tags: Kaggle 機械学習
---

Kaggleの初心者向けプログラム「30 Days of ML」に挑戦中。  

## Day9の課題

1. the Intro to ML Courseの[Lesson 3のチュートリアル](https://www.kaggle.com/dansbecker/your-first-machine-learning-model)を読む  
1. the Intro to ML Courseの[Lesson 3のexercise](https://www.kaggle.com/kernels/fork/1404276)を実施する  
1. the Intro to ML Courseの[Lesson 4のチュートリアル](https://www.kaggle.com/dansbecker/model-validation)を読む  
1. the Intro to ML Courseの[Lesson 4のexercise](https://www.kaggle.com/kernels/fork/1259097)を実施する  

## the Intro to ML CourseのLesson 3の内容

+ モデリングのためのデータを選ぶ

#### 覚えておきたいと思った点

+ 予測の手順：
    1. 予測対象のデータを決める
    1. 予測に使うデータ(特徴)を決める
    1. モデルを構築する
+ Pandasでは、読み込んだcsvデータに対して以下の操作が可能
    + `data.columns`で列の見出しを取得
    + `data.列名`で列のデータを取得
    + `data[列見出しリスト]`で、指定した複数列のデータを取得
    + `data.head()`で先頭の5行を取得

## the Intro to ML CourseのLesson 4の内容

+ モデルを評価する

#### 覚えておきたいと思った点

+ `train_test_split()`でテストデータと検証データの分割ができる
+ `mean_absolute_error()`で平均絶対誤差を取得できる