---
layout: post
title: "Kaggle's 30 Days of ML - Day14"
tags: Kaggle 機械学習
---

Kaggleの初心者向けプログラム「30 Days of ML」に挑戦中。  

## Day14の課題

1. the Intermediate ML Courseの[Lesson 6のチュートリアル](https://www.kaggle.com/alexisbcook/xgboost)を読む  
1. the Intermediate ML Courseの[Lesson 6のexercise](https://www.kaggle.com/kernels/fork/3370271)を実施する  
1. the Intermediate ML Courseの[Lesson 7のチュートリアル](https://www.kaggle.com/alexisbcook/data-leakage)を読む  
1. the Intermediate ML Courseの[Lesson 7のexercise](https://www.kaggle.com/kernels/fork/3370270)を実施する  

## the Intermediate ML CourseのLesson 6の内容

+ XGBoost(= eXtreme Gradient Boosting)

#### 覚えておきたいと思った点

+ `XGBRegressor`を使う
    + [Early Stopping](https://note.com/okonomiyaki011/n/n371ef12f40e0)もできる
    + `n_jobs`で使用するコア数を指定できる
        + データが多い場合に有効
+ パラメータのチューニング
    + `n_estimators`: 100~1000くらい
    + `early_stopping_rounds`: 5くらい？
    + `learning_rate`: デフォルトは0.1

## the Intermediate ML CourseのLesson 7の内容

+ data leakage

#### 覚えておきたいと思った点

+ **Target Leakage**
    + 予測を行う時点で知り得ないデータが混入
+ **Train-Test Contamination** (Contamination = 汚染)
    + 訓練データとテストデータを分割する前に前処理をしてしまうことにより、精度が落ちる
    + 検証データやテストデータを学習に使用してしまうことにより、ある特定のデータに対してのみ精度が上がり、未知のデータに対する精度が落ちる
