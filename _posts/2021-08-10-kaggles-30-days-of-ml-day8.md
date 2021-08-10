---
layout: post
title: "Kaggle's 30 Days of ML - Day8"
tags: Kaggle 機械学習
---

Kaggleの初心者向けプログラム「30 Days of ML」に挑戦中。  

## Day8の課題

1. the Intro to ML Courseの[Lesson 1のチュートリアル](https://www.kaggle.com/dansbecker/how-models-work)を読む  
1. the Intro to ML Courseの[Lesson 2のチュートリアル](https://www.kaggle.com/dansbecker/basic-data-exploration)を読む  
1. the Intro to ML Courseの[Lesson 2のexercise](https://www.kaggle.com/kernels/fork/1258954)を実施する  

## the Intro to ML CourseのLesson 1の内容

+ How Models Work
    + 決定木を使って不動産の価格を予想することを考える

## the Intro to ML CourseのLesson 2の内容

+ Basic Data Exploration
    + Pandasを使ってデータの概要を見てみる

#### 覚えておきたいと思った点

+ 以下の要領でcsvファイルを読み込める
    ```
    data = pd.read_csv()
    ```
+ 上記で読み込んだデータの概要を、以下のようにして表示できる  
    ```
    data.describe()
    ```
+ データの中身に問題がないか、予めチェックしておく必要がある