---
layout: post
title: "Kaggle's 30 Days of ML - Day7"
tags: Kaggle 機械学習
---

Kaggleの初心者向けプログラム「30 Days of ML」に挑戦中。  

## Day7の課題

1. Python Courseの[Lesson 7のチュートリアル](https://www.kaggle.com/colinmorris/working-with-external-libraries)を読む
1. Python Courseの[Lesson 7のexercise](https://www.kaggle.com/kernels/fork/1275190)を実施する

## Python CourseのLesson 7の内容

+ 外部ライブラリの使い方

#### 覚えておきたいと思った点

+ モジュールの関数を呼ぶ時に、モジュール名を省略する例：
    ```
    from math import *
    print(pi, log(32, 2))
    ```
+ 謎のオブジェクトを理解するのに役立つメソッド
    + `type()`
        + オブジェクトの型を調べる
    + `dir()`
        + そのオブジェクトで何ができるのか調べる(変数、関数など)
    + `help()`
+ 演算子のオーバーロード
    + オブジェクトの演算子がオーバーロードされている場合があり、予想と違う動作をする場合があるので要注意

## メモ

+ この課題はヘビーな問題が2つ出て、予想より時間がかかってしまった。でも楽しかった