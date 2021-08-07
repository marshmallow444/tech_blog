---
layout: post
title: "Kaggle's 30 Days of ML - Day6"
tags: Kaggle 機械学習
---

Kaggleの初心者向けプログラム「30 Days of ML」に挑戦中。  

## Day6の課題

1. Python Courseの[Lesson 6のチュートリアル](https://www.kaggle.com/colinmorris/strings-and-dictionaries)を読む
1. Python Courseの[Lesson 6のexercise](https://www.kaggle.com/kernels/fork/1275185)を実施する

## Python CourseのLesson 6の内容

+ Strings  
+ Dictionaries

#### 覚えておきたいと思った点

+ Pythonでは
    + 文字列は`'`または`"`で囲む
        + 文中に`'`が含まれるときは`"`で囲むと便利
    + 文中に`'` `"` `\`を含める場合、前に`\`をつける
    + `\n`で改行可能
    + `"""`で囲むと、中の文が改行された位置で(`\n`をつけなくても)改行される
    + Listと同様に各文字へアクセスできるが、immutable
        + 文字列操作の結果は何かの変数へ入れる必要がある
    + 文字列中に数値を入れたいとき
        + 数値を`str()`で文字列に変換する
        + `"{}, you'll always be the {}th planet to me.".format(planet, position)`のように書く
    + 以下のように書くと、Listの各要素とindexが取れる
        ```
        for i, item in enumerate(item_list)
        ```
    + `str.rstrip()`で引数の文字を探して除去する
