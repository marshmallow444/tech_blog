---
layout: post
title: "Kaggle's 30 Days of ML - Day5"
tags: Kaggle 機械学習
---

Kaggleの初心者向けプログラム「30 Days of ML」に挑戦中。  

## Day5の課題

1. Python Courseの[Lesson 4のチュートリアル](https://www.kaggle.com/colinmorris/lists)を読む
1. Python Courseの[Lesson 4のexercise](https://www.kaggle.com/kernels/fork/1275173)を実施する
1. Python Courseの[Lesson 5のチュートリアル](https://www.kaggle.com/colinmorris/loops-and-list-comprehensions)を読む
1. Python Courseの[Lesson 5のexercise](https://www.kaggle.com/kernels/fork/1275177)を実施する

## Python CourseのLesson 4の内容

+ Lists   
+ Tuples

#### 覚えておきたいと思った点

+ 一つのListに異なる型の要素を入れることが可能
+ Listの要素にアクセスする際、インデックスに`-x`を指定すると、リストの最後から`x`番目の要素にアクセスできる
+ `リスト[a:b]`のように書くことで、`a`番目から`b`番目の要素のリストを取得できる
    + `a`を省けば最初から`b`番目まで
    + `b`を省けば`a`番目から最後まで
    + インデックスに負の値を入れても良い
+ Pythonに出てくる全てのものはobject
+ `要素 in リスト`で、リスト内に要素が含まれるか確認できる
+ TupleとListの違い
    + Tupleは`( )`または括弧なし, Listは`[ ]`で囲む
    + Tupleはimmutable, Listはmutable
+ 返り値が複数ある場合、それを受け取る変数をTupleにしておくと、それぞれの値をひとつずつの変数に格納できる
    + 例：`numerator, denominator = x.as_integer_ratio()`
+ Listを内包するListの`len()`を使うと、内包されるListが一つの要素として数えられる

## Python CourseのLesson 5の内容

+ Loops   
+ List Comprehensions

#### 覚えておきたいと思った点

+ for文は`for 要素 in リスト等`のように書く
+ `[num for num in nums if num < 0]`みたいな書き方もできる

## メモ

+ 問題文にまさかのマリオカート登場。びっくりした。。(Lesson 4)