---
layout: post
title: "Kaggle's 30 Days of ML - Day13"
tags: Kaggle 機械学習
---

Kaggleの初心者向けプログラム「30 Days of ML」に挑戦中。  

## Day13の課題

1. the Intermediate ML Courseの[Lesson 4のチュートリアル](https://www.kaggle.com/alexisbcook/pipelines)を読む  
1. the Intermediate ML Courseの[Lesson 4のexercise](https://www.kaggle.com/kernels/fork/3370278)を実施する  
1. the Intermediate ML Courseの[Lesson 5のチュートリアル](https://www.kaggle.com/alexisbcook/cross-validation)を読む  
1. the Intermediate ML Courseの[Lesson 5のexercise](https://www.kaggle.com/kernels/fork/3370281)を実施する  

## the Intermediate ML CourseのLesson 5の内容

+ パイプライン

#### 覚えておきたいと思った点

+ パイプラインの長所
    + コードが整理される
    + バグが減る
    + 製品化しやすい
    + 交差検証法のオプションが多い
+ `Pipeline`を使う

## the Intermediate ML CourseのLesson 6の内容

+ 交差検証法

#### 覚えておきたいと思った点

+ Pipelineを使うことで交差検証法が簡単にできる
    + `cross_val_score()`を使う
        + 負のMAEの値が返ってくるので注意
+ foldごとのMAEを求めて、平均値を取る
