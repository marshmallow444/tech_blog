---
layout: post
title: "Kaggle's 30 Days of ML - Day12"
tags: Kaggle 機械学習
---

Kaggleの初心者向けプログラム「30 Days of ML」に挑戦中。  

## Day12の課題

1. the Intermediate ML Courseの[Lesson 1のチュートリアル](https://www.kaggle.com/alexisbcook/introduction)を読む  
1. the Intermediate ML Courseの[Lesson 1のexercise](https://www.kaggle.com/kernels/fork/3370272)を実施する  
1. the Intermediate ML Courseの[Lesson 2のチュートリアル](https://www.kaggle.com/alexisbcook/missing-values)を読む  
1. the Intermediate ML Courseの[Lesson 2のexercise](https://www.kaggle.com/kernels/fork/3370280)を実施する  
1. the Intermediate ML Courseの[Lesson 3のチュートリアル](https://www.kaggle.com/alexisbcook/categorical-variables)を読む  
1. the Intermediate ML Courseの[Lesson 3のexercise](https://www.kaggle.com/kernels/fork/3370279)を実施する  

## the Intermediate ML CourseのLesson 1の内容

+ Introduction
    + `RandomForestRegressor`を使って、家の価格の予測をしてみる(Intro to ML Courseのおさらい的な感じ)

## the Intermediate ML CourseのLesson 2の内容

+ 欠損値

#### 覚えておきたいと思った点

+ 欠損値の扱い方
    + 欠損値を含む列を削除する
    + 適当な値で置換する
    + 置換する + 「値が欠損していたかどうか」の列を追加する
        + 非常に有効な場合もあれば、全く意味がない場合もある
    + 列を削除すると情報が失われるので、精度が落ちがち
        + 欠損値の数が20%以内の場合は削除しない方が良いことが多い
    + `Imputer`の`fit_transform()`と`transform()`の違いについては[こちら](https://qiita.com/makopo/items/35c103e2df2e282f839a)を参照

## the Intermediate ML CourseのLesson 3の内容

+ カテゴリ変数

#### 覚えておきたいと思った点

+ カテゴリ変数の扱い方
    + カテゴリ変数を除去する
        + カテゴリ変数が有益な情報でない場合に有効
    + 整数に変換する(Ordinal Encoding)
        + `OrdinalEncoder`を使う
    + One Hot Encodingを行う
        + `OneHotEncoder`を使う
            + 今回は`handle_unknown='ignore'`, `sparse=False`にする
        + カテゴリに順番がない場合に向いている
        + カテゴリ数が多くない場合(だいたい15以下)に向いている
        + **cardinality**: あるカテゴリにおける値の数
+ One-hot encodingの手順
    1. `OneHotEncoder`を生成し、カテゴリデータの各列に適用する
    1. 1.でindexが削除されてしまうので、それを元に戻す
    1. カテゴリデータの入った列を削除する
    1. One-hot encodingされた列を加える