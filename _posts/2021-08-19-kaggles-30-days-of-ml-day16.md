---
layout: post
title: "Kaggle's 30 Days of ML - Day16"
tags: Kaggle 機械学習
---

Kaggleの初心者向けプログラム「30 Days of ML」に挑戦中。  

## Day15以降の課題

- 指定のCompetitionへ、1日1回Submitする  
(本プログラム参加者限定コンペ、リンク共有不可)

## Competitionの概要

+ 保険のクレームの量を予測する

## 今日やったこと

+ 昨日作ったNotebookの改良
    + RandomForest→XGBoostに置き換えてみる
+ LGBMを使ってみる
+ [ローカルでKaggleのソースを動かしてみる](./run-kaggles-code-in-local-env.html)
    + KaggleのKernel上だと実行にかなり長い時間がかかったので、試してみた

## 覚えておきたいこと

+ 一度SaveしたVersionは削除できないらしい([参考](https://www.kaggle.com/getting-started/55855))

## メモ

+ Dockerを使って環境構築する方法も試してみたい
    + 参考：[爆速でKaggle環境を構築する](https://qiita.com/bam6o0/items/354faa9394755a984661)
+ [LightGBM 徹底入門 – LightGBMの使い方や仕組み、XGBoostとの違いについて](https://www.codexa.net/lightgbm-beginner/)