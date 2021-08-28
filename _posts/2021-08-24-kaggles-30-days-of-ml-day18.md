---
layout: post
title: "Kaggle's 30 Days of ML - Day18"
tags: Kaggle 機械学習
---

Kaggleの初心者向けプログラム「30 Days of ML」に挑戦中。  

## Day15以降の課題

- 指定のCompetitionへ、1日1回Submitする  
(本プログラム参加者限定コンペ、リンク共有不可)

## Competitionの概要

+ 保険のクレームの量を予測する

## 今日やったこと

+ LightGBM Tunerを使ってハイパーパラメータを最適化してみる  
    + LightGBM Tuner: LightGBM に特化したハイパーパラメータ自動最適化のためのモジュール。7個のパラメータにおいて、最適な値の組み合わせを探索

## 覚えておきたいこと

+ LightGBM Tunerを使うときは、LightGBMのimport行を以下のように変更する
    ```
    import optuna.integration.lightgbm as lgb
    ```

## メモ

+ LightGBM Tunerを使ってはみたが、前回LGBMを使ったときより精度は落ちてしまった
+ LightGBM Tunerの処理はかなり時間がかかる

## 参考

+ [Kaggler がよく使う「LightGBM」とは？【機械学習】](https://rightcode.co.jp/blog/information-technology/lightgbm-useful-for-kaggler)
+ [【Python覚書】LightGBMで交差検証を実装してみる ](https://potesara-tips.com/lightgbm-k-fold-cross-validation/#toc4)
+ [Optuna の拡張機能 LightGBM Tuner によるハイパーパラメータ自動最適化](https://tech.preferred.jp/ja/blog/hyperparameter-tuning-with-optuna-integration-lightgbm-tuner/)
+ [LightGBM Tunerを用いたハイパーパラメーターのチューニング](https://qiita.com/askl4267/items/28b476f76b01699430fe)
+ [LightGBMの「No further splits with positive gain」というwarningの意味](https://nigimitama.hatenablog.jp/entry/2021/01/05/205741)
+ [LightGBM - Parameters Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)