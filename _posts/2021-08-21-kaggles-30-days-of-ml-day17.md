---
layout: post
title: "Kaggle's 30 Days of ML - Day17"
tags: Kaggle 機械学習
---

Kaggleの初心者向けプログラム「30 Days of ML」に挑戦中。  

## Day15以降の課題

- 指定のCompetitionへ、1日1回Submitする  
(本プログラム参加者限定コンペ、リンク共有不可)

## Competitionの概要

+ 保険のクレームの量を予測する

## 今日やったこと

+ [Dockerを使ってKaggleの環境を構築してみた](./build_kaggle_env_on_docker.html)
+ LGBMを使ったやり方のパラメータを調整してみた
    + 精度は上げられなかった

## 覚えておきたいこと

+ [勾配ブースティングで大事なパラメータの気持ち](https://nykergoto.hatenablog.jp/entry/2019/03/29/%E5%8B%BE%E9%85%8D%E3%83%96%E3%83%BC%E3%82%B9%E3%83%86%E3%82%A3%E3%83%B3%E3%82%B0%E3%81%A7%E5%A4%A7%E4%BA%8B%E3%81%AA%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E3%81%AE%E6%B0%97%E6%8C%81%E3%81%A1)
+ [LightGBMの解説](https://data-analysis-stats.jp/python/lightgbm%E3%81%AE%E8%A7%A3%E8%AA%AC/)

## メモ

+ `Optuna`を使うとハイパーパラメータを最適化できるらしいので、試してみたい
    + [ハイパーパラメータを最適化するOptunaの使い方](https://qiita.com/a_shiba/items/89cfcc012edd5bdc95dc)
    + [Python: Optuna で機械学習モデルのハイパーパラメータを選ぶ](https://blog.amedama.jp/entry/2018/12/06/015217)
+ LGBMRegressorのドキュメントは[こちら](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)