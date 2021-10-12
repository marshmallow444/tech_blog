---
layout: post
title: "ローカルでDocker+VSCodeを使ったKaggleの環境構築メモ"
tags: Kaggle
---

以前「[Dockerを使ってKaggleの環境を構築する](../../08/21/build_kaggle_env_on_docker.html)」の手順で構築した環境にて  
VSCodeを使う方法がわからなかったので、新たに環境を作ってみた。  
手っ取り早く環境を構築したい場合は、上記記事の手順の方が手軽に試せるかも？  

<br>

この記事の内容に沿って環境構築した：  
[公式Dockerとvscodeを使ったKaggleの環境構築](https://qiita.com/Artela_0000/items/4b0f3c02b1e9e1b2695b)  

### 詰まった箇所

- dockerのビルド時
    - `--gpu` でエラー
        - macでは無理みたい
    - ストレージ不足？でエラー
        - `docker system prune` をしたら解決
    - メモリ不足でエラー
        - dockerの設定でメモリを2GBから4GBにしたら解決
        - 最低4GBは必要らしい ([参考](https://github.com/facebook/prophet/issues/991))

### 備考

+ mac OS(10.15.7)にて構築した
+ `docker-python`フォルダ内に`.code-workspace`ファイルを作成しておくと便利かも