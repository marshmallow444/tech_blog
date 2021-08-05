---
layout: post
title: "Kaggle's 30 Days of ML - Day4"
tags: Kaggle 機械学習
---

Kaggleの初心者向けプログラム「30 Days of ML」に挑戦中。  

## Day4の課題

1. Python Courseの[Lesson 3のチュートリアル](https://www.kaggle.com/colinmorris/booleans-and-conditionals)を読む
1. Python Courseの[Lesson 3のexercise](https://www.kaggle.com/kernels/fork/1275163)を実施する

## Python CourseのLesson 3の内容

+ Boolean型  
+ 条件文

#### 覚えておきたいと思った点

+ Pythonでは
    + bool型は`True`か`False`
        + `bool()`でbool型に変換できる
    + 論理演算子は`and` `or` `not`
        + 同じ条件式内に`and`があれば、これが優先される
    + if文は`if` `elif` `else`
        + コードブロックを作るには、以下のように`:`と` `(半角スペース)を入れる
        ```
        if 条件：
                ←半角スペース4つでインデントを下げて処理を書く
                インデントをやめるとブロックの終わりになる
        ```
        + 1行で書くと以下 (→[参考](https://www.atmarkit.co.jp/ait/articles/2104/02/news016.html))
        ```
        値1 if 条件 else 値2
        ```

## メモ

+ blackjackでhitすべきかどうかを判定する関数を作る問題(Optional)が楽しかった。  
勝率を4割くらいまでしか上げられなかったけど。。  
(blackjackはやったことがないので、どうしたら勝ちやすいのかよくわからない・・・)
+ Github Pagesのコードブロック内でインデントするには半角スペース8個が必要