---
layout: post_toc
title: "【ラビット・チャレンジ】深層学習 後編 Day3"
tags: ラビット・チャレンジ E資格 機械学習
---

<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

[ラビット・チャレンジ](https://ai999.careers/rabbit/)の受講レポート。  

---  

# 前回の復習

**確認テスト**  

サイズ5×5の入力画像を、サイズ3×3のフィルタで畳み込んだときの出力画像のサイズは？  
ストライドは2, パディングは1とする  

【解答】  

3×3  

# 再帰型ニューラルネットワークについて

## 再帰型ニューラルネットワークの概念

### RNN全体像

#### RNNとは

<u>時系列データ</u>に対応可能なニューラルネットワーク  


#### 時系列データ

時間的順序を追って一定間隔ごとに観察され、  
しかも相互に統計的依存関係が認められるようなデータの系列  

例：  

+ 音声データ
+ テキストデータ

#### RNNについて

全体像  

[![RNN](http://www.net.c.dendai.ac.jp/~ogata/figure/004.jpg)](http://www.net.c.dendai.ac.jp/~ogata/figure/004.jpg)  
(画像：[http://www.net.c.dendai.ac.jp/~ogata/](http://www.net.c.dendai.ac.jp/~ogata/))

「展開」の矢印の左右の図の意味は同じ  

<br>

[![weight](https://4.bp.blogspot.com/-4sNWGgBFLkE/WKcLzxWRa9I/AAAAAAAAiGQ/sqV16vvhGH09_QyZK5sIPJ8UcEyhekp5ACLcB/s1600/OldR%251C%251CNN.png)](https://4.bp.blogspot.com/-4sNWGgBFLkE/WKcLzxWRa9I/AAAAAAAAiGQ/sqV16vvhGH09_QyZK5sIPJ8UcEyhekp5ACLcB/s1600/OldR%251C%251CNN.png)  
(画像：[http://maruyama097.blogspot.com/2017/02/lstmrnn.html](http://maruyama097.blogspot.com/2017/02/lstmrnn.html))

**RNNの数学的記述**  

$$
    u^t = W_{(in)}x^t + W \overbrace{z^{t-1}}^{*1}  + b \\  
    z^t = f(W_{(in)}x^t + Wz^{t-1} + b) \\  
    v^t = W_{(out)} z^t + c \\  
    y^t = g(W_{(out)} z^t + c) \\  
$$

$$
    \begin{split}
        W_{(in)}: &入力層から中間層への重み \\  
        W_{(out)}: &中間層から出力層への重み \\  
        *1: &前の時間の中間層の出力
    \end{split}
$$
 

活性化関数を通る前：$u / v$  
活性化関数を通った後：$z / y$  
入力側の矢印：$u, z$  
出力側の矢印：$v, y$  

```Python
u[:, t+1] = np.dot(X, W_in) + np.dot(z[:, t].reshape(1, -1), W)
z[:, t+1] = functions.sigmoid(u[:, t+1])
np.dot(z[:, t+1].reshape(1, -1), W_out) # vにあたる部分
y[:, t] = functions.sigmoid(np.dot(z[:, t+1].reshape(1, -1), W_out))
```

**確認テスト**  

RNNのネットワークには大きく分けて3つの重みがある。  
1つは入力から現在の中間層を定義する際にかけられる重み、  
1つは中間層から出力を定義する際にかけられる重みである。  
残り1つの重みについて説明せよ。  

【解答】  
中間層から中間層への重み  

**RNNの特徴**  
時系列モデルを扱うには、初期の状態と過去の時間t-1の状態を保持し、  
そこから次の時間でのtを再帰的に求める再帰構造が必要  

### BPTT

#### BPTTとは

#### 逆誤差伝播法の復習

#### BPTTの数学的記述

#### BPTTの全体像

## LSTM

### 全体像

### CEC

### 入力ゲートと出力ゲート

### 忘却ゲート

### 覗き穴結合

## GRU

## 双方向RNN

# RNNでの自然言語処理

## Seq2Seq

### 全体像

### Encoder RNN

### Decoder RNN

### HRED

### VHRED

### VAE

#### オートエンコーダー

#### VAE

## Word2Vec

## AttentionMechanism

# 実装演習