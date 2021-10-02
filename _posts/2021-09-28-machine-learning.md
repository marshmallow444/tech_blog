---
layout: post
title: "【ラビット・チャレンジ】機械学習"
tags: ラビット・チャレンジ E資格 機械学習
---

<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

[ラビット・チャレンジ](https://ai999.careers/rabbit/)の受講レポート。  

---  

## 線形回帰モデル

線形とは：比例  

n次元空間における超平面の方程式  

\\[
    $y = a_0 + a_1 x_1 + a_2 x_2 + \cdots + a_{n-1} x_{n-1}$\\\\  
    $ = a_0 + \sum_{i=1}^{n-1} a_i x_i$\\\\  
    $ = \sum_{i=0}^{n-1} a_i x_i, \mathrm{where}　x_0 = 1$\\\\  
    $= a^T x, \mathrm{where} $\\\\  
    a = 
    \begin{pmatrix}
        a_0 \\\\  
        a_1 \\\\  
        \vdots \\\\  
        a_{n-1}
    \end{pmatrix} (n次元ベクトル) \\\\  
    $a^T = (a_0 a_1 \cdots a_{n-1})$\\\\  
    x = 
    \begin{pmatrix}
        x_0 \\\\  
        x_1 \\\\  
        \vdots \\\\  
        x_{n-1}
    \end{pmatrix} (n次元ベクトル) \\\\  
    (a_0 a_1 \cdots a_{n-1})
    \begin{pmatrix}
        x_0 \\\\  
        x_1 \\\\  
        \vdots \\\\  
        x_{n-1}
    \end{pmatrix}
\\]  

ベクトルでも表すことが可能  

+ 回帰問題
    + ある入力から出力を予測
        + **線形回帰**：直線
        + **非線形回帰**：曲線
+ 回帰で扱うデータ
    + 入力 (各要素を説明変数または特徴量と呼ぶ)
        + m次元のベクトル
        + $ \boldsymbol{x} = (x_1, x_2, \cdots, x_m)^T \in \mathbb{R}^m$ (←太字はベクトル、細字は要素、$\mathbb{R}$は実数全体)
    + 出力 (目的変数)
        + スカラー値
        + $y \in \mathbb{R}^1$

cf：バクニックの原理  
cf：密度比推定  

+ 線形回帰モデル
    + 入力とm次元パラメータの線形結合を出力するモデル
        + 予測値にはハットをつける

教師データ  
$\\{(x_i, y_i); i = 1, \cdots, n\\}$  

パラメータ  
$\boldsymbol{w} = (w_1, w_2, \cdots, w_m)^T \in \mathbb{R}^m$  

線形結合  
$\hat{y} = \boldsymbol{w} ^T \boldsymbol{x} + w_0 = \sum ^m _{j=1} w_j x_j + w_0$

+ 線形結合
    + 入力ベクトルと未知のパラメータの各要素を掛け算して足し合わせたもの
    + 出力は1次元

<br>

+ モデルのパラメータ
    + 特徴量が予測値に対してどのように影響を与えるかを決定する重みの集合
        + 重みが大きいほど影響大
        + y軸との交点を表す

説明変数が1次元の場合  

モデル数式

$\overbrace{y}^{目的変数} = \overbrace{w_0}^{切片} + \overbrace{w_1}^{回帰係数} \overbrace{x_1}^{説明変数} + \overbrace{ \epsilon }^{誤差}$  
$x, y$：既知 / 入力データ  
$w$：未知 / 学習で決める  

目的変数の予測に必要な説明変数をモデルに含めていない場合、  
その説明変数の影響は誤差に乗っかってくる  

連立方程式  

$y_1 = w_0 + w_1 x_1 + \epsilon_1$  
$y_2 = w_0 + w_1 x_2 + \epsilon_2$  
$\vdots$  
$y_n = w_0 + w_1 x_n + \epsilon_n$  

行列表現  

$\boldsymbol{y} = X \boldsymbol{w} + \epsilon$

\\[
    \overbrace{
        \begin{pmatrix}
            y_1 \\\\  
            y_2 \\\\  
            \vdots \\\\  
            y_n
        \end{pmatrix}
    }^{n × 1}
    = 
    \overbrace{
        \begin{pmatrix}
            1 & x_1 \\\\  
            1 & x_2 \\\\  
            \vdots & \vdots \\\\  
            1 & x_n
        \end{pmatrix}
    }^{n × 2}
    \overbrace{
        \begin{pmatrix}
            w_0 \\\\  
            w_1
        \end{pmatrix}
    }^{2 × 1}
\\]  

プログラムを書くときは、行列のshapeに注意  

説明変数が多次元の場合(m > 1)  

モデル数式

$\overbrace{y}^{目的変数} = \overbrace{w_0}^{切片} + \overbrace{w_1}^{回帰係数} \overbrace{x_1}^{説明変数} +  \overbrace{w_2}^{回帰係数} \overbrace{x_2}^{説明変数} + \overbrace{ \epsilon }^{誤差}$  
$x, y$：既知 / 入力データ  
$w$：未知 / 学習で決める  

データは回帰局面に誤差が加わり観測されていると仮定  

連立方程式  

$y_1 = w_0 + w_1 x_{11} + w_2 x_{12} + \cdots + w_m x_{1m} + \epsilon_1$  
$y_2 = w_0 + w_1 x_{21} + w_2 x_{22} + \cdots + w_m x_{2m} + \epsilon_2$  
$\vdots$  
$y_n = w_0 + w_1 x_{n1}+ w_2 x_{n2} + \cdots + w_m x_{nm} + \epsilon_n$  

$ \Longrightarrow \boldsymbol{y} = \underbrace{X}_{計画行列} \boldsymbol{w}$

未知パラメータ$w$は最小に情報により推定  

行列表現  

$\boldsymbol{y} = X \boldsymbol{w} + \epsilon$

\\[
    \overbrace{
        \begin{pmatrix}
            y_1 \\\\  
            y_2 \\\\  
            \vdots \\\\  
            y_n
        \end{pmatrix}
    }^{n × 1}
    = 
    \overbrace{
        \begin{pmatrix}
            1 & x_{11} & \cdots & x_{1m} \\\\  
            1 & x_{21} & \cdots & x_{2m} \\\\  
            \vdots & \vdots & \vdots & \vdots \\\\  
            1 & x_{n1} & \cdots & x_{nm}
        \end{pmatrix}
    }^{n × (m + 1)}
    \overbrace{
        \begin{pmatrix}
            w_0 \\\\  
            w_1 \\\\  
            \vdots \\\\  
            w_m
        \end{pmatrix}
    }^{(m + 1) × 1}
\\]  

行列を見てこんがらがったら、いったん連立方程式に変換して考えてみるとよい  
データ数(n) > m+1でないと解けない  

---