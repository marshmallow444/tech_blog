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
        + $ \bm{x} = (x_1, x_2, \cdots, x_m)^T \in \mathbb{R}^m$ (←太字はベクトル、細字は要素、$\mathbb{R}$は実数全体)
    + 出力 (目的変数)
        + スカラー値
        + $y \in \mathbb{R}^1$

cf：バクニックの原理  
cf：密度比推定  

