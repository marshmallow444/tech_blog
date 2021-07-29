---
layout: post
title: "【ラビット・チャレンジ】応用数学"
tags: ラビット・チャレンジ E資格 機械学習
---

<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

[ラビット・チャレンジ](https://ai999.careers/rabbit/)の受講レポート。  

## 線形代数学 (行列)

### スカラーとベクトル

#### スカラー

- 普通の数
- 四則演算可能
- ベクトルの係数になれる

#### ベクトル

- 「大きさ」と「向き」をもつ
- 矢印で図示される
- スカラーのセットで表示される
- 数字の組み合わせ

どちらも数として扱える

### 行列

- スカラーを表にしたもの
- ベクトルを並べたもの

### 連立方程式

\\[
    x_1 + 2x_2 = 3\\\\  
    2x_1 + 5x_2 = 5
\\]

の式を $A\vec{x} = \vec{b}$ の形にすると、以下のようになる

\\[
    \left(
        \begin{array}{cc}
            1 & 2 \\\\  
            2 & 5
        \end{array}
    \right)
    \left(
        \begin{array}{c}
            x_1 \\\\  
            x_2
        \end{array}
    \right) = 
    \left(
        \begin{array}{c}
            3 \\\\  
            5
        \end{array}
    \right)
\\]

係数をまとめて表のようにした部分を**行列**という

#### 行基本変形  

= 行列の変形  
　→行列を左からかけることで表現できる

(1) i行目をc倍する  
(2) s行目にt行目のc倍を加える  
(3) p行目とq行目を入れ替える  
　　(→連立方程式での例：2行目に$x_1$, 1行目に$x_2$が残ってしまっているので入れ替える)  

参考：[連立方程式の解き方(加減法,代入法)](https://math.005net.com/yoten/renrituKagen.php)  

各工程で使用する行列

(1) i行目をc倍する
\\[
    Q_{i, c} =
    \begin{pmatrix}
        1   &           &   &   &   &           &   \\\\  
            & \ddots    &   &   &   &           &   \\\\  
            &           & 1 &   &   &           &   \\\\  
            &           &   & c &   &           &   \\\\  
            &           &   &   & 1 &           &   \\\\  
            &           &   &   &   & \ddots    &   \\\\  
            &           &   &   &   &           & 1 \\\\  
    \end{pmatrix}
\\]

+ $(i, i)$番目の要素を$c$倍する  

(2) s行目にt行目のc倍を加える
\\[
    R_{s, t, c} =
    \begin{pmatrix}
        1   &           &   &           &   &           &   \\\\  
            & \ddots    &   &           &   &           &   \\\\  
            &           & 1 &           & c &           &   \\\\  
            &           &   & \ddots    &   &           &   \\\\  
            &           &   &           & 1 &           &   \\\\  
            &           &   &           &   & \ddots    &   \\\\  
            &           &   &           &   &           & 1 \\\\  
    \end{pmatrix}
\\]

+ $(s, t)$の成分を$c$に変える

(3) p行目とq行目を入れ替える  

\\[
    P_{p, q} =
    \begin{pmatrix}
        1   &           &   &           &   &           &   \\\\  
            & \ddots    &   &           &   &           &   \\\\  
            &           & 0 &           & 1 &           &   \\\\  
            &           &   & \ddots    &   &           &   \\\\  
            &           & 1 &           & 0 &           &   \\\\  
            &           &   &           &   & \ddots    &   \\\\  
            &           &   &           &   &           & 1 \\\\  
    \end{pmatrix}
\\]

+ $(p, p)$, $(q, q)$の成分を0に変える
+ $(p, q)$, $(q, p)$の成分を1に変える

#### 逆行列

まるで逆数のような働きをする行列

#### 単位行列

かけてもかけられても相手が変化しない行列


## 線形代数学 (固有値)

## 統計学1

## 統計学2