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
### 単位行列

かけてもかけられても相手が変化しない行列

\\[
    I = 
        \begin{pmatrix}
            1 &   &         \\\\  
              & 1 &         \\\\  
              &   & \ddots  \\\\  
        \end{pmatrix}
\\]

### 逆行列

まるで逆数のような働きをする行列

$AA^{-1} = A^{-1}A = I$  
(「-1乗」ではなく 「**inverse**」)

掃き出し法などで求める

#### 逆行列が存在しない行列  

解がない/一組に定まらない連立方程式の係数を抜き出したような行列  
形式的には

\begin{pmatrix}
    a & b       \\\\  
    c & d       \\\\  
\end{pmatrix} という行列があったとき、$ad - bc = 0$  

また  

\\[
    \begin{pmatrix}
        a & b       \\\\  
        c & d       \\\\  
    \end{pmatrix} =
    \begin{pmatrix}
        \vec{v_1}       \\\\  
        \vec{v_2}       \\\\  
    \end{pmatrix}
\\]
と考えたとき、二つのベクトルに囲まれた  
`平行四辺形の面積 = 0`  
の場合は逆行列が存在しない

## 統計学1
### 行列式(determinant)

上記の平行四辺形の面積が逆行列の有無を示す  
これを  
\\[
    \begin{vmatrix}
        a & b       \\\\  
        c & d       \\\\  
    \end{vmatrix} =
    \begin{vmatrix}
        \vec{v_1}       \\\\  
        \vec{v_2}       \\\\  
    \end{vmatrix}
\\]  
と表し、**逆行列**と呼ぶ

#### 特徴  
+ 同じ行ベクトルが含まれていると行列式は0
+ 1つのベクトルが$\lambda$倍されると行列式は$\lambda$倍される
+ 他の成分が全部同じで$i$番目のベクトルだけが違う場合、行列式の足し合わせになる

3つ以上のベクトルからできている行列式は展開できる

\\[
    \begin{vmatrix}
        \vec{v_1} \\\\  
        \vec{v_2} \\\\  
        \vec{v_3} 
    \end{vmatrix} = 
    \begin{vmatrix}
        a & b & c \\\\  
        d & e & f \\\\  
        g & h & i 
    \end{vmatrix} = 
    \begin{vmatrix}
        a & b & c \\\\  
        0 & e & f \\\\  
        0 & h & i 
    \end{vmatrix} + 
    \begin{vmatrix}
        0 & b & c \\\\  
        d & e & f \\\\  
        0 & h & i 
    \end{vmatrix} + 
    \begin{vmatrix}
        0 & b & c \\\\  
        0 & e & f \\\\  
        g & h & i 
    \end{vmatrix}
\\]  

\\[
    = a
    \begin{vmatrix}
        e & f \\\\  
        h & i 
    \end{vmatrix} - 
    d
    \begin{vmatrix}
        b & c \\\\  
        h & i 
    \end{vmatrix} +
    g
    \begin{vmatrix}
        b & c \\\\  
        e & f 
    \end{vmatrix}
\\]

#### 行列式の求め方

\\[
    \begin{vmatrix}
        a & b       \\\\  
        c & d       \\\\  
    \end{vmatrix} = ad - bc
\\]
3つ以上のベクトルでできている場合は展開して求める

参考：[行列式の基本的な性質と公式](https://risalc.info/src/determinant-formulas.html)

## 統計学2