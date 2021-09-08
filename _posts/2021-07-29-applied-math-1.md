---
layout: post
title: "【ラビット・チャレンジ】応用数学"
tags: ラビット・チャレンジ E資格 機械学習
---

<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

[ラビット・チャレンジ](https://ai999.careers/rabbit/)の受講レポート。  

---  

# 【線形代数学 (行列)】

## スカラーとベクトル

+ **スカラー**：普通の数
+ **ベクトル**：「大きさ」と「向き」をもつ

## 行列

- スカラーを表にしたもの
- ベクトルを並べたもの

## 連立方程式

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

### 行基本変形  

= 行列の変形  
　→行列を左からかけることで表現できる  

手順：  

(1) i行目をc倍する  
(2) s行目にt行目のc倍を加える  
(3) p行目とq行目を入れ替える  
　　(→連立方程式での例：2行目に$x_1$, 1行目に$x_2$が残ってしまっているので入れ替える)  

参考：[連立方程式の解き方(加減法,代入法)](https://math.005net.com/yoten/renrituKagen.php)  

各工程で使用する行列

(1) i行目をc倍する
+ $(i, i)$番目の要素を$c$倍する  
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

(2) s行目にt行目のc倍を加える
+ $(s, t)$の成分を$c$に変える  
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

(3) p行目とq行目を入れ替える  
+ $(p, p)$, $(q, q)$の成分を0に変える
+ $(p, q)$, $(q, p)$の成分を1に変える

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

---

# 【線形代数学 (固有値)】

\\[
	A\vec{x} = \lambda\vec{x}
\\]
が成り立つような行列$A$, 特殊なベクトル$\vec{x}$, 右辺の係数$\lambda$があるとき、

+ $\vec{x}$: 行列$A$に対する**固有ベクトル**
    + 一つに定まらない
    + 「$\vec{x}$の定数倍」のように表す
+ $\lambda$: 行列$A$に対する**固有値**
    + 一つに定まる

#### 固有値と固有ベクトルの求め方：  
\\[
	\begin{vmatrix}
	    A - \lambda I = 0	\\\\  
	\end{vmatrix}
\\]
となるような$\lambda$を求め、
\\[
    A
	\begin{pmatrix}
		x_1\\\\  
        x_2
	\end{pmatrix}
    = \lambda
    \begin{pmatrix}
		x_1\\\\  
        x_2
	\end{pmatrix}
\\]
を解いて$x_1$と$x_2$の比を求める

### 固有値分解

+ ある実数を正方形に並べた行列$A$
+ $A$の固有値$\lambda_1, \lambda_2, ...$
+ $A$の固有ベクトル$\vec{v_1}, \vec{v_2}, ...$

があるとき、固有値を対角線上に並べた行列  
\\[
    \Lambda = 
	\begin{pmatrix}
		\lambda_1 &  &  \\\\  
         & \lambda_2 & \\\\  
         & & \ddots
	\end{pmatrix}
\\]
と、それに対応する固有ベクトルを並べた行列
\\[
    V = 
	\begin{pmatrix}
         & & \\\\  
		\vec{v_1} & \vec{v_2} & \cdots \\\\  
         & & \\\\  
	\end{pmatrix}
\\]
を用意したとき、$AV = V\Lambda$となる  
変形すると$A = V \Lambda V^{-1}$  
+ **固有値分解**：正方形の行列を上記のような3つの行列の積に分解すること
    + 利点：行列の累乗が容易になる　など
+ $\Lambda$の中身は、$\lambda$を小さい順or大きい順に並べることが多い

### 特異値分解

正方行列以外の行列において
\\[
	M\vec{v} = \sigma\vec{u} \\\\  
    M^T\vec{u} = \sigma\vec{v}
\\]
となる特殊な単位ベクトルがある場合、**特異値分解**が可能
\\[
    M = USV^T
\\]

+ $U$や$V$は直行行列
    + 複素数を要素に持つ場合はユニタリ行列
+ $S$ = Sigma

#### 特異値の求め方

$MV = US　→　M = USV^T$  
$M^TU = VS^T　→　M^T = VS^TU^T$  
これらの積は  
$MM^T = USV^TVS^TU^T = USS^TU^T$  
($MM^T$で正方行列を作って固有値分解する)

#### 特異値分解の利用例

+ 画像データの圧縮
+ 機械学習の前処理
    + 特異値の大きい部分が似ている画像どうしは、画像の特徴も似ている
    + 画像の分類などができる

---

# 【統計学1】

## 集合とは？

→ものの集まり

$S = \\{ a, b, c, d, e, f, g \\}$  
$a \in S$ ← $a$は集合$S$の要素  
$h \notin S$ ← $h$は集合$S$の要素ではない  
(「要素」は「元(げん)」と呼ばれることもある)

$M = \\{ c, d, e \\}$  
$M \subset S$ ← $M$は$S$の一部  

---

# 【統計学2】