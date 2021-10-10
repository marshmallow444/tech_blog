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

線形≒比例  
→n次元空間における超平面の方程式  
ベクトルでも表すことが可能  

\\[
    \begin{split}
        y &= a_0 + a_1 x_1 + a_2 x_2 + \cdots + a_{n-1} x_{n-1}\\\\  
        &= a_0 + \sum_{i=1}^{n-1} a_i x_i\\\\  
        &= \sum_{i=0}^{n-1} a_i x_i, \mathrm{where}　x_0 = 1\\\\  
        &= a^T x, \mathrm{where}\\\\  
        &a = 
        \begin{pmatrix}
            a_0 \\\\  
            a_1 \\\\  
            \vdots \\\\  
            a_{n-1}
        \end{pmatrix} (n次元ベクトル) \\\\  
        &a^T = (a_0 a_1 \cdots a_{n-1})\\\\  
        &x = 
        \begin{pmatrix}
            x_0 \\\\  
            x_1 \\\\  
            \vdots \\\\  
            x_{n-1} \\\\  
        \end{pmatrix} (n次元ベクトル) \\\\  
        &(a_0 a_1 \cdots a_{n-1})
        \begin{pmatrix}
            x_0 \\\\  
            x_1 \\\\  
            \vdots \\\\  
            x_{n-1}
        \end{pmatrix}
    \end{split}  
\\]  

### 回帰問題
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

参考：
+ [Vapnikの原理](https://scrapbox.io/Nodewww/Vapnikの原理)  
+ 密度比推定 (東工大 杉山先生の研究がおもしろい)  

<br>

### 線形回帰モデル
+ 入力とm次元パラメータの線形結合を出力するモデル
    + 予測値にはハットをつける

+ 教師データ:  
    + $\\{(x_i, y_i); i = 1, \cdots, n\\}$  
+  パラメータ:  
    + $\boldsymbol{w} = (w_1, w_2, \cdots, w_m)^T \in \mathbb{R}^m$  
+ 線形結合:  
    + $\hat{y} = \boldsymbol{w} ^T \boldsymbol{x} + w_0 = \sum ^m _{j=1} w_j x_j + w_0$
    + 入力ベクトルと未知のパラメータの各要素を掛け算して足し合わせたもの
    + 出力は1次元
+ モデルのパラメータ
    + 特徴量が予測値に対してどのように影響を与えるかを決定する重みの集合
        + 重みが大きいほど影響大
        + y軸との交点を表す

<br>

#### 説明変数が1次元の場合  

【モデル数式】

$\overbrace{y}^{目的変数} = \overbrace{w_0}^{切片} + \overbrace{w_1}^{回帰係数} \overbrace{x_1}^{説明変数} + \overbrace{ \epsilon }^{誤差}$  

$x, y$：既知 / 入力データ  
$w$：未知 / 学習で決める  

目的変数の予測に必要な説明変数をモデルに含めていない場合、  
その説明変数の影響は誤差に乗っかってくる  

【連立方程式】  

$y_1 = w_0 + w_1 x_1 + \epsilon_1$  
$y_2 = w_0 + w_1 x_2 + \epsilon_2$  
$\vdots$  
$y_n = w_0 + w_1 x_n + \epsilon_n$  

【行列表現】  

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

<br>

#### 説明変数が多次元の場合(m > 1)  

【モデル数式】

$\overbrace{y}^{目的変数} = \overbrace{w_0}^{切片} + \overbrace{w_1}^{回帰係数} \overbrace{x_1}^{説明変数} +  \overbrace{w_2}^{回帰係数} \overbrace{x_2}^{説明変数} + \overbrace{ \epsilon }^{誤差}$  

$x, y$：既知 / 入力データ  
$w$：未知 / 学習で決める  

データは回帰局面に誤差が加わり観測されていると仮定  

【連立方程式】  

$y_1 = w_0 + w_1 x_{11} + w_2 x_{12} + \cdots + w_m x_{1m} + \epsilon_1$  
$y_2 = w_0 + w_1 x_{21} + w_2 x_{22} + \cdots + w_m x_{2m} + \epsilon_2$  
$\vdots$  
$y_n = w_0 + w_1 x_{n1}+ w_2 x_{n2} + \cdots + w_m x_{nm} + \epsilon_n$  

$ \Longrightarrow \boldsymbol{y} = \underbrace{X}_{計画行列} \boldsymbol{w}$

未知パラメータ$w$は最小に情報により推定  

【行列表現】  

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

備考：
+ 行列を見てこんがらがったら、いったん連立方程式に変換して考えてみるとよい  
+ データ数$n > m+1$でないと解けない  
+ 参考文献 「[機械学習のエッセンス](https://www.amazon.co.jp/dp/B07GYS3RG7/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)」

### 汎化

データを学習用と検証用に分割  
→モデルの汎化性能を測定するため  

### 最小二乗法

線形回帰モデルのパラメータはこの方法で推定  

+ 平均二乗誤差(残差平方和)
    + データとモデル出力の二乗誤差の和
+ 最小二乗法
    + 学習データの**平均二乗誤差**を最小とするパラメータを探索
        + その勾配が0となる点を求めればよい

$ \sum _i \overbrace{ (\hat{y}_i - y_i)^2 }^{二乗誤差} = \sum _i \epsilon^2_i $  

※二乗損失は一般に外れ値に弱い  
→Huber損失、Tukey損失  

参考文献：  
+ イラストで学ぶ機械学習

$\mathrm{MSE_{train}} = J(W) = \frac{1}{n_{train}} \sum_{i=1}^{n_{train}} (\overbrace{ \hat y_i^{(train)} }^{W^Tx_i^{(train)}} - y_i^{(train)})^2$  

+ $\hat{ \boldsymbol{w} } = \mathrm{arg} \space \mathrm{min} \space \mathrm{MSE} _{train}$  
$ \qquad \boldsymbol{w} \in \mathbb{R}^{m+1}$
    + MSEを最小にするような$w$ (m次元)
    + `arg`：パラメータを指す
    + `min`：最小値
+ $\frac{\partial}{\partial \boldsymbol{w}} \mathrm{MSE}_{train} = 0$
    + MSEをwに関して微分したものが0となるwの点を求める

<br>

【$\sum$を行列に変換】

$ \hat{ \boldsymbol{w} } = \mathrm{arg} \space \mathrm{min} \space \mathrm{MSE} _{train}$  

$\frac{\partial}{\partial \boldsymbol{w}} \mathrm{MSE} _{train} = 0$  

$\Longrightarrow \frac{\partial}{\partial \boldsymbol{w}} \\{\frac{1}{n_{train}} \sum_{i=1}^{n_{train}} ( \hat y_i^{(train)} - y_i^{(train)} )^2 \\} = 0$  

$\Longrightarrow \frac{\partial}{\partial \boldsymbol{w}} \\{\frac{1}{n_{train}} \sum_{i=1}^{n_{train}} ( \boldsymbol{x}_i^T \cdot \boldsymbol{w} - y_i^{(train)} )^2 \\} = 0$  

$\Longrightarrow \frac{\partial}{\partial \boldsymbol{w}} \\{\frac{1}{n_{train}} (\boldsymbol{X} \boldsymbol{w} - \boldsymbol{y})^T (\boldsymbol{X} \boldsymbol{w} - \boldsymbol{y}) \\} = 0$  

\\[
    \boldsymbol{X} \boldsymbol{w} - \boldsymbol{y} \\\\  
    \longrightarrow
    \overbrace{
        \begin{pmatrix}
            1 & x_{11} & \cdots & x_{1m} \\\\  
            1 & x_{21} & \cdots & x_{2m} \\\\  
            \vdots & \vdots & \vdots & \vdots \\\\  
            1 & x_{n1} & \cdots & x_{nm}
        \end{pmatrix}
    }^{X}
    \overbrace{
        \begin{pmatrix}
            w_0 \\\\  
            w_1 \\\\  
            \vdots \\\\  
            w_m
        \end{pmatrix}
    }^{w} -
    \overbrace{
        \begin{pmatrix}
            y_1 \\\\  
            y_2 \\\\  
            \vdots \\\\  
            y_n
        \end{pmatrix}
    }^{y}
\\]  

$\Longrightarrow \frac{1}{n_{train}} \cdot \frac{\partial}{\partial \boldsymbol{w}} \\{ (\boldsymbol{w}^T \boldsymbol{X}^T  - \boldsymbol{y}^T) (\boldsymbol{X} \boldsymbol{w} - \boldsymbol{y}) \\} = 0$  

$\Longrightarrow \frac{1}{n_{train}} \cdot \frac{\partial}{\partial \boldsymbol{w}} \\{ \boldsymbol{w}^T \boldsymbol{X}^T \boldsymbol{X} \boldsymbol{w} - \overbrace{ \boldsymbol{w}^T \boldsymbol{X}^T \boldsymbol{y} - \boldsymbol{y}^T \boldsymbol{X} \boldsymbol{w} }^{等しい} - \boldsymbol{y}^T\boldsymbol{y} \\} = 0$  

$\Longrightarrow \frac{1}{n_{train}} \cdot \frac{\partial}{\partial \boldsymbol{w}} \\{ \underbrace{ \boldsymbol{w}^T \boldsymbol{X}^T \boldsymbol{X} \boldsymbol{w} }_{wの2次項} - \overbrace{ 2 \boldsymbol{w}^T \boldsymbol{X}^T \boldsymbol{y} }^{wの1次項} - \boldsymbol{y}^T\boldsymbol{y} \\} = 0$  

$\Longrightarrow \frac{1}{n_{train}} \\{ 2 \boldsymbol{X}^T \boldsymbol{X} \boldsymbol{w} - 2 \boldsymbol{X}^T \boldsymbol{y} \\} = 0$  

\\[
    \begin{split}
        (∵) \frac{ \partial( \boldsymbol{w} ^T \boldsymbol{w} ) }{ \partial \boldsymbol{w} }  &= \boldsymbol{x} \\\\  
        \frac{\partial (\boldsymbol{w} ^T A \boldsymbol{w})}{\partial \boldsymbol{w}} &= (A + A^T) \cdot \boldsymbol{x} \\\\  
        &= 2A \boldsymbol{x} (A:対称行列)
    \end{split}
\\]

$\Longrightarrow 2 \boldsymbol{X}^T \boldsymbol{X} \boldsymbol{w} - 2 \boldsymbol{X}^T \boldsymbol{y} = 0$  

$\Longrightarrow 2 \boldsymbol{X}^T \boldsymbol{X} \boldsymbol{w} = 2 \boldsymbol{X}^T \boldsymbol{y}$  

$\Longrightarrow \overbrace{ (\boldsymbol{X}^T \boldsymbol{X})^{-1} (\boldsymbol{X}^T \boldsymbol{X}) }^{A^{-1}A = I (単位行列)} \cdot \boldsymbol{w} = (\boldsymbol{X}^T \boldsymbol{X})^{-1} \boldsymbol{X}^T \cdot \boldsymbol{y}$  

$\Longrightarrow \boldsymbol{w} = (\boldsymbol{X}^T \boldsymbol{X})^{-1} \cdot \boldsymbol{X}^T \boldsymbol{y}$

参考書籍：
+ Matrix Cook Book

<br>

【回帰係数】  

$\hat{ \boldsymbol{w} } = (X^{(train)T} X^{(train)})^{-1} X^{(train)T} \boldsymbol{y} ^{(train)}$  

【予測値】  

$\hat{ \boldsymbol{y} } = X (X^{(train)T} X^{(train)})^{-1} X^{(train)T} \boldsymbol{y} ^{(train)}$  

<br>

$\hat{ \boldsymbol{w} } = \underbrace{(X^{T} X)^{-1}} X^{T} \boldsymbol{y}$  

$\qquad$ ↑逆行列は常に存在するわけではない (一般化逆行列)

$\hat{ \boldsymbol{y} } = \overbrace{ X_* }^{予測したい新たな入力点(n_* 個)} \cdot \hat{ \boldsymbol{w} } = \overbrace{ X_* }^{n_* \times (m+1)} \cdot \overbrace{ (X^{T} X)^{-1} X^{T} \boldsymbol{y} }^{(m+1) \times 1}$  

$\qquad ↑ X_* \cdot \hat{ \boldsymbol{w} } =  X_* \cdot (X^{T} X)^{-1} X^{T}$ の部分を**射影行列**という  

<br>

### ハンズオン  

#### ボストンの住宅価格予測

【データセットの中身を確認する際の注意点】  

+ 不正な値がないか？
    + 範囲外の値？外れ値？
+ ヒストグラムに直してみる
+ avr, min, maxを確認してみる
+ 上限を超えた値は一律上限の値にされている、など

まず最初の数行を見て、気になる点があれば表示する行数を増やしてみて確認する  

【予測した値がおかしかった際に疑うこと】  

+ モデルの設計をミスした？
    + 範囲外の値をとらないモデルにする？
+ 元のデータセットにない値を使って予測しようとした？
    + →外挿問題

ドメイン知識に基づいて、予測結果が正しいかを確認する  

<br>

---  

## 非線形回帰モデル  

単回帰 / 重回帰  

$y = w_0 + w_1 \cdot x \quad / \quad y = w_0 + w_1 \cdot x_1 + w_2 \cdot x_2 + \cdots + w_m \cdot x_m$  

<br>

非線形な回帰を考えたい：  
e.g.  
+ $y = w_0 + w_1 \cdot x + w_2 \cdot \overbrace{ x^2 }^{ \phi_2(x) } + w_3 \cdot \overbrace{ x^3 }^{ \phi_3(x) }$
+ $y = w_0 + w_1 \cdot \overbrace{ \sin x } + w_2 \cdot \overbrace{ \cos x } + w_3 \cdot \overbrace{ \log x }$

$\Longrightarrow$Idea: $x$の代わりに、$\underbrace{ \phi(x) }_{xの関数(任意)}$を用いる  
$\qquad x$が$\phi(x)$に代わるだけ！  

※$x$から$\phi(x)$に代えても、$w$については線形のまま  
$f(x) = ax^2+bx+c$ ・・・$x$についての2次関数  
$f(a) = ax^2+bx+c$ ・・・$a$についての1次関数  

予測モデル：$\hat y = w_0 + w_1 \cdot \phi_1(x) + w_2 \cdot \phi_w(x) + \cdots + w_m \cdot \phi_m(x)$  
$\Longrightarrow$これは重み$w$について線形 (linear-in-parameter)  

+ 基底展開法
    + 回帰係数として、基底関数とパラメータベクトルの線形結合を使用
        + 基底関数：既知の非線形関数
    + 未知パラメータは最小2乗法や最尤法により推定
        + (線形回帰モデルと同様)

$\qquad \qquad y_ = w_0 + \sum _{j=1}^m w_j \overbrace{ \phi_j(x_i) }^{基底関数} + \epsilon_i$  

+ よく使われる基底関数
    + 多項式関数
        + $\phi_j = x^j$
    + ガウス型基底関数
        + $\phi_j(x) = \exp{ \frac{(x - \mu_j)^2}{2h_j} }$
    + スプライン関数 / Bスプライン関数

☆基底関数 $\phi(x)$ に多項式関数$\phi_j = x^j$ を考える  

\\[
    \begin{split}
        \hat y_i &= w_0 + w_1 \cdot \overbrace{ \phi_1(x_i) }^{x_i^1} + w_2 \cdot \overbrace{ \phi_2(x_i) }^{x_i^2} + \cdots + w_9 \cdots \overbrace{ \phi_9(x_i) }^{x_i^9} \\\\  
        &= \overbrace{ w_0 } + \overbrace{ w_1 } \cdot x_i + \overbrace{ w_2 } \cdot x_i^2 + \cdots + \overbrace{ w_9 } \cdot x_i^9 \\\\  
        &求めるべきwについては線形であることに注意 \\\\
    \end{split}
\\]

  