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
+ [イラストで学ぶ機械学習](https://www.amazon.co.jp/イラストで学ぶ-機械学習-最小二乗法による識別モデル学習を中心に-KS情報科学専門書-杉山/dp/4061538217)

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
+ [Matrix Cook Book](https://www.opst.co.jp/openinnovation/report/blog/blog-report210305_01)

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
+ e.g.  
    + $y = w_0 + w_1 \cdot x + w_2 \cdot \overbrace{ x^2 }^{ \phi_2(x) } + w_3 \cdot \overbrace{ x^3 }^{ \phi_3(x) }$
    + $y = w_0 + w_1 \cdot \overbrace{ \sin x } + w_2 \cdot \overbrace{ \cos x } + w_3 \cdot \overbrace{ \log x }$

$\Longrightarrow$Idea: $x$の代わりに、$\underbrace{ \phi(x) }_{xの関数(任意)}$を用いる  
$\qquad x$が$\phi(x)$に代わるだけ！  

※$x$から$\phi(x)$に代えても、**$w$については線形のまま**  
+ $f(x) = ax^2+bx+c$ ・・・$x$についての2次関数  
+ $f(a) = ax^2+bx+c$ ・・・$a$についての1次関数  

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
        &= \underbrace{ w_0 } + \underbrace{ w_1 } \cdot x_i + \underbrace{ w_2 } \cdot x_i^2 + \cdots + \underbrace{ w_9 } \cdot x_i^9 \\\\  
        &求めるべきwについては線形であることに注意 \\\\
    \end{split}
\\]

☆Gauss型基底関数  

$\phi_j(x) = \exp \\{ - \frac{(x-\mu_j)^2}{2h_j} \\} (= \exp \\{ - \frac{(x - \mu_j)^2}{\sigma^2} \\})$

【2次元ガウス型基底関数】  

$\phi_j(\boldsymbol{x}) = \exp \\{ \frac{(\boldsymbol{x} - \boldsymbol{\mu}_j)^T(\boldsymbol{x} - \boldsymbol{\mu}_j)}{2h_j} \\}$

<br>

+ 説明変数
    + $\boldsymbol{x_i} = ( x_{i1}, x_{i2}, \cdots, x_{im} ) \in \mathbb{R}^m$ (m:説明変数の数)
+ 非線形関数ベクトル
    + $ \phi( \boldsymbol{x}_i ) = (\phi_1(\boldsymbol{x}_i), \phi_2(\boldsymbol{x}_i), \cdots, \phi_k(\boldsymbol{x}_i) )^T \in \mathbb{R}^k$ (k:基底関数の数)
+ 非線形関数の計画行列
    + $ \Phi^{train} = ( \phi(\boldsymbol{x}_1), \phi(\boldsymbol{x}_2), \cdots, \phi(\boldsymbol{x}_n) )^T \in \mathbb{R}^{n \times k} $
+ 最尤法による予測値
    + $ \boldsymbol{ \hat y } = \Phi(\Phi^{(train)T} \Phi^{(train)})^{-1}\Phi^{(train)T} \boldsymbol{y}^{(train)} $

$\Longrightarrow$基底展開法も線形回帰と同じ枠組みで推定可能

復習: $\overbrace{y}^{n \times 1} = \overbrace{X}^{n \times (m+1)} \overbrace{w}^{(m+1) \times 1} \Longrightarrow $ Now: $\overbrace{y}^{n \times 1} = \overbrace{\Phi}^{n \times (m+1)} \cdot \overbrace{w}^{(m+1) \times 1}$  

→結局、MSEを最小化する$w$は先と同様に、  

$\hat w = (X^T X)^{-1} X^T y$  
$\rightarrow \hat w = (\Phi^T \Phi)^{-1} \Phi^T y$  

∴$\hat y = \Phi_* \cdot \hat w = \Phi_* \cdot (\Phi^T \Phi)^{-1} \Phi^T y$  

<br>

### 未学習(underfitting)と過学習(overfitting)  

+ **未学習**：学習データに対し、十分小さな誤差が得られない
    + 対策：
        + 表現力の高いモデルを利用
+ **過学習**：小さな誤差は得られたが、テスト集合誤差との差が大きい
    + 対策：
        1. 学習データの数を増やす
        1. **不要な基底関数(変数)を削除**して表現力を抑止
            + 特徴量選択
            + ドメイン知識に基づいて削除するものを判断
            + AICによりモデルを選択
        1. **正規化法を利用**して表現力を抑止

<br>

+ 不要な基底関数を削除
    + **モデルの複雑さ**：基底関数の数、位置やバンド幅により変化
    + 適切な基底関数を用意(CV(cross validation)などで選択)
        + 解きたい問題に対して基底関数が多いと過学習の問題が発生

参考：
+ [オッカムの剃刀](https://ja.wikipedia.org/wiki/オッカムの剃刀)
+ 変数選択問題：変数の組み合わせによりモデルの複雑さが変わる(線形回帰モデル)
+ [Double Decent](https://www.acceluniverse.com/blog/developers/2020/01/deep-double-descent-where-bigger-models-and-more-data-hurt.html) (線形回帰では基本的に発生しない))

#### 正則化法(罰則化法)

+ 「モデルの複雑さに伴って、その値(w)が大きくなる **正則化項(罰則項)** を課した関数」を最小化
    + 形状によりいくつもの種類がある
        + それぞれ推定量の性質が異なる
    + 正則化(平滑化)パラメータ
        + モデルの曲線のなめらかさを調節  
        $S \gamma = (y - \overbrace{\Phi}^{n \times k}w)^T (y - \Phi w) + \overbrace{ \gamma R(w) }^{モデルの複雑化に伴う罰則} \quad \gamma (> 0)$
            + $\gamma$は重み(ハイパーパラメータ)
            + 基底関数の数(k)が増加→パラメータ増加、残差は減少。モデルは複雑化

予測：$\hat y = X_* \cdot \overbrace{ (X^T X)^{-1} X^T y }^{\hat w}$

$\hat w$が大きくなるとき：逆行列が計算できない状態＝[一次独立](https://manabitimes.jp/math/1193)でない(平行な)ベクトルが並ぶ場合  
$\Longrightarrow (X^T X)^{-1}$の要素が非常に大きくなる  

→$E(w) = \overbrace{ J(w) }^{ \mathrm{MSE} } + \overbrace{ \lambda \cdot w^T w }^{罰則項}$  
MSEが小さくなるよう$w$を考えるが、$w$→大 で 罰則→大  

+ 正則化項の役割
    + 最小2乗推定量：ない
    + **Ridge推定量**：L2ノルムを利用
        + 縮小推定：パラメータを0に近づけるよう推定
    + **Lasso推定量**：L1ノルムを利用
        + スパース推定：いくつかのパラメータを正確に0に推定
+ 正則化パラメータの役割
    + 小さく→制約面が大きく
    + 大きく→制約面が小さく

解きたいのは  
min MSE s.t. $\overbrace{R(w) \leq r}^{条件} \quad$ (不等式条件のついた最適化法)  
$\Downarrow$  
[KKT条件](https://ja.wikipedia.org/wiki/カルーシュ・クーン・タッカー条件)より、  
min MSE + $\overbrace{\lambda \cdot R(w)}^{これを付けると不等式条件を回避}$

(メモ：[JDLA_E資格の機械学習「L1ノルム、L2ノルム」](https://qiita.com/TakoTree/items/bf0b456030b114cade6d)がわかりやすかった)  

<br>

### データの分割とモデルの汎化性能測定

+ **ホールドアウト法**
    + 有限のデータを学習用とテスト用に分割
    →<u>予測精度</u>や<u>誤り率</u>を推定するために使用
        + 学習用を増やす→学習精度↑、性能評価↓
        + テスト用を増やす→学習精度↓
        + **手元にデータが大量にある場合以外は、良い性能評価を与えない**
    + 基底展開法に基づく非線形モデルでは、ホールドアウト値を小さくするモデルで以下を決定する
        + 基底関数についての値
            + 数
            + 位置
            + バンド幅
        + チューニングパラメータ
    + 学習用、テスト用のデータは固定

+ **クロスバリデーション(交差検証)**
    + データを学習用と評価用に分割
        + データをいくつかに分割
        + 検証用のデータを入れ替えながら、複数回検証を行いCV値を得る
        + 一番小さいCV値を採用

精度の計算方法：MSE  
注意：検証には**検証用データ**を使用すること！  
$\qquad$→学習用データで精度を出しては意味がない  

【ハイパーパラメータの調整方法】  

+ **グリッドサーチ**
    + 全てのパラメータの組み合わせで評価値を算出
    + 最も良い評価値を持つチューニングパラメータを持つ組み合わせを、「良いモデルのパラメータ」として採用

参考：ニューラルネットワークでの調整方法
+ [ベイズ最適化(Bayesian Optimization)](https://qiita.com/masasora/items/cc2f10cb79f8c0a6bbaa)　
    + [ハイパーパラメータ自動最適化ツール「Optuna」](https://tech.preferred.jp/ja/blog/optuna-release/)

---

## ロジスティック回帰モデル  

+ 分類問題(クラス分類)
    + ある入力からクラスに分類
+ 分類で扱うデータ
    + 入力(各要素を**説明変数**または**特徴量**と呼ぶ)
        + m次元のベクトル
            + $\boldsymbol{x} = (x_1, x_2, \cdots, x_m)^T \in \mathbb{R}^m$
    + 出力(目的変数)
        + 0 or 1の値
            + $y \in {0, 1}$
    + タイタニックデータ、IRISデータなど
    + 教師データ
        + $\\{ (x_i, y_i); \space i = 1, \cdots, n \\}$

(参考)  
[Using Machine Learning to Predict Parking Difficulty](https://ai.googleblog.com/2017/02/using-machine-learning-to-predict.html)  
→NNよりロジスティック回帰の方が精度が良かった例  

【E資格によく出る！】  

識別的アプローチ：  
$p(C_k | x)$を直接モデル化  
ロジスティック回帰はこれ  
識別関数の構成もある(SVMなど)  

生成的アプローチ：  
$p(C_k)$と$p(x | C_k)$をモデル化  
その後Bayesの定理を用いて$p(C_k | x)$を求める  
+ 外れ値の検出ができる  
+ 新たなデータを生成できる(GANとか)  

\\[
    \qquad p(C_k\|x) = \frac{p(C_k, x)}{p(x)} = \frac{p(x\|C_k) \cdot p(C_k)}{p(x)}
\\]

sigmoid関数：  

\\[
    \overbrace{x^T w}^{sigmoid関数で[0, 1]に} \in \mathbb{R} \leftarrow \rightarrow y \in \\{0, 1\\}\\\\  
    a(z) := \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}\\\\  
    実数全体から[0, 1]へつぶす  
\\]

#### ロジスティック線形回帰モデル

+ 分類問題を解くための教師あり機械学習モデル
    + 入力とm次元パラメータの**線形結合**をシグモイド関数へ入力
        + $\hat y = \boldsymbol{w^T x} + w_0 =  \sum_{j=1}^{m} w_j x_j + w_0$
    + 出力はy=1になる確率の値

+ パラメータが変わるとシグモイド関数の形が変わる
    + aを増加→0付近での勾配が増加
    + aを極めて大きくすると、単位ステップ関数に近づく
    + バイアス変化は段差の位置

\\[
    \sigma(x) = \frac{1}{1 + \exp (-ax)}
\\]

+ シグモイド関数の性質
    + シグモイド関数の微分は、シグモイド関数自身で表現することが可能
    + 尤度関数の微分の際にこの事実を利用すると計算が容易

\\[
    \begin{split}
	    &a(z) = \frac{1}{1 + \exp (-z)} = \\{1 + \exp (-z)\\}^{-1}\\\\  
        &\Longrightarrow \frac{\partial a(z)}{\partial z} = \overbrace{ -1 \cdot \\{ 1 + \exp (-x) \\}^{-2}}^{-1乗の微分} \times \overbrace{ \exp(-z) }^{\frac{\partial}{\partial z}\\{1 + \exp (-z)\\}} \times \overbrace{ (-1) }^{\frac{\partial}{\partial z}(-z)}\\\\  
        \\\\  
        & \qquad ※合成関数の微分をcheck!\\\\  
        & \qquad \frac{\partial}{\partial x}f(g(x)) = f'(g(x)) \cdot g'(x)\\\\  
        \\\\  
        &\Longrightarrow \qquad \qquad = \overbrace{ \frac{1}{1 + \exp(-z)} }^{sigmoid関数 \sigma (z)} \times \frac{\exp (-z)}{1 + exp(-z)}\\\\  
        &\Longrightarrow \qquad \qquad = a(z) \times (1 - a(z))
    \end{split}
\\]

