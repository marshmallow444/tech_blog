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

【課題】  

+ 部屋数が4で犯罪率が0.3の物件はいくらになるか？

![answer.jpg]({{site.baseurl}}/images/20211012.png)  

結果：  
<u>約4,240ドル</u>  

考察：  
+ 部屋数が増えれば住宅価格が上がる
+ 犯罪率が下がれば住宅価格が上がる
    + 部屋数の方が、住宅価格への影響は大きい
+ 学習用データセットには、部屋数が5未満のデータがほとんど含まれていないので、この予測は不正確である可能性がある
    + 同じ犯罪率で部屋数が6のときの予測結果と比較すると、上記の予測は安すぎるかもしれない
    ![result_6_rooms.jpg]({{site.baseurl}}/images/20211012_1.png)  
    (部屋数が6、犯罪率が0.3のとき、予想価格は約21,022ドル)  

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

+ シグモイド関数の出力をY=1になる確率に対応させる
    + データYは確率が基準値以上なら1、未満なら0と予測
        + 基準値：0.5など、場合に応じて決める

【求めたい値】  

$ \overbrace{P(Y=1 \| x)}^{説明変数の実現値が与えられた際にY=1になる確率}  = \overbrace{\sigma}^{シグモイド関数} ( \overbrace{w_0 + w_1 x_1 + \cdots + w_m x_m}^{データのパラメータに対する線形結合} )$  

【表記】  

$p_i = \sigma (w_0 + w_1 x_{i1} + \cdots + w_m x_{im})$

【数式】  

$P(Y = 1\| \boldsymbol{x}) = \sigma (\overbrace{w_0}^{切片} + \overbrace{w_1}^{回帰係数} \underbrace{x_1}_{説明変数})$  
$w$は未知。学習で決める  

### 最尤推定  

+ 様々な確率分布
    + 正規分布
    + t分布
    + ガンマ分布
    + 一様分布
    + ディリクレ分布
    + :
    + ロジスティック回帰モデルではベルヌーイ分布を利用
+ ベルヌーイ分布
    + 数学において、確率pで1、確率1-pで0をとる、離散確率分布(例：コイントス)
    + **「生成されるデータ」は分布のパラメータにより異なる** (この場合は確率p)

【ベルヌーイ分布に従う確率変数Y】  

$Y \sim Be(p)$  

【Y=0とY=1になる確率をまとめて表現】  

$P(y) = p^y(1 - p)^{1 - y}$  

<br>

+ ベルヌーイ分布のパラメータの推定  
    + データからそのデータを生成したであろう尤もらしい分布(パラメータを推定したい  
        + 最尤推定
+ 同時確率
    + あるデータが得られた時、それが同時に得られる確率
    + それぞれの確率の掛け算(確率変数は独立と仮定)
+ 尤度関数
    + データは固定、パラメータを変化させる
    + 尤度関数を最大化するようなパラメータを選ぶ推定方法を**最尤推定**という

【1回の試行でy=y_1になる確率】  

$P(y) = p^y (1-p)^{1-y}$  

【n回の試行でy1~ynが同時に起こる確率(p固定)】  

$P(y_1, y_2, \cdots, y_n; p) = \prod_{i=1}^{n} \overbrace{p^{y_i}}^{p:既知}  (1 - \overbrace{p}^{既知})^{1 - y_i}$  

【y1~ynのデータが得られた際の尤度関数】  

$P( \overbrace{y_1, y_2, \cdots, y_n}^{既知} ; p) = \prod_{i=1}^{n} \overbrace{p^{y_i}}^{p:未知, y_i:既知}(1 - \overbrace{p}^{未知})^{1 - \overbrace{y_i}^{既知}}$  

<br>

+ ロジスティック回帰モデルの最尤推定
    + 確率pはシグモイド関数となるため、推定するパラメータは重みパラメータとなる
    + $(x_1, y_1), (x_2, y_2), \cdots,  (x_3, y_3)$を生成するに至ったもっとmらしいパラメータを探す

$P(Y = y_1 \| x_1) = p_1^{y_1}(1 - p_1)^{1 - y_1} = \sigma (\boldsymbol{w}^T \boldsymbol{x_1})^{y_1}(1 - \sigma ((\boldsymbol{w}^T \boldsymbol{x_1}))^{1-y_1}$  
$P(Y = y_2 \| x_2) = p_2^{y_2}(1 - p_2)^{1 - y_2} = \sigma (\boldsymbol{w}^T \boldsymbol{x_2})^{y_2}(1 - \sigma ((\boldsymbol{w}^T \boldsymbol{x_2}))^{1-y_2}$  
$\vdots$  
$P(Y = y_n \| x_n) = p_n^{y_n}(1 - p_n)^{1 - y_n} = \sigma (\boldsymbol{w}^T \boldsymbol{x_n})^{y_n}(1 - \sigma ((\boldsymbol{w}^T \boldsymbol{x_n}))^{1-y_n}$  

↑$w$は未知  

【y1~ynのデータが得られた際の尤度関数】  

尤度関数Lを最大とするパラメータを探索  

+ 確率変数が独立を仮定
    + 確率の積に分解可能
+ 尤度関数はパラメータのみに依存する関数

\\[
    \begin{split}
        P(y_1, y_2, \cdots, y_n \| w_0, w_1, \cdots, w_m) &= \prod_{i=1}^{n} p_i^{y_i} (1 - p_i)^{1 - y_i} \\\\  
        & = \prod_{i=1}^{n} \sigma (\boldsymbol{w}^T \boldsymbol{x_i})^{y_i}(1 - \sigma ((\boldsymbol{w}^T \boldsymbol{x_i}))^{1-y_i} \\\\  
        &= L( \boldsymbol{w} ) \\\\  
        ↑x, yは既知、w, pは未知  
    \end{split}
\\]

+ 尤度関数を最大とするパラメータを探す(推定)
    + 対数をとると、
        + 微分の計算が簡単
            + 同時確率の積が和に変換可能
            + 指数が積の演算に変換可能
        + 桁落ちも防げる
    + 対数尤度関数が最大になる点と尤度関数が最大になる点は同じ
        + 対数関数は単調増加
            + ある尤度の値がx1 < x2のとき、必ずlog(x1) < log(x2)となる
    + 「尤度関数にマイナスをかけたものを最小化」→「最小2乗法の最小化」と合わせる

\\[
    \begin{split}
        E(w_0, w_1, \cdots, w_m) &= - \log L(w_0, w_1, \cdots, w_m) \\\\  
        &= \sum_{i=1}^{n} \\{ y_i \log p_i + (1 - y_i) \log (1 - p_i) \\}
    \end{split}
\\]

+ 勾配降下法 (Gradient decent)
    + 参考：[機械学習にでてくる勾配降下法/勾配ベクトルなどの整理。ついでにPythonで試してみた。](https://qiita.com/masatomix/items/d4e5fb3b52fa4c92366f)
    + $\eta$(イータ)：学習率 (ハイパーパラメータ)
        + モデルのパラメータの収束しやすさを調整する
+ なぜ必要か？
    + [線形回帰モデル(最小2乗法)]：
        + MSEのパラメータに関する微分が0になる値を解析に求めることが可能
    + [ロジスティック回帰モデル(最尤法)]：
        + 対数尤度関数をパラメータで微分して0になる値を求める必要があるが、解析的にこの値を求めることは困難

\\[
    \boldsymbol{w}(k + 1) = \boldsymbol{w}^k - \eta \frac{\partial E( \boldsymbol{w} )}{\partial \boldsymbol{w}}
\\]

対数尤度関数を、係数とバイアスに関して微分  

\\[
    \begin{split}
        Loss: E(w) &= - \log L(w) \cdots 負の対数尤度(negative \space log-likelihood) \\\\  
        &= - \sum_{i = 1}^{n} \\{ \overbrace{ y_i \cdot \log P_i + (1-y_i) \log (1-p_i) }^{E_i} \\} \\\\  
        & (p_i = \overbrace{\sigma (w^T x_i)}^{z} = \frac{1}{1 + \exp (w^T x_i)}) \\\\  
        \space\\\\  
        \Longrightarrow \frac{\partial E( \boldsymbol{w} )}{\partial \boldsymbol{w}} &= - \sum_{i=1}^{n} \frac{\partial E_i}{\partial p_i} \times \frac{\partial p_i}{\partial z_i} \times \frac{\partial z_i }{\partial \boldsymbol{w}} \qquad \qquad \cdots 微分のchain \space rule \\\\  
        &= - \sum_{i=1}^{n} (\overbrace{\frac{y_i}{p_i} - \frac{1 - y_i}{1 - p_i}}^{\log の微分} ) \times \overbrace{p_i(1 - p_i)}^{sigmoidの微分}  \times \overbrace{x_i}^{w^T x_iの微分} →分母・分子でcancelできる \\\\  
        &= - \sum_{i=1}^{n} \\{ y_i \cdot (1 - p_i) - (  - y_i) \cdot p_i \\} x_i \\\\  
        &= - \sum_{i=1}^{n} \\{ y_i - p_i \\} x_i \\\\  
    \end{split}
\\]

+ パラメータが更新されなくなった場合
    + = 勾配が0
    + 少なくとも反復学習で探索した範囲では最適な解が求められた

$$
    \boldsymbol{w}^{(k + 1)} = \boldsymbol{w}^{(k)} + \eta \sum_{i=1}^{n}(y_i - p_i) \boldsymbol{x}_i
$$

+ 勾配降下法では、パラメータを更新するのにN個すべてのデータに対する和を求める必要がある
    + nが巨大となったときの問題
        + データをオンメモリに載せる容量が足りない
        + 計算時間が莫大
    + 確率的勾配降下法を利用して解決
+ 確率的勾配降下法(SGD)
    + データを一つずつランダムに選んでパラメータを更新
    + 勾配降下法でパラメータを1回更新するのと同じ計算量で、パラメータをn回更新
        + 効率よく最適解を探索可能

$$
    \boldsymbol{w} (k + 1) = \boldsymbol{w}^k + \eta (y_i - p_i) \boldsymbol{x}_i
$$

参考：  
[京都大学集中講義 機械学習と深層学習の 数理と応用 (2)](http://ibis.t.u-tokyo.ac.jp/suzuki/lecture/2018/kyoto/Kyoto_02.pdf)  

### ハンズオン  

#### タイタニックの乗客の生存予測

【課題】  

年齢が30歳で男の乗客は生き残れるか？  

![titanic]({{site.baseurl}}/images/20211015.png)  

結果：  

生き残れない(生存率：約19%)  

考察：  

+ 性別が生存率に大きく影響する
    + 男性は生存率が低く、女性は高い
+ 年齢も生存率に影響はあるが、小さい
    + 年齢が低い方が生存率が高い
    ![titanic]({{site.baseurl}}/images/20211015_1.png)  

メモ：  

+ モデルの設定をよく確認しておく必要がある
    + 例：`LogisticRegression(C=1.0, penalty='l2')` 
        + `C=1.0`: 正則化がかかっている  
        + `penalty='l2'`: L2ノルム
+ 判断に困るような確率であれば、棄却オプションも使うのもよい
+ 複数の特徴量を組み合わせるなどして、新たな特徴量を作る方法もある
    + 参考：[特徴量エンジニアリングについて6つ｜前処理と性能を高める手法](https://www.fenet.jp/infla/column/engineer/%E7%89%B9%E5%BE%B4%E9%87%8F%E3%82%A8%E3%83%B3%E3%82%B8%E3%83%8B%E3%82%A2%E3%83%AA%E3%83%B3%E3%82%B0%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A66%E3%81%A4%EF%BD%9C%E5%89%8D%E5%87%A6%E7%90%86%E3%81%A8%E6%80%A7/#%E7%89%B9%E5%BE%B4%E9%87%8F%E3%82%A8%E3%83%B3%E3%82%B8%E3%83%8B%E3%82%A2%E3%83%AA%E3%83%B3%E3%82%B0%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A63%EF%BC%9A%E6%96%B0%E3%81%97%E3%81%84%E5%A4%89%E6%95%B0%E3%82%92%E4%BD%9C%E3%82%8B%E7%89%B9%E5%BE%B4%E9%87%8F)

---

## 主成分分析  

+ 多変量データの持つ構造を、より少数個の指標に圧縮
    + 変数の個数を減らすことに伴う、情報の損失はなるべく小さく
    + 少数変数を利用した分析や可視化(2, 3次元の場合)が実装可能

【学習データ】  

$\boldsymbol{x_i} = (x_{i1}, x_{i2}, \cdots, x_{im}) \in \mathbb{R}^m$  

【平均(ベクトル)】  

$\bar x = \frac{1}{n} \sum_{i=1}^{n} x_i$  

【データ行列】  

$\bar X =  (\boldsymbol{x}_1 - \bar{ \boldsymbol{x} }, \cdots , \boldsymbol{x}_n - \bar{ \boldsymbol{x} })^T \in \mathbb{R}^{n \times m}$  

【分散共分散行列】  

$\sum = Var(\bar X) = \frac{1}{n} \bar X^T \bar X$  

【線形変換後のベクトル】  

$ \boldsymbol{s_j} = (s_{1j}, \cdots, s_{nj})^T = \bar X \boldsymbol{a_j} \qquad \boldsymbol{a_j} \in \mathbb{R}^m$  

(jは射影軸のインデックス)  

+ 係数ベクトルが変われば**線形変換後の値**が変化
    + 情報の量を分散の大きさと捉える
    + 線形変換後の変数の**分散が最大**となる射影軸を探索

$$
    \boldsymbol{s}_j = (s_{1j}, \cdots, s_{nj})^T = \bar X \boldsymbol{a}_j \qquad \boldsymbol{a}_j \in \mathbb{R}^m
$$

[線形変換後の分散]  

$$
    Var(s_j) = \frac{1}{n} (\bar X \boldsymbol{a}_j)^T (\bar X \boldsymbol{a}_j) = \frac{1}{n} \boldsymbol{a}_j^T \bar X \bar X \boldsymbol{a}_j = \boldsymbol{a}_j^T Var(\bar X) \boldsymbol{a}_j
$$

+ 以下の制約付き最適化問題を解く
    + ノルムが1となる制約を入れる (制約がないと解が無限にある)

【目的関数】  

$\arg \max \boldsymbol{a}_j^T Var (\bar X) \boldsymbol{a}_j$  

【制約条件】  

$\boldsymbol{a}_j^T \boldsymbol{a}_j = 1$  

+ 制約付き最適化問題の解き方
    + ラグランジュ関数を最大にする係数ベクトルを探索 (微分して0になる点)

【ラグランジュ関数】  

$E(\boldsymbol{a}_j) = \overbrace{\boldsymbol{a}_j^T Var (\bar X) \boldsymbol{a}_j}^{目的関数}  - \overbrace{\lambda}^{ラグランジュ係数} (\overbrace{\boldsymbol{a}_j^T \boldsymbol{a}_j - 1}^{制約条件} )$

### ハンズオン  

#### 乳がん検査データ

【課題】  

32次元のデータを2次元上に次元圧縮した際に、うまく判別できるかを確認    

![cancer]({{site.baseurl}}/images/20211020.png)  

結果:  

+ 次元圧縮したときのscoreは高い精度を保っており、うまく判別できている
    + ロジスティック回帰による予測に比べると、精度はわずかに下がった
    + 混同行列を見ると、5つのデータがTPからFNに変わっている

---

## k近傍法  

+ 分類問題のための機械学習手法
    + 最近傍のデータをk個とってきて、それらが最も多く所属するクラスに識別
    + kを変えると結果も変わる
    + kを大きくすると決定境界はなめらかになる

### ハンズオン  

【課題】人口データと分類結果をプロットしてください  

結果：  

![n_neighbors_3]({{site.baseurl}}/images/20211020_1.png)  
![n_neighbors_5]({{site.baseurl}}/images/20211020_2.png)  
![n_neighbors_15]({{site.baseurl}}/images/20211020_3.png)  

`n_neighbors`の値を大きくするほど、境界がなめらかになる  

<br>

---

## k-means

+ 教師なし学習
+ クラスタリング手法
+ 与えられたデータをk個のクラスタに分類
+ 中心の初期値が変わると結果の変わりうる
    + 初期値が近いとうまくクラスタリングできないことも

【アルゴリズム】  

1. 各クラスタ中心の初期値を設定
1. 各データ点に対して、各クラスタ中心との距離を計算し、最も距離が近いクラスタを割り当てる
1. 各クラスタの平均ベクトル(中心)を計算する
1. クラスタの再割当てと中心の更新を、収束するまで繰り返す

参考：機械学習アルゴリズム辞典

### ハンズオン  

k-meansにてsyntheticデータと分類結果をプロット  

![n_clusters:2]({{site.baseurl}}/images/20211020_4.png)  
![n_clusters:3]({{site.baseurl}}/images/20211020_5.png)  
![n_clusters:5]({{site.baseurl}}/images/20211020_6.png)  
![n_clusters:6]({{site.baseurl}}/images/20211020_7.png)  

---

## SVM  

### 2クラス分類

+ 与えられた入力データが2つのカテゴリーのどちらに属するかを識別する

### 決定関数と分類境界

+ **決定関数(decision function)**
    + 特徴ベクトルxがどちらのクラスに属するか判定するための関数  
        + 一般に2クラス分類では $f(x) = \boldsymbol{w}^T \boldsymbol{x} + b$
    + ある入力データxに対して決定関数$f(x)$を計算し、その符号により2つのクラスに分類  
    ($\mathrm{sgn}(\cdot)$は符号関数)

$$
    y = \mathrm{sgn} f(x) = \left\{
    \begin{array}{ll}
        +1 & (f(x) > 0) \\
        -1 & (f(x) < 0)
    \end{array}
    \right.
$$

+ **分類境界(classi cation boundary)**
    + 特徴ベクトルを2つのクラスに分ける境界

### 線形サポートベクトル分類(ハードマージン)

+ 分離可能性を仮定したSV分類のこと
+ **サポートベクトル**
    + 分類境界に最も近いデータ$x_i$
+ **マージン**
    + 分類境界を挟んで2つのクラスがどのくらい離れているか
+ **マージン最大化(margin maximization)**
    + なるべく大きなマージンを持つ分類境界を探す
+ **分離可能(separable)**
    + n個の訓練データを全て正しく分類できるwとbの組が存在する場合、訓練データは決定関数$f(x)$により<u>分離可能(separable)</u>と表現する
+ 分類境界$f(x) = 0$と$x_i$との距離  

$$
    \overbrace{
        \frac{|f(x_i)|}{|| \boldsymbol{w} ||} = \frac{| \boldsymbol{w}^T \boldsymbol{x}_i + b|}{|| \boldsymbol{w} ||}
    }^{Hesseの公式}
        = \frac{y_i[ \boldsymbol{w}^T \boldsymbol{x}_i + b ]}{|| \boldsymbol{w} ||} \\  
    \Longrightarrow \mathrm{min}_i \frac{y_i[ \boldsymbol{w}^T \boldsymbol{x}_i + b ]}{|| \boldsymbol{w} ||} = \frac{M( \boldsymbol{w}, b )}{|| \boldsymbol{w} ||} \\  
    \space \\  
    || \boldsymbol{w} || = \sqrt{w_1^2 + w_2^2 + \cdots + w_n^2} (\boldsymbol{w}のL2ノルム)
$$

+ SVMの目的関数  

    $$
        \mathrm{max}_{w, b} 
        \biggl[ 
            \mathrm{min}_i \frac{y_i[ \boldsymbol{w}^T \boldsymbol{x}_i + b ]}{|| \boldsymbol{w} ||}
        \biggr]
        = \mathrm{max}_{w, b} \frac{M( \boldsymbol{w}, b )}{|| \boldsymbol{w} ||} \tag{1}
    $$

+ **制約条件(constrain)**

    $$
        \mathrm{min}_i 
        \Bigl[ 
            y_i[ \boldsymbol{w}^T \boldsymbol{x}_i + b ]
        \Bigr]
        = M( \boldsymbol{w}, b ) 
        \Longleftrightarrow 全ての i に対して y_i[ \boldsymbol{w}^T \boldsymbol{x}_i + b ] \geq M( \boldsymbol{w}, b ) \tag{2}
    $$

<br>

【式(1)と(2)を簡素化する】  

$$
    \tilde{ \boldsymbol{w} } = \frac{ \boldsymbol{w} }{M(\boldsymbol{w}, b)}, \qquad \tilde{ b } = \frac{b}{M(\boldsymbol{w}, b)}
$$

とすると、  

$$
    \begin{split}
        &\mathrm{min}_i 
        \Bigl[ 
            y_i[ \tilde{\boldsymbol{w}}^T \boldsymbol{x}_i + \tilde{b} ]
        \Bigr]
        = 1 
        \Longleftrightarrow 全ての i に対して y_i[ \tilde{\boldsymbol{w}}^T \boldsymbol{x}_i + \tilde{b} ] \geq 1  \\  
        &\mathrm{max}_{\tilde{w}, \tilde{b}} \frac{1}{|| \tilde{\boldsymbol{w}} ||}
    \end{split}
$$

$\frac{1}{\|\| \tilde{ \boldsymbol{w} } \|\|}$の最大化は  
+ $\|\| \tilde{ \boldsymbol{w} } \|\|$の最大化と等価
+ $\frac{1}{2}\|\| \tilde{ \boldsymbol{w} } \|\|^2$の最小化と等価

なので、  

$$
    \mathrm{min}_{w, b} \frac{1}{2} ||\boldsymbol{w}||^2, \quad ただし、全てのiに対して y_i[\boldsymbol{w}^T x_i + b] \geq 1 \tag{3}
$$

(式(3)以降、$\tilde{w}, \tilde{b}$を単に$w, b$と表す。表記の簡単のため)  

<br>

### 線形サポートベクトル分類(ソフトマージン)

+ SV分類を分離可能でないデータに適用できるように拡張  
式(3)の条件を、次のように緩和  

$$
    y_i[\boldsymbol{w}^T x_i + b] \geq 1 - \xi_i \quad (i = 1, \cdots, n) \\  
    \underbrace{\xi_i}_{グザイ / クサイ / クシー}: スラック変数(\mathrm{slack \space variable})
$$

+ $f(x) = 1$と$f(x) = -1$の間の距離をマージンと解釈
+ マージンを最大化、分類の誤差$\xi_i$を最小化する
    

$$
    \begin{split}
        &\mathrm{min}_{w, b, \xi}
        \Bigl[
            \overbrace{
                \frac{1}{2}||w||^2
            }^{マージン最大化}
            + 
            \overbrace{
                C \sum_{i=1}^{n} \xi_i
            }^{誤差最小化}
        \Bigr] \\  
        &ただし \space y_i[\boldsymbol{w}^T x_i + b] \geq 1 - \xi_i, \quad \xi_i \geq 0 \quad (i = 1, \cdots, n) \\  
        &\qquad \qquad \boldsymbol{\xi} = {\xi_1, \cdots, \xi_n}^T 
    \end{split} 
    \tag{4}
$$

+ **正則化係数 (regularization parameter)** $C$
    + 正の定数、ハイパーパラメータ
    + 大きいほどハードマージンに近づく
        + 大きすぎると分離境界に対して目的関数が発散して計算できなくなる
    + 小さいほど誤分類を許容する
    + **交差検証法(cross validation)** などで決める

### SVMにおける双対(そうつい)表現

+ **主問題**
    + 式(3), (4)のような最適化問題
+ **双対問題(dual problem)**
    + ラグランジュ関数に関する最適化問題

    $$
        \max_{\alpha, \mu} \min_{\boldsymbol{w}, b, \boldsymbol{\xi}} L(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) \tag{5}
    $$

    + メリット
        + 主問題より変数を少なくできる
        + 分類境界の非線形化を考える際、双対形式の方が有利

+ 補足:
    + **勾配(gradient)**

        $$
            \left(
                \begin{array}{c}
                    \partial_{x_1} f(\boldsymbol{x}) \\  
                    \vdots \\  
                    \partial_{x_n} f(\boldsymbol{x})
                \end{array}
            \right)
            \equiv \frac{\partial}{\partial \boldsymbol{x}} f(\boldsymbol{x}) = \partial_x f(\boldsymbol{x}) = \overbrace{\nabla}^{nabla}_x f(\boldsymbol{x})
        $$

        + 関数$f(\boldsymbol{x})$の停留値$\boldsymbol{x}^*$

        $$
            \frac{\partial}{\partial \boldsymbol{x}} f(\boldsymbol{x}) | _{x = x^*} = \boldsymbol{0} \\  
            \space \\  
            (0はn次元のゼロベクトル)
        $$
        + **停留点(Stationary point)**
            + 関数の微分係数が0となる点
        + **鞍点(saddle point)**
            + ある方向から見ると極小点
            + その直交する方向から見ると極大点

### 双対表現の導出

ソフトマージンの場合の最適化問題の双対問題を考える  

ラグランジュ関数は  

$$
    L(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) = \frac{1}{2}||\boldsymbol{w}||^2 + C \sum_{i=1}^{n} \xi_i - \sum_{i=1}^{n} \alpha_i 
    \Bigl[
            y_i[\boldsymbol{w}^T \boldsymbol{x}_i + b] - 1 + \xi_i
    \Bigr]
    - \sum_{i=1}^{n} \mu_i \xi_i \tag{6}
$$

$$
    \boldsymbol{\alpha} = (\alpha_1, \cdots, \alpha_n)^T,
    \quad
    \boldsymbol{\mu} = (\mu_1, \cdots, \mu_n)^T \\  
    \alpha_i \geq 0, \space \mu_i \geq 0 \space (i = 1, \cdots, n)
$$

+ **主変数(primal variable)**
    + $\boldsymbol{w}, b, \boldsymbol{\xi}$
    + 双対問題の場合、制約条件は課されていない
+ **双対変数(dual variable)**
    + $\boldsymbol{\alpha}, \boldsymbol{\mu}$

式(5)をより簡素に書き換える  

1. ラグランジュ関数を主変数に関して最小化  
    以下の連立方程式を満たす $\boldsymbol{w}^\*, b^\*, \boldsymbol{\xi}^\*$ が、主変数に関する最適化問題の最適解  

    $$
        \begin{split}
            \frac{\partial}{\partial \boldsymbol{w}} L(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) | _{\boldsymbol{w} = \boldsymbol{w}^*} &= \boldsymbol{w}^* - \sum_{i=1}^{n} \alpha_i y_i \boldsymbol{x}_i = \boldsymbol{0} \\  
            \frac{\partial}{\partial b} L(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) | _{b = b^*} &= - \sum_{i=1}^{n} \alpha_i y_i = 0 \\  
            \frac{\partial}{\partial \xi_i} L(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) | _{\xi_i = \xi_i^*} &= C -  \alpha_i - \mu_i = 0 \quad (i = 1, \cdots, n) \\  
        \end{split}  
        \tag{7} 
    $$

    この最適解($\boldsymbol{w}^\*, b^\*, \boldsymbol{\xi}^\*$)を用いて双対変数を書き直すと  

    $$
        \max_{\alpha, \mu} \min_{\boldsymbol{w}, b, \boldsymbol{\xi}} L(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) = 
        \max_{\alpha, \mu} L(\boldsymbol{w}^*, b^*, \boldsymbol{\xi}^*, \boldsymbol{\alpha}, \boldsymbol{\mu})
    $$
1. 双対変数$\boldsymbol{\alpha}, \boldsymbol{\mu}$に関する最大化問題を考える  
    式(7)を式(6)に代入して整理
    
    $$
        \begin{split}
            L(\boldsymbol{w}^*, b^*, \boldsymbol{\xi}^*, \boldsymbol{\alpha}, \boldsymbol{\mu}) &= \frac{1}{2} \boldsymbol{w}^{*T} \boldsymbol{w} - \sum_{i=1}^{n} \alpha_i y_i \boldsymbol{w}^{*T} \boldsymbol{x}_i - b^*  \sum_{i=1}^{n} \alpha_i y_i + \sum_{i=1}^{n} \alpha_i + \sum_{i=1}^{n} [C - \alpha_i - \mu_i] \xi_i^* \\  
            &= \frac{1}{2}
            \Biggl(
                \sum_{i=1}^{n} \alpha_i y_i \boldsymbol{x}_i
            \Biggr)^T
            \Biggl(
                \sum_{j=1}^{n} \alpha_j y_j \boldsymbol{x}_j
            \Biggr)
            - \sum_{i=1}^{n} \alpha_i y_i 
            \Biggl(
                \sum_{j=1}^{n} \alpha_j y_j \boldsymbol{x}_j
            \Biggr)^T
            \boldsymbol{x}_i + \sum_{i=1}^{n} \alpha_i \\  
            &= \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \boldsymbol{x}_i^T \boldsymbol{x}_j 
            - \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \boldsymbol{x}_i^T \boldsymbol{x}_j
            + \sum_{i=1}^{n} \alpha_i \\  
            &= - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \boldsymbol{x}_i^T \boldsymbol{x}_j + \sum_{i=1}^{n} \alpha_i
        \end{split}
    $$

    1. ラグランジュ関数を展開して書き換え
    1. 以下を代入 (式(7)より)  

        $$
            \boldsymbol{w}^* = \sum_{i=1}^{n} \alpha_i y_i \boldsymbol{x}_i
        $$  

        3項目と5項目は0 (式(7)より)
    1. 以下の性質を使用  

        $$
            \begin{split}
                &(\alpha_i y_i \boldsymbol{X})^T = \alpha_i y_i \boldsymbol{X}^T \\  
                &\sum_{i=1}^{n} \alpha_i y_i \boldsymbol{x}_i^T \sum_{j=1}^{n} \alpha_j y_j \boldsymbol{x}_j = \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \boldsymbol{x}_i^T \boldsymbol{x}_j
            \end{split}
        $$
        
    1. 


参考：  

+ ラグランジュの未定乗数法