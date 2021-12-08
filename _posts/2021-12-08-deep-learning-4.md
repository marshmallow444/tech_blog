---
layout: post_toc
title: "【ラビット・チャレンジ】深層学習 後編 Day4"
tags: ラビット・チャレンジ E資格 機械学習
---

<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

[ラビット・チャレンジ](https://ai999.careers/rabbit/)の受講レポート。  

---  

# 強化学習

## 強化学習とは

長期的に報酬を最大化できるように環境の中で行動を選択できるエージェントを  
作ることを目標とする機械学習の一分野  
→行動の結果として与えられる報酬をもとに、  
　行動を決定する原理を改善していく仕組み

不完全な知識を元に行動しながら、データを収集  
最適な行動を見つけていく  

関数近似法とQ学習を組み合わせる  
+ **Q学習**
    + 行動価値関数を、行動するごとに更新することにより学習を進める方法
+ **関数近似法**
    + 価値関数や方策関数を関数近似する手法

## 強化学習の応用例

マーケティング  
キャンペーンのお知らせを、どの顧客へ送るか？  
なるべくコストは低く、売上は高くなるように選ぶ  

## 探索と利用のトレードオフ

[探索が足りない状態]  
　　↑  
トレードオフの関係性  
　　↓  
[利用が足りない状態]  

## 強化学習のイメージ

![reinforcement learning]({{site.baseurl}}/images/20211208.drawio.png)  

+ 方策関数: $\pi (s, a)$
+ 行動価値関数: $Q(s, a)$

## 強化学習の差分

目標が違う

+ 教師あり/なし学習
    + データに含まれるパターンを見つけ出す
    + データから予測する
+ 強化学習
    + 優れた方策を見つける

## 行動価値関数

+ 価値関数
    + 状態価値関数
        + ある状態の価値に注目
    + **行動価値関数**
        + 状態と価値を組み合わせた価値に注目

## 方策関数

方策ベースの強化学習手法において、  
ある状態でどのような行動をとるのかの確率を与える関数  

$$
    \pi(s) = a
$$

関数の関係  
エージェントは方策に基づいて行動する  
+ $\pi(s,a)$: VやQを基にどういう行動をとるか
    + 経験を活かす / チャレンジする など  

$\Rightarrow$その瞬間、その瞬間の行動をどうするか

+ $V^{\pi}(s)$: 状態関数
    + ゴールまで今の方策を続けた時の報酬の予測値が得られる
+ $Q^{\pi}(s)$: 状態 + 行動関数
    + ゴールまで今の方策を続けた時の報酬の予測値が得られる

$\Rightarrow$やり続けたら最終的にどうなるか  


## 方策勾配法

方策をモデル化して最適化する手法  
関数はNNにできる = 学習できる  
→方策関数をNNとして学修させる  

$$
    \pi(s, a | \theta) \\  
    \space\\  
    \underbrace{\theta^{(t+1)}}_{重み} = \theta^{(t)} + \epsilon \nabla \underbrace{J(\theta)}_{*1} \\  
$$

+ *1: ここでは期待収益 (NNでは誤差関数)
    + NNでは誤差を"小さく"
    + 強化学習では期待収益を"大きく"
+ $J$: 方策の良さ。定義しなければならない  
+ $\theta$: 方策関数のパラメータ  

定義方法
+ 平均報酬
+ 割引報酬和

この定義に対応して、行動価値関数Q(s, a)の定義を行う  
方策勾配定理が成り立つ  

$$
    \nabla_\theta J(\theta) = \mathbb{E}_{\pi \theta}[(\nabla_{\theta} \log \pi_{\theta}(a | s) Q^{\pi}(s, a))]
$$

元の式：
$$
    \nabla_\theta J(\theta) = \nabla_{\theta} \underbrace{\sum_{a \in A}}_{*2} \underbrace{\pi_{\theta}(a|s) Q^{\pi}(s, a)}_{ある行動をとるときの報酬}
$$

*2 ＝ すべての行動パターンに対して全部足す  

# Alpha Go

【Alpha Go Lee】

**Alpha Go LeeのPolicyNet**

![Alpha Go Lee]({{site.baseurl}}/images/20211208_1.drawio.png)  

出力：19 * 19マスの着手予想確率  

+ 現在のPolicyNetとPolicyPoolからランダムに選択されたPolicyNetと対極シミュレーションを行う  
+ その結果を用いて方策勾配法で学習  
+ PolicyPool: PolicyNetの強化学習の過程を500iterationごとに記録し保存したもの  
→対局に幅をもたせ、過学習を防ぐため


**Alpha Go LeeのValueNet**

![Alpha Go Lee]({{site.baseurl}}/images/20211208_2.drawio.png)  

出力：現局面の勝率を-1~1で表したもの  

+ PolicyNetを使用して対局シミュレーションを行う
+ その結果の勝敗を教師として学習
+ 教師データの作成手順:
    1. SL PolicyNetでN手まで打つ
        + SL PolicyNet: 教師あり学習で作成したPolicyNet
    1. N+1手目の手をランダムに選択、その手で進めた局面をS(N+1)とする
    1. S(N+1)からRL PolicyNetで終局まで打ち、その勝敗報酬をRとする
        + RL PolicyNet: 強化学習で作成したPolicyNet
+ S(N+1)とRを教師データ対、損失関数を平均二乗誤差とし、回帰問題として学習
+ N手までとN+1手からのPolicyNetを別にしてある理由：過学習を防ぐため

PolicyNet, ValueNetの入力(表(a))  

[![input](https://cz-cdn.shoeisha.jp/static/images/article/10952/10952_03.jpg)](https://cz-cdn.shoeisha.jp/static/images/article/10952/10952_03.jpg)  
(画像：[https://codezine.jp/article/detail/10952](https://codezine.jp/article/detail/10952))  

ValueNetの入力には、  
「手番」(現在の手番が黒番であるか？)が1ch追加される  

【Alpha Go Zero】  

【Alpha Goの学習】  
以下のステップで行われる  

[![learning](https://livedoor.blogimg.jp/lunarmodule7/imgs/f/9/f9dba059.png)](https://livedoor.blogimg.jp/lunarmodule7/imgs/f/9/f9dba059.png)  
(画像：[http://blog.livedoor.jp/lunarmodule7/archives/4635352.html](http://blog.livedoor.jp/lunarmodule7/archives/4635352.html))  

1. 教師あり学習によるRollOutPolicyとPolicyNetの学習
1. 強化学習によるPolicyNetの学習
1. 強化学習によるValueNetの学習

**RollOutPolicy**  

NNではなく線形の方策関数  
探索中、高速に着手確率を出すために使用  

[![RollOutPolicy](https://ichi.pro/assets/images/max/724/1*ZUa3KYLqB8Cm7YlSOVrwpQ.png)](https://ichi.pro/assets/images/max/724/1*ZUa3KYLqB8Cm7YlSOVrwpQ.png)  
(画像：[https://ichi.pro/alphago-gijutsuteki-ni-dono-yoni-kinoshimasu-ka-42734052725529](https://ichi.pro/alphago-gijutsuteki-ni-dono-yoni-kinoshimasu-ka-42734052725529))  

# 軽量化・高速化技術

## モデル並列

## データ並列

## GPU

## 量子化

## 蒸留

## プルーニング

# 応用技術

試験によく出る！

## MobileNet

## DenseNet

## Layer正規化 / Instance正規化

## WaveNet

# 例題解説

# Appendix

# 実装演習