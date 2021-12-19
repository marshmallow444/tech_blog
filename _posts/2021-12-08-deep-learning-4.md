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

## Alpha Go Lee

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

## Alpha Go Zero  

AlphaGo Leeとの違い  
+ 教師あり学習を一切行わず、強化学習のみで作成
+ 特徴入力からヒューリスティックな要素を排除、石の配置のみにした
+ PolicyNetとValueNetを一つのネットワークに統合した
+ Residual Netを導入した
+ モンテカルロ木探索からRollOutシミュレーションをなくした

**Alpha Go ZeroのPolicyValueNet**  

![PolicyValueNet]({{site.baseurl}}/images/20211209.drawio.png)  

**Residual Network**  

[![Residual Networkの基本構造](https://i.stack.imgur.com/TfMe9.png)](https://i.stack.imgur.com/TfMe9.png)  
(画像：[https://stackoverflow.com/questions/49293450/why-each-block-in-deep-residual-network-has-two-convolutional-layers-instead-of](https://stackoverflow.com/questions/49293450/why-each-block-in-deep-residual-network-has-two-convolutional-layers-instead-of))  

+ ネットワークにショートカットを追加
    + 勾配の爆発・消失を抑える
+ 100層を超えるネットワークでの安定した学習を可能にした
+ 層数の違うネットワークのアンサンブル効果が得られている、という説も
+ 派生形
    + Residual Blockの工夫
        + Bottleneck
            + 1 * 1カーネルのConvolutionを使用
            + 1層目で次元削減、3層目で次元復元の3層構造
            + メリット：計算量は2層とほぼ同じままで1層増やせる
        + PreActivation
            + ResidualBlockの並びを替えたことで性能上昇
                + BatchNorm→ReLU→Convolution→BatchNorm→ReLU→Convolution→Add
    + Network構造の工夫
        + WideResNet
            + ConvolutionのFilter数をk倍にしたResNet
            + 1倍→k倍xブロック→2*k倍yブロック、、と段階的に幅を増やす
                + 浅い層数でも深い層数のものと同等以上の精度
                + GPUを効率的に使用でき、学習が早い
        + PyramidNet
            + 段階的にではなく、角層でFilter数を増やしていくResNet
                + WideResNetで幅が広がった直後の層に過度の負担←精度が落ちる原因
    + (E資格にこういうのが出がち。名前と説明の組み合わせを問う)

## Alpha Goの学習  
以下のステップで行われる  

[![learning](https://livedoor.blogimg.jp/lunarmodule7/imgs/f/9/f9dba059.png)](https://livedoor.blogimg.jp/lunarmodule7/imgs/f/9/f9dba059.png)  
(画像：[http://blog.livedoor.jp/lunarmodule7/archives/4635352.html](http://blog.livedoor.jp/lunarmodule7/archives/4635352.html))  

1. 教師あり学習によるRollOutPolicyとPolicyNetの学習
    + 人間の打った棋譜データを教師として学習
1. 強化学習によるPolicyNetの学習
1. 強化学習によるValueNetの学習

+ [モンテカルロ木探索](https://jsai.ixsq.nii.ac.jp/ej/index.php?action=pages_view_main&active_action=repository_action_common_download&item_id=9431&item_no=1&attribute_id=1&file_no=1&page_id=13&block_id=23)
    + 強化学習の手法
    + Q関数の更新に使う手法

**RollOutPolicy**  

NNではなく線形の方策関数  
探索中、高速に着手確率を出すために使用  

[![RollOutPolicy](https://ichi.pro/assets/images/max/724/1*ZUa3KYLqB8Cm7YlSOVrwpQ.png)](https://ichi.pro/assets/images/max/724/1*ZUa3KYLqB8Cm7YlSOVrwpQ.png)  
(画像：[https://ichi.pro/alphago-gijutsuteki-ni-dono-yoni-kinoshimasu-ka-42734052725529](https://ichi.pro/alphago-gijutsuteki-ni-dono-yoni-kinoshimasu-ka-42734052725529))  

# 軽量化・高速化技術

+ 高速化
    + モデル並列化
    + データ並列化
    + GPU
+ 軽量化
    + 量子化
    + 蒸留
    + プルーニング
+ 分散深層学習
    + 複数の計算資源(ワーカー)を使用し、並列的にNNを構成することで効率の良い学習を行いたい

## モデル並列

+ 親モデルを各ワーカーに分割し、それぞれのモデルを学習させる
    + 処理の分岐している部分を複数ワーカーに分けるのが主流
+ 全てのデータで学習が終わったら、一つのモデルに復元
+ モデルが大きい場合はモデル並列化がよい  
  データが大きい場合はデータ並列化がよい
+ 1台のPCで複数GPUを使う場合が多い
    + 最後に各ワーカーの出した結果を集める際、比較的高速にできる
+ モデルのパラメータが多いほど、スピードアップの効率も向上

## データ並列

+ 親モデルを各ワーカーに子モデルとしてコピー
+ データを分割し、各ワーカーごとに計算させる
+ 複数台のPCを使う場合が多い

例：
+ 複数台のPCを並列に動かして計算させる
+ 演算器を増やして、1台のPCで複数ワーカーを動かす
+ 暇しているスマホをワーカーに使う

**【同期型と非同期型】**  

+ 各モデルのパラメータの合わせ方で、どちらかが決まる
    + **同期型**
        + 各ワーカーが計算が終わるのを待つ
        + 全ワーカーの勾配が出たところで勾配の平均を計算し、親モデルのパラメータを更新
        + 
    + **非同期型**
        + 各ワーカーはお互いの計算を待たない
        + 学習が終わった子モデルはパラメータサーバへpushされる
        + 新たに学習を始める時は、パラメータサーバからpopしたモデルに対して学習
+ 比較
    + 同期型：精度が良い。主流
        + 全ワーカーを自由に制御できる場合
    + 非同期型：処理は同期型より早いが、学習が不安定になりやすい
        + Stale Gradient Problem
        + 全ワーカーを自由に制御できない場合(スマホをワーカーにするときなど)

【参照論文】  
+ [Large Scale Distributed Deep Networks](https://proceedings.neurips.cc/paper/2012/file/6aca97005c68f1206823815f66102863-Paper.pdf)
    + TensorFlowの前身といわれる
    + 並列コンピューティングを用いることで大規模なネットワークを高速に学習させる仕組みを提案
    + 主にモデル並列とデータ並列(非同期型)の提案

## GPU

+ GPGPU (General-purpose on GPU)
    + グラフィック以外の用途で使用されるGPUの総称
+ CPU
    + 高性能なコアが少数
    + 複雑で連続的な処理が得意
+ GPU
    + 比較的低性能なコアが多数
    + **簡単な並列処理が得意**
    + NNの学習は単純な行列計算が多いので、高速化が可能

【GPGPU開発環境】  
+ CUDA (DLではほぼこれが使われる)
    + GPU上で並列コンピューティングを行うためのプラットフォーム
    + NVIDIA社が開発しているGPUでのみ使用可能
    + Deep Learning用に提供、使いやすい
+ OpenCL
    + オープンな並列コンピューティングのプラットフォーム
    + NVIDIA社以外の会社(Intel, AMD, ARMなど)のGPUからでも使用可能
    + Deep Learning用の計算に特化してはいない
+  Deep Learningフレームワーク(Tensorflow, Pytorch)内で実装されている。使用時は指定すれば良い

## 量子化 (Quantization)

通常のパラメータの64bit浮動小数点を32bitなど下位の精度に落とす  
→メモリと演算処理を削減  

+ 利点
    + 計算の高速化
    + 省メモリ化
+ 欠点
    + 精度の低下

【精度の低下】  

ほとんどのモデルでは半精度(16bit)で十分  
実際の問題では、倍精度を単精度にしてもほぼ精度が変わらない  

+ 64bit: 倍精度
+ 32bit: 単精度
+ 16bit: 半精度
+ FLOPS: 小数の計算を1秒間に何回行えるか？の単位 (floating operations)
    + 現在のGPUは、16bitで~150TeraFLOPSくらいの性能

## 蒸留

精度の高いモデルはニューロンの規模が大きい  
→規模の大きなモデルの知識を使い、軽量なモデルの作成を行う  

精度の高いモデル→ **知識の継承**  →軽量なモデル  

+ **教師モデル**
    + 予測精度の高い、複雑なモデルやアンサンブルされたモデル
+ **生徒モデル**
    + 教師モデルをもとに作られる軽量なモデル

[![distillation](https://assets.st-note.com/production/uploads/images/9000535/picture_pc_f51221ab67a49133bd41bfa915ff1a12.jpg?width=800)](https://assets.st-note.com/production/uploads/images/9000535/picture_pc_f51221ab67a49133bd41bfa915ff1a12.jpg?width=800)  
(画像：[https://note.com/imaimai/n/nae4bc0776c74](https://note.com/imaimai/n/nae4bc0776c74))  

+ 教師モデルの重みを固定
+ 生徒モデルの重みを更新していく
    + 教師モデルと生徒モデルそれぞれの誤差を使う

<br>

+ 利点
    + 少ない学習回数で精度の良いモデルを生成できる

## プルーニング

モデルの精度に寄与が少ないニューロンを削減  
→モデルの軽量化、高速化  

[![pruning](https://cocon-corporation.com/wp-content/uploads/2019/02/pruning_neuron.png)](https://cocon-corporation.com/wp-content/uploads/2019/02/pruning_neuron.png)  
(画像：[https://cocon-corporation.com/cocontoco/pruning-neural-network/](https://cocon-corporation.com/cocontoco/pruning-neural-network/))  

重みが閾値以下の場合ニューロンを削減し、再学習を行う  

参考：[ニューラルネットワークの全結合層における
パラメータ削減手法の比較](https://db-event.jpn.org/deim2017/papers/62.pdf)  

# 応用技術

ネットワークの特徴が試験によく出る！

## MobileNet

[![MobileNet](https://iq.opengenus.org/content/images/2021/08/MobileNet-V1-1.png)](https://iq.opengenus.org/content/images/2021/08/MobileNet-V1-1.png)  
(画像：[https://iq.opengenus.org/ssd-mobilenet-v1-architecture/](https://iq.opengenus.org/ssd-mobilenet-v1-architecture/))

+ 提案手法
    + ディープラーニングモデルの軽量化・高速化・高精度化を実現
+ 画像認識モデル
    + 以下の組み合わせで軽量化を実現  
    (**Depthwise Separatable Convolution**)
        + **Depthwise Convolution**
            + 仕組み
                + 入力チャネルごとに畳み込みを実施
                + 出力マップをそれらと結合
                + フィルタ数(M): 1
            + 特徴
                + 計算量が大幅に削減可能
                    + 通常の畳み込みカーネルは全ての層にかかる
                    + 出力マップの計算量：$H \times W \times C \times K \times K$
                + 層間の関係性は全く考慮されない
                    + PW畳み込みとセットで使うことで解決
        + **Pointwise Convolution**
            + 仕組み
                + 入力マップのポイントごとに畳み込みを実施
                    + 1 x 1 conv とも呼ばれる
                + 出力マップ(チャネル数)はフィルタ数分だけ作成可能
                    + 任意のサイズが指定可能
            + 特徴
                + 出力マップの計算量：$H \times W \times C \times M$
+ [https://qiita.com/HiromuMasuda0228/items/7dd0b764804d2aa199e4](https://qiita.com/HiromuMasuda0228/items/7dd0b764804d2aa199e4)
+ 論文
    + [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
+ (参考)一般的な畳み込みレイヤー
    + ストライド1でパディングを適用した場合の計算量
        + $H(eight) \times W(eight) \times K(ernel) \times K(ernel) \times C(hannel) \times M(ap)$

## DenseNet

[![DenseNet](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-20_at_11.35.53_PM_KroVKVL.png)](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-20_at_11.35.53_PM_KroVKVL.png)  
(画像：[https://paperswithcode.com/method/densenet](https://paperswithcode.com/method/densenet))  

+ 概要
    + 画像認識モデル
    + CNNの一種
    + 前方の層から後方の層へアイデンティティ接続を介してパスを作り、NNの層が深くなっても学習できるようにした
    + Dense Blockというモジュールを使用
+ 構造
    + 初期の畳み込み
    + Denseブロック
        + 出力層に前の層の入力を足し合わせる
            + 層間の情報の伝達を最大にするために、全ての同特徴量サイズの層を結合する
            [![Dense Block](https://ichi.pro/assets/images/max/724/1*9ysRPSExk0KvXR0AhNnlAA.gif)](https://ichi.pro/assets/images/max/724/1*9ysRPSExk0KvXR0AhNnlAA.gif)  
            (画像：[https://ichi.pro/rebyu-densenet-komitsudo-tatamikomi-nettowa-ku-gazo-bunrui-200536763594755](https://ichi.pro/rebyu-densenet-komitsudo-tatamikomi-nettowa-ku-gazo-bunrui-200536763594755))
        + 特徴マップの入力に対し、下記の処理で出力を計算
            + Batch正規化
            + ReLU関数による変換
            + 3 x 3畳み込み層による処理
        + この出力に入力特徴マップを足し合わせる
        + 第$l$層の出力:
            + $x_l = H_l([x_0, x_1, \cdots, x_{l-1}])$
        + $k$(チャネル数): ネットワークの<u>Growth Rate</u>
        + 各ブロック内で特徴マップのサイズは同じ
    + 変換レイヤー
        + 2つのDense Blockの間にCNNとPoolingを行う
        + CNNの中間層でチャネルサイズを変更
        + 特徴マップのサイズを変更し、ダウンサンプリングを行う
    + 判別レイヤー
+ 特徴
    + DenseNetとResNetとの違い
        + DenseNet：前方の各層からの出力全てが後方の層への入力として用いられる
        + ResNet：前1層の出力のみ、後方の層へ入力
    + Dense BlockにGrowth Rateというハイパーパラメータ
+ 論文
    + [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
    + [https://www.slideshare.net/harmonylab/densely-connected-convolutional-networks](https://www.slideshare.net/harmonylab/densely-connected-convolutional-networks)


## Layer正規化 / Instance正規化

+ Batch Norm
    + ミニバッチに含まれるsampleの同一チャネルが同一分布に従うよう正規化  
        (全ての画像, 一つのch)
+ Layer Norm
    + それぞれのsampleの全てのpixelsが同一分布に従うよう正規化  
        (一つの画像、全てのch)
+ Instance Norm
    + さらにchannelも同一分布に従うよう正規化  
        (一つの画像、一つのch)

[![Normalization](https://miro.medium.com/max/1340/1*qmWTg5fwhKmMaMhfo0HJIQ.png)](https://miro.medium.com/max/1340/1*qmWTg5fwhKmMaMhfo0HJIQ.png)  
(画像：[https://medium.com/@arxivtimes/%E3%82%B7%E3%83%B3%E3%83%97%E3%83%AB%E3%81%AA%E6%AD%A3%E8%A6%8F%E5%8C%96group-normalization%E3%81%A8-%E3%82%B3%E3%83%B3%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%81%AE%E8%BB%A2%E7%A7%BB%E3%82%92%E8%A9%A6%E3%81%BF%E3%81%9Felmo-86b9313f3e24](https://medium.com/@arxivtimes/%E3%82%B7%E3%83%B3%E3%83%97%E3%83%AB%E3%81%AA%E6%AD%A3%E8%A6%8F%E5%8C%96group-normalization%E3%81%A8-%E3%82%B3%E3%83%B3%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%81%AE%E8%BB%A2%E7%A7%BB%E3%82%92%E8%A9%A6%E3%81%BF%E3%81%9Felmo-86b9313f3e24))  

### Batch Norm

<!--
[![Batch Norm](https://cdn-ak.f.st-hatena.com/images/fotolife/j/jinbeizame007/20180924/20180924132105.png)](https://cdn-ak.f.st-hatena.com/images/fotolife/j/jinbeizame007/20180924/20180924132105.png)  
(画像：[https://jinbeizame.hateblo.jp/entry/understanding_batchnorm](https://jinbeizame.hateblo.jp/entry/understanding_batchnorm))  
-->

+ レイヤー間を流れるデータの分布を、<u>ミニバッチ単位で</u>  
    平均が0, 分散が1になるように正規化
    + H x W x CのsampleがN個あった場合に、N個の同一チャネルが正規化の単位
    + チャネルごとに正規化された特徴マップを出力
+ NNにおいて以下の効果
    + 学習時間の短縮
    + 初期値への依存低減
    + 過学習の抑制
+ 問題点
    + Batch Sizeが小さいと学習が収束しないことがある  
    →代わりにLayer Normalizationなどが使われることが多い
    + Batch Sizeがマシンのスペック等の影響を受ける

### Layer Norm

+ N個のsampleのうち一つに注目  
    H x W x Cの全てのpixelが正規化の単位
+ あるsampleの平均と分散を求め正規化を実施
    + 特徴マップごとに正規化された特徴マップを出力
+ ミニバッチの数に依存しない
+ 入力データや重み行列に対して、以下の操作をしても出力が変わらない
    + 入力データのスケールに関してロバスト
    + 重み行列のスケールやシフトに関してロバスト
    + 参考：[https://www.slideshare.net/KeigoNishida/layer-normalizationnips](https://www.slideshare.net/KeigoNishida/layer-normalizationnips)

### Instance Norm

+ 各sampleの各チャネルごとに正規化
    + Batch Normalizationのバッチサイズ1の場合と等価
+ コントラストの正規化に寄与
+ 画像のスタイル転送やテクスチャ合成タスクなどで利用
+ 参考
    + [https://blog.cosnomi.com/posts/1493/](https://blog.cosnomi.com/posts/1493/)
    + [https://gangango.com/2019/06/16/post-573/](https://gangango.com/2019/06/16/post-573/)
    + [https://blog.albert2005.co.jp/2018/09/05/group_normalization/](https://blog.albert2005.co.jp/2018/09/05/group_normalization/)

## WaveNet

[![WaveNet](http://musyoku.github.io/images/post/2016-09-17/dilated_conv.png)](http://musyoku.github.io/images/post/2016-09-17/dilated_conv.png)  
(画像：[http://musyoku.github.io/2016/09/18/wavenet-a-generative-model-for-raw-audio/](http://musyoku.github.io/2016/09/18/wavenet-a-generative-model-for-raw-audio/))  

↑各点は音声データの1サンプル(1/fs秒ごとのデータ)を表す  

+ Aaron van den Oord et. al., 2016
    +  Alpha  Goのプログラムを開発、2014年にGoogleに買収される
+ 生の音声波形を生成する深層学習モデル
+ Pixel CNN(高解像度の画像を精密に生成できる手法)を音声に応用したもの
+ メインアイディア
    + 時系列データに対して畳み込みを適用
    + →Dilated Convolution
        + 層が深くなるにつれて畳み込みリンクを離す
        + 受容野を簡単に増やせる
+ 論文
    + [WAVENET: A GENERATIVE MODEL FOR RAW AUDIO](https://arxiv.org/pdf/1609.03499.pdf)
+ 参考
    + [https://gigazine.net/news/20171005-wavenet-launch-in-google-assistant/](https://gigazine.net/news/20171005-wavenet-launch-in-google-assistant/)
    + [https://qiita.com/MasaEguchi/items/cd5f7e9735a120f27e2a](https://qiita.com/MasaEguchi/items/cd5f7e9735a120f27e2a)
    + [https://www.slideshare.net/NU_I_TODALAB/wavenet-86493372](https://www.slideshare.net/NU_I_TODALAB/wavenet-86493372)


# 例題解説

+ 深層学習を用いて結合確率を学習する際に、効率的に学習が行えるアーキテクチャを提案したことがWaveNetの大きな貢献の一つ  
    提案された新しいConvolution型アーキテクチャは(あ)と呼ばれ、結合確率を効率的に学習できるようになっている
    + 答え：Dilated causal convolution
    + Deconvolution(逆畳み込み)：画像の高解像度化などに使う
+ (あ)を用いた際の大きな利点は、単純なConvolution layerと比べ(い)ことである
    + パラメータ数に対する受容野が広い

---

# 物体検出とSS解説

SS: Semantic Segmentation  

## Introduction

[![intro](https://data-analysis-stats.jp/wp-content/uploads/2020/07/10_object_detection_classifcation_segmentation.png)](https://data-analysis-stats.jp/wp-content/uploads/2020/07/10_object_detection_classifcation_segmentation.png)  
(画像：[https://data-analysis-stats.jp/%E6%B7%B1%E5%B1%9E%E5%AD%A6%E7%BF%92/%E7%94%BB%E5%83%8F%E5%88%86%E9%A1%9E%E3%83%BB%E7%89%A9%E4%BD%93%E6%A4%9C%E5%87%BA%E3%83%BB%E3%82%BB%E3%82%B0%E3%83%A1%E3%83%B3%E3%83%86%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%AE%E6%AF%94%E8%BC%83/](https://data-analysis-stats.jp/%E6%B7%B1%E5%B1%9E%E5%AD%A6%E7%BF%92/%E7%94%BB%E5%83%8F%E5%88%86%E9%A1%9E%E3%83%BB%E7%89%A9%E4%BD%93%E6%A4%9C%E5%87%BA%E3%83%BB%E3%82%BB%E3%82%B0%E3%83%A1%E3%83%B3%E3%83%86%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%AE%E6%AF%94%E8%BC%83/))  

+ 入力：画像
+ 出力：
    + 分類
        + (画像に対し)(単一または複数の)クラスラベル
    + 物体検知(物体検知)
        + Bounding Box
    + 意味領域分割(Semantic Segmentation)
        + (各ピクセルに対し)(単一の)クラスラベル
    + 個体領域分割(Instance Segmentation)
        + (各ピクセルに対し)(単一の)クラスラベル

## 物体検知

以下が出力される  

+ Bounding Box
+ 予測ラベル
+ confidence

### 代表的データセット  

物体検出コンペで用いられたデータセット  

|  | クラス | Train+Val | Box/画像 |  |
| --- | --- | --- | --- | --- |
| VOC12 | 20 | 11540 | 2.4 | Instance Annotation |
| ILSVRC17 | 200 | 476668 |  1.1|  |
| MS COCO18 | 80 | 123287 | 7.3 | Instance Annotation |
| OICOD18 | 500 | 1743042 | 7.0 | Instance Annotation |

+ VOC: Visual Object Classes
+ ILSVRC: ImageNet Scale Visual Recognition Challenge
    + ImageNetのサブセット
+ MS COCO: (MicroSoft) Common Object in Context
    + 物体位置推定に対する新たな評価指標を提案
+ OICOD: Open Images Challenge Object Detection
    + ILSVRCやMS COCOとは異なるannotation process
    + Open Images V4のサブセット

Box / 画像
+ 小：アイコン的な写り、日常感とはかけはなれやすい
+ 大：部分的な重なり等も見られる。日常生活のコンテキストに近い

注意点
+ 目的に応じた **Box/画像** の選択を！
+ クラス数が大きければよいとも限らない
    + 同一の物体に対して違うラベルが付けられることも

### 評価指標

cf: 混同行列  
    →thresholdを変えると、検出される物体の数も変わる

#### IoU: Intersection over Union

別名: **Jaccard係数**  

物体検出においてはクラスラベルだけでなく、物体位置の予想精度も評価したい  

[![IoU](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.amazonaws.com%2F0%2F199265%2F54b47877-9a97-5dbc-f461-b9cf832faefe.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=79ea20fa54664ed839953b4bcb672889)](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.amazonaws.com%2F0%2F199265%2F54b47877-9a97-5dbc-f461-b9cf832faefe.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=79ea20fa54664ed839953b4bcb672889)  
(画像：[https://qiita.com/mshinoda88/items/9770ee671ea27f2c81a9](https://qiita.com/mshinoda88/items/9770ee671ea27f2c81a9))  

Confusion Matrixの要素を用いて表現すると  

$$
    \mathrm{IoU} = \frac{\mathrm{TP}}{\mathrm{TP + FP + FN}}
$$

+ TP: Area of Overlap
+ FN: Ground-Truth Bounding Box - TPの領域
+ FP: Predected Bounding Box - TPの領域

[![Confusion Matrix & IoU](https://ml.1book.info/cv/example_of_confusion_matrix_2_4.jpg)](https://ml.1book.info/cv/example_of_confusion_matrix_2_4.jpg)  
(画像：[https://ml.1book.info/cv/example_of_confusion_matrix_2.html](https://ml.1book.info/cv/example_of_confusion_matrix_2.html))  

【入力1枚で見るPrecision/Recall】  

+ conf.の閾値を超えているものをピックアップ  
+ IoUも閾値を超えていればTP
+ 既に同じ物体を検出済であれば、最もconf.の高いものを残して他はFP扱いにする   
    (閾値を超えていても)

+ クラス単位でPrecision/Recallを計算する

【Average Precision】  

+ IoUの閾値を0.5で固定
+ conf.の閾値を0.05~0.95の範囲で0.05ずつ変化させていく
    + 各閾値でPrecision, Recallを求める

conf.の閾値を$\beta$とすると、  
Recall = $R(\beta)$, Precision = $P(\beta)$
→P = f(R) PR曲線(Precision-Recall curve)

$$
    \mathrm{AP} = \int_{0}^{1} P(R)dR \qquad ←PR曲線の下側面積
$$

【mAP: mean Avarage Precision】  

APはクラスラベル固定のもとで考えていた  
→クラス数が$C$のとき、  

$$
    \mathrm{mAP} = \frac{1}{C} \sum_{i=1}^{C} \mathrm{AP}_i
$$

(おまけ)MS COCOで導入された指標  

IoUも0.5から0.95まで0.05刻みでAP&mAPを計算して算術平均を計算  
位置を厳しく調べていく  

$$
    \mathrm{mAPcoco} = \frac{\mathrm{mAP}_{0.5} + \mathrm{mAP}_{0.55} + \cdots + \mathrm{mAP}_{0.95}}{10}
$$

#### FPS

**Flames per Second**：検出速度  

[![FPS](https://ichi.pro/assets/images/max/724/1*SnjcWGpeClUN9XwxpOoJGg.png)](https://ichi.pro/assets/images/max/724/1*SnjcWGpeClUN9XwxpOoJGg.png)  
(画像：[https://ichi.pro/buttai-kenshutsu-moderu-no-rebyu-269886166492508](https://ichi.pro/buttai-kenshutsu-moderu-no-rebyu-269886166492508))  

**inference time**(1フレームあたりにかかる時間)で速度を表すこともある  

[![inference time](https://assets.st-note.com/production/uploads/images/9307533/picture_pc_23d39b7e4350ce52d5754a269a018d19.png?width=800)](https://assets.st-note.com/production/uploads/images/9307533/picture_pc_23d39b7e4350ce52d5754a269a018d19.png?width=800)  
(画像：[https://note.com/seishin55/n/n542b2b682721](https://note.com/seishin55/n/n542b2b682721))  

### マイルストーン

+ 2012
    + AlexNetの登場→時代はSIFTからDCNNへ
        + SIFT: Scale Invariant Feature Transform
            + 参考文献：[object recognition from local scale-invariant features](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)
        + DCNN: Deep Convolutional Neural Network
            + AlexNetの元論文：[ImageNet Classification with Deep Convolutional
Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
+ 2013~2018の代表的なネットワーク
    + VGGNet
    + GoogLeNet (Inception-v1)
    + ResNet
    + Inception-ResNet (Inception-v4)
    + DenseNet
    + MobileNet
    + AmoebaNet
+ 2013~2018の物体検知のフレームワーク
    + DetectorNet(1段階)
    + RCNN(2段階)
    + SPPNet(2段階)
    + FastRCNN(2段階)
    + YOLO(1段階)
    + FasterRCNN(2段階)
    + SSD(1段階)
    + RFCN(2段階)
    + YOLO9000(1段階)
    + FPN(2段階)
    + RetinaNet(1段階)
    + Mask RCNN(2段階)
    + CornerNet(1段階)

### 物体検知のフレームワーク  

[![framework](https://www.researchgate.net/profile/Anqi-Bao/publication/332612704/figure/fig1/AS:754572620468225@1556915540722/Schematic-plot-for-a-one-stage-detector-and-b-two-stage-detector.jpg)](https://www.researchgate.net/profile/Anqi-Bao/publication/332612704/figure/fig1/AS:754572620468225@1556915540722/Schematic-plot-for-a-one-stage-detector-and-b-two-stage-detector.jpg)  
(画像：[https://www.researchgate.net/figure/Schematic-plot-for-a-one-stage-detector-and-b-two-stage-detector_fig1_332612704](https://www.researchgate.net/figure/Schematic-plot-for-a-one-stage-detector-and-b-two-stage-detector_fig1_332612704))

【2段階検出器(Two-stage detector)】  
+ 候補領域の検出とクラス推定を  **別々に** 行う
+ 相対的に精度が高い
+ 計算量が多く推論も遅め

【1段階検出器(One-stage detector)】  
+ 候補領域の検出とクラス推定を **同時に** 行う
+ 相対的に精度が低い
+ 計算量が小さく推論も早め

動作例  

+ 2段階検出器では、検出した物体を一旦切り出す
+ 1段階検出器では、検出した物体を切り出さない

[![example](https://www.researchgate.net/profile/Phil-Ammirato/publication/308320592/figure/fig1/AS:408230695063552@1474341191358/Two-stage-vs-Proposed-a-The-two-stage-approach-separates-the-detection-and-pose.png)](https://www.researchgate.net/profile/Phil-Ammirato/publication/308320592/figure/fig1/AS:408230695063552@1474341191358/Two-stage-vs-Proposed-a-The-two-stage-approach-separates-the-detection-and-pose.png)
(画像：[https://www.researchgate.net/figure/Two-stage-vs-Proposed-a-The-two-stage-approach-separates-the-detection-and-pose_fig1_308320592](https://www.researchgate.net/figure/Two-stage-vs-Proposed-a-The-two-stage-approach-separates-the-detection-and-pose_fig1_308320592))

#### SSD: Single Shot Detector

1. Default BOXを用意
    + 適当に一つBoxを用意する
1. Default BOXを変形し、conf.を出力

VGG16がベースになっている  

[![VGG16](https://ichi.pro/assets/images/max/724/0*xurYLT8UBpFKPNQA)](https://ichi.pro/assets/images/max/724/0*xurYLT8UBpFKPNQA)  
(画像：[https://ichi.pro/vgg-16-to-wa-nani-desu-ka-vgg-16-no-gaiyo-267001881294357](https://ichi.pro/vgg-16-to-wa-nani-desu-ka-vgg-16-no-gaiyo-267001881294357))  

+ 16: convolution + ReLU 13層 + fully connected + ReLU 3層
+ 原著論文：[VERY DEEP CONVOLUTIONAL NETWORKS
FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf)  

【ネットワークアーキテクチャ】  

[![SSD](https://blog.negativemind.com/wp-content/uploads/2019/02/SSD_network.jpg)](https://blog.negativemind.com/wp-content/uploads/2019/02/SSD_network.jpg)  
(画像：[https://blog.negativemind.com/2019/02/26/general-object-recognition-single-shot-multibox-detector/](https://blog.negativemind.com/2019/02/26/general-object-recognition-single-shot-multibox-detector/))  

【特徴】

+ 入力サイズ: SSD300
+ VGGのFC層2層をConv層に変更
    + VGG16の最後のFC層は削除
+ マルチスケール特徴マップ
    + 解像度の異なる特徴マップから出力をつくる

【特徴マップからの出力】  

+ マップ中の一つの特徴量における一つのDefault BOXについて
    + 出力サイズ：クラス数 + 4
        + オフセット項が4つ ($\Delta x, \Delta y, \Delta w, \Delta h$)
            + $x = x_0 + 0.1 \times \Delta x \times w_0$
            + $y = y_0 + 0.1 \times \Delta y \times h_0$
            + $w = w_0 \times \exp (0.2 \times \Delta w)$
            + $h = h_0 \times \exp (0.2 \times \Delta h)$
+ マップ中の各特徴量にk個のDefault BOXを用意するとき、
    + 出力サイズ：k(クラス数 + 4)
+ 特徴マップのサイズがmxnであるとすれば、
    + 出力サイズ：k(クラス数 + 4) mn
        + k, mn: 特徴マップごとに用意するDefault BOX数
+ VOCデータセットでのデフォルトボックス数を考える
    + クラス数20 + 背景クラス → #Class = 21
    + 以下の層について、Default Boxは各特徴に4つ  
        + Conv4_3
        + Conv10_2
        + Conv11_2
        + →{4 * (21 + 4) * 38 * 38} + {4 * (21 + 4) * 3 * 3} + {4 * (21 + 4) * 1 * 1}
    + 以下の層について、Default Boxは各特徴に6つ
        + Conv7
        + Conv6_2
        + Conv9_2
        + →{6 * (21 + 4) * 19 * 19} + {6 * (21 + 4) * 10 * 10} + {6 * (21 + 4) * 5 * 5}
    + 合計: 8732 * (21 + 4)

注意：画像の物理的なサイズ≠解像度  

【多数のDefault Boxを用意したことで発生する問題】  

+ Non-Maximum Suppression
    + 一つの物体に複数のBounding Box
    + 対策：IoUの最も高いBounding Box以外は排除する
        + IoUが0.3以上のものが対象
+ Hard Negative Mining
    + Negative Class: 背景クラス
    + 背景クラス(Negative Class)と他のクラス(Positive Class)の検出数の不均衡
        + 背景クラスの方が圧倒的に多い
    + 対策：Negative : Positive = 3 : 1までになるようにする

【損失関数】  

$$
    L(x, c, l, g) = \frac{1}{N}(\underbrace{L_{conf}(x, c)}_{*1} + \alpha \underbrace{L_{loc}(x, l, g)}_{*2})
$$

*1: confidenceに対する損失  

$$
    L_{conf}(x, c) = - \sum_{i \in Pos}^{N} x^p_{ij} log (\hat{c}^p_i) - \sum_{i \in Neg} log (\hat{c}^0_i) \qquad \mathrm{where} \quad \hat{c}^p_i = \frac{\exp (c^p_i)}{\sum_p \exp (c^p_i)}
$$

*2: 位置検出に対する損失  

$$
    L_{loc}(x, l, g) = \sum_{i \in Pos}^{N} \sum_{m \in {cx, xy, w, h}} x^k_{ij} \underbrace{\mathrm{smooth}_{L1} (l^m_i - \hat{g}^m_j)}_{*3} \\  
    \hat{g}^{cx}_j = (g^{cx}_j - d^{cx}_i) / d^w_i \qquad
    \hat{g}^{cy}_j = (g^{cy}_j - d^{cy}_i) / d^h_i \\  
    \hat{g}^w_j = \log \left( \frac{g^w_j}{d^w_i} \right) \qquad
    \hat{g}^h_j = \log\left( \frac{g^h_j}{d^h_i} \right)
$$

*3: Faster RCNNでも用いられる Smooth L1 Loss  

【SSDの進化】  

+ [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
+ [DSSD: Deconvolutional Single Shot Detector](https://arxiv.org/abs/1701.06659)
+ [Extend the shallow part of Single Shot MultiBox Detector via Convolutional Neural Network](https://arxiv.org/abs/1801.05918)
+ [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/abs/1711.06897)

## Semantic Segmentation

### 概略

【Semantic Segmentationの肝】  

[![FCN](https://cvml-expertguide.net/wp-content/uploads/2021/06/FCN-1536x797.png)](https://cvml-expertguide.net/wp-content/uploads/2021/06/FCN-1536x797.png)  
(画像：[https://cvml-expertguide.net/2021/06/13/fcn/](https://cvml-expertguide.net/2021/06/13/fcn/))  

+ 課題
    + Up-Samplingの壁：Convolution + Poolingで落ちた解像度をどう元に戻すのか？
+ そもそも、Poolingは必要？
    + →受容野を広げるために必要！

【FCNの基本アイディア】  

[![FCN](https://blog.negativemind.com/wp-content/uploads/2019/03/output_heatmap.jpg)](https://blog.negativemind.com/wp-content/uploads/2019/03/output_heatmap.jpg)  
(画像：[https://blog.negativemind.com/2019/03/11/semantic-segmentation-by-fully-convolutional-network/](https://blog.negativemind.com/2019/03/11/semantic-segmentation-by-fully-convolutional-network/))  

+ Fully Connected層をConvolution層に変更
+ 出力がheatmapになる

### Up-sampling

#### Deconvolution / Transposed Convolution  

[![Deconvolution](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.amazonaws.com%2F0%2F103085%2F20e0e845-f631-1995-d85e-bc39f69ae9f8.gif?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=c9a27c7311724d91b0ea801b58efdcf6)](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.amazonaws.com%2F0%2F103085%2F20e0e845-f631-1995-d85e-bc39f69ae9f8.gif?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=c9a27c7311724d91b0ea801b58efdcf6)  
(画像：[https://qiita.com/shngt/items/9c86e69e16ce6d61a0c6](https://qiita.com/shngt/items/9c86e69e16ce6d61a0c6))  

+ 逆畳み込みとも呼ばれる
    + ただし、畳み込みの逆演算ではない

【処理手順】  

1. 特徴マップのpixel間隔をstrideだけ空ける
1. 特徴マップの周りに(kernel size - 1) - paddingだけ余白をつくる
1. 畳み込み演算を行う

【輪郭情報の補完】  

FCNはこのように用いられる  

[![complement](https://blog.negativemind.com/wp-content/uploads/2019/03/upsampling.jpg)](https://blog.negativemind.com/wp-content/uploads/2019/03/upsampling.jpg)  
(画像：[https://blog.negativemind.com/2019/03/11/semantic-segmentation-by-fully-convolutional-network/](https://blog.negativemind.com/2019/03/11/semantic-segmentation-by-fully-convolutional-network/))  

+ 低レイヤーPooling層の出力をelement-wise addition  
    →ローカルな情報を補完してからUp-sampling

【U-Net】  

[![U-Net](https://blog.negativemind.com/wp-content/uploads/2019/03/u-net-architecture-1.jpg)](https://blog.negativemind.com/wp-content/uploads/2019/03/u-net-architecture-1.jpg)  
(画像：[https://blog.negativemind.com/2019/03/11/semantic-segmentation-by-fully-convolutional-network/](https://blog.negativemind.com/2019/03/11/semantic-segmentation-by-fully-convolutional-network/))  

+ element-wise additionではなく、チャネル方向の結合を行う

(参考)【DeconvNet】  

[![DeconvNet](https://ichi.pro/assets/images/max/724/1*LW8Anre45o9nfamxIVTY8Q.png)](https://ichi.pro/assets/images/max/724/1*LW8Anre45o9nfamxIVTY8Q.png)  
(画像：[https://ichi.pro/rebyu-deconvnet-anpu-ringureiya-semanthikku-segumente-shon-94349869070350](https://ichi.pro/rebyu-deconvnet-anpu-ringureiya-semanthikku-segumente-shon-94349869070350))  

(参考)【SegNet】  

[![SegNet](https://cdn-ak.f.st-hatena.com/images/fotolife/m/mabonki0725/20170419/20170419074518.png)](https://cdn-ak.f.st-hatena.com/images/fotolife/m/mabonki0725/20170419/20170419074518.png)  
(画像：[https://mabonki0725.hatenablog.com/entry/2017/04/19/113424](https://mabonki0725.hatenablog.com/entry/2017/04/19/113424))  

#### Unpooling  

[![Unpooling1](https://ichi.pro/assets/images/max/724/1*8RHLqbd2UJshkoDfZMdOmg.png)](https://ichi.pro/assets/images/max/724/1*8RHLqbd2UJshkoDfZMdOmg.png)  
[![Unpooling2](https://ichi.pro/assets/images/max/724/1*AbCrAqPBfkqGRdhKtiZQqA.png)](https://ichi.pro/assets/images/max/724/1*AbCrAqPBfkqGRdhKtiZQqA.png)  
(画像：[https://ichi.pro/rebyu-deconvnet-anpu-ringureiya-semanthikku-segumente-shon-94349869070350](https://ichi.pro/rebyu-deconvnet-anpu-ringureiya-semanthikku-segumente-shon-94349869070350))  

+ Poolingした時の位置情報を保持

### Dilated Convolution

[![Dilated Convolution](https://miro.medium.com/max/790/0*3cTXIemm0k3Sbask.gif)](https://miro.medium.com/max/790/0*3cTXIemm0k3Sbask.gif)  
[![Dilated Convolution2](https://miro.medium.com/max/1400/1*tnDNIyPePgHvb8JIx8SbqA.png)](https://miro.medium.com/max/1400/1*tnDNIyPePgHvb8JIx8SbqA.png)  
(画像：[https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5?gi=ccef31e3dad2](https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5?gi=ccef31e3dad2))  

+ Poolingを用いず、Convolutionの段階で受容野を広げる工夫  
+ 上記の例では、
    + 3x3 Conv. → 3x3(rate=2)Conv. → 3x3(rate=4)Conv.
    + 最終的に3層で15 x 15の範囲をカバーできている
        + (同範囲で3x3のConv.を続けた場合、7層必要)

# BERT

## Seq2Seq

+ 系列を入力し、系列を出力
+ Encoder-Decoderモデルとも呼ばれる
+ 入力系列→(Encode)→内部状態→(Decode)→出力系列
+ 実用例
    + 翻訳(英→日)
    + 音声認識(波形→テキスト)
    + チャットボット(テキスト→テキスト)

【理解に必要な材料】  

+ RNNの理解
+ 言語モデルの理解

### 言語モデル  

+ 単語の並びに確率を与える
    + 尤度 ←文章として自然かを評価
+ 数式的には同時確率を事後確率に分解して表せる

$$
    P(w_1, \cdots, w_m) = \prod_{i=1}^{m} P(w_i | w_1, \cdots, w_{i-1})
$$

+ 時刻t-1までの情報で、時刻tの事後確率を求めることが目標
    →これで同時確率が計算できる

$$
    \underbrace{ \underset{w \in V}{\mathrm{argmax}}}_{*1} P( \mathrm{I, have, a,} \underbrace{w}_{dog, \space pen, \cdots})
$$

*1: Set of all words in the vocabulary

【RNN x 言語モデル】  

[![RNN x Lang model](https://cdn-ak.f.st-hatena.com/images/fotolife/t/taxa_program/20190108/20190108000309.png)](https://cdn-ak.f.st-hatena.com/images/fotolife/t/taxa_program/20190108/20190108000309.png)  
(画像：[https://www.takapy.work/entry/2019/01/09/080338](https://www.takapy.work/entry/2019/01/09/080338))  

+ 言語モデルを再現するようにRNNの重みが学習されていれば、ある時点の次の単語を予測することが可能
    + 先頭単語を与えれば文章も生成できる
+ 初期値は適当(0, 乱数など)

### Seq2Seq  

+ EncoderからDecoderへ渡される内部状態ベクトルが鍵
+ Decoder側の構造は言語モデルRNNとほぼ同じ
    + 隠れ状態の初期値にEncoder側の内部状態を受け取る
+ DecoderのOutput側に正解を当てれば、教師あり学習がEnd2Endで行える
    + Lossを計算可能→誤差逆伝播可能→重みの更新ができる

[![Seq2Seq](https://miro.medium.com/max/1400/1*x4wsJobiSC7zlTkP8yv40A.png)](https://miro.medium.com/max/1400/1*x4wsJobiSC7zlTkP8yv40A.png)  
(画像：[https://medium.com/analytics-vidhya/seq2seq-model-and-the-exposure-bias-problem-962bb5607097](https://medium.com/analytics-vidhya/seq2seq-model-and-the-exposure-bias-problem-962bb5607097))  

【Teacher Forcing】  

+ 正解ラベルを直接Decoderの入力にする
+ RNNでDecoderを学習させると連鎖的に誤差が大きくなっていき、学習が不安定になったり収束が遅くなったりする問題の対策
+ 学習時と予測時で出力を行う環境が異なる問題がある
    + 予測時には教師データがない
    + Scheduled Samplingで対策する
        + ターゲット系列を入力とするか生成された結果を入力とするかを確率的にサンプリングする

【BLEU】  

+ プロの翻訳結果に対し、モデルの出力結果をスコア化する
    + 翻訳精度を相対的に測れる

## Transformer

【ニューラル機械翻訳の問題点】  

+ 長さに弱い
    + 翻訳元の文の内容を一つのベクトルで表現
        + 文が長くなると表現力が足りなくなる

### Attention (注意機構)

+ 辞書オブジェクト
    + queryに一致するkeyを索引
    + 対応するvalueを取り出す
        + 正規化された尤度を利用
+ 翻訳先の各単語を選択する際に、翻訳元の文中の各単語の隠れ状態を利用

[![Attention](https://cdn-ak.f.st-hatena.com/images/fotolife/h/hilinker/20181121/20181121195334.png)](https://cdn-ak.f.st-hatena.com/images/fotolife/h/hilinker/20181121/20181121195334.png)  
(画像：[https://hilinker.hatenablog.com/entry/2018/12/08/002003](https://hilinker.hatenablog.com/entry/2018/12/08/002003))  

【翻訳元の各単語の隠れ状態の加重平均】  

$$
    c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j \cdot
$$

【重み】  

+ 全て足すと1
+ FFNNで求める

$$
    \begin{split}
        \alpha_{ij} &= \frac{\exp (e_{ij})}{\sum_{k=1}^{T_x} \exp (e_{ik})}, \\  
        e_{ij} &= a(s_{i-1}, h_j)
    \end{split}
$$

【2種類のAttention】  

$$
    \mathrm{softmax}(QK^T)V
$$

[![2 Attentions](https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20171221/20171221164314.png)](https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20171221/20171221164314.png)  
(画像：[https://forest1040.growi.cloud/nmt/attention](https://forest1040.growi.cloud/nmt/attention))  

+ Source Target Attention
    + 情報が来る場所と情報が狙う場所が別れている
    + 受け取った情報に対して狙う情報が近いものをAttentionベクトルとして取り出し注目
+ Self Attention
    + Q, K, Vが全て同じところから来る
    + どの情報に注目すべきか、自分の入力だけで決める
    [![Self Attention](https://camo.qiitausercontent.com/c0357c70af7308f9be5bb30ad4e69fa2f7a00629/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f3132333538392f37353566343237302d336331302d653134612d343033362d3561626639306565663137312e706e67)](https://camo.qiitausercontent.com/c0357c70af7308f9be5bb30ad4e69fa2f7a00629/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f3132333538392f37353566343237302d336331302d653134612d343033362d3561626639306565663137312e706e67)  
    (画像：[https://seiichiinoue.github.io/post/nlp/](https://seiichiinoue.github.io/post/nlp/))  
    + CNNと似ている。文脈依存
        + 文脈を反映した自己表現を得られる
        + 違い：
            + CNN：ウインドウサイズのため、決められた範囲内のconvolutionしか行わない
            + Self Attention：ウインドウサイズが文の長さのconvolution

【Position-Wise Feed-Forward Networks】  

+ 位置情報を保持したまま順伝播させる
+ 各Attention層の出力を決定
    + 2層の全結合NN
    + 線形変換→ReLU→線形変換

$$
    \mathrm{FFN}(x) = \max (0, x W_1 + b_1) W_2 + b_2 \\  
    W_1 \in \mathbb{R}^{512 \times 2048} \qquad b_1 \in \mathbb{R}^{2048} \\  
    W_2 \in \mathbb{R}^{2048 \times 512} \qquad b_2 \in \mathbb{R}^{512}
$$

【Scaled dot product attention】  

全単語に関するAttentionをまとめて計算する  

[![Scaled dot product attension](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.ap-northeast-1.amazonaws.com%2F0%2F209705%2F144a439c-1381-3b41-8318-3259a5015ded.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=95240c3f24b3cf93062588ea4177ca95)](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.ap-northeast-1.amazonaws.com%2F0%2F209705%2F144a439c-1381-3b41-8318-3259a5015ded.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=95240c3f24b3cf93062588ea4177ca95)  
(画像：[https://qiita.com/jun40vn/items/9135bf982d73d9372238](https://qiita.com/jun40vn/items/9135bf982d73d9372238))  

+ $d_k$ : 内部状態の次元数  
    →積の演算が大きくなっていくのを防ぐため、スケーリングしている  
+ Mask: UNK、PADなどをカットしたい場合に使う

$$
    \underbrace{
        \left[
            \begin{array}{c}
                & q_1 & \\  
                & q_2 & \\  
                & q_3 &
            \end{array}
        \right]
    }_{Q}
    \underbrace{
        \left[
            \begin{array}{c}
                \space \\  
                k_1 & k_2 & k_3 \\  
                \space
            \end{array}
        \right]
    }_{K} = 
    \left[
        \begin{array}{c}
            q_1 \cdot k_1 & q_1 \cdot k_2 & q_1 \cdot k_3 \\  
            q_2 \cdot k_1 & q_2 \cdot k_2 & q_2 \cdot k_3 \\  
            q_3 \cdot k_1 & q_3 \cdot k_2 & q_3 \cdot k_3
        \end{array}
    \right] \\  
    \underbrace{
        \left[
            \begin{array}{c}
                0.1 & 0.2 & 0.7 \\  
                0.4 & 0.3 & 0.3 \\  
                0.8 & 0.1 & 0.1
            \end{array}
        \right]
    }_{\mathrm{softmax}(QK)}
    \underbrace{
        \left[
            \begin{array}{c}
                & v_1 & \\  
                & v_2 & \\  
                & v_3 &
            \end{array}
        \right]
    }_{V} = 
    \left[
        \begin{array}{c}
            0.1 v_1 + 0.2 v_1 + 0.7 v_1 \\  
            0.4 v_2 + 0.3 v_2 + 0.3 v_2 \\  
            0.7 v_3 + 0.1 v_3 + 0.1 v_3 
        \end{array}
    \right] \\  
$$

### Transformer

[![Transformer](https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20171221/20171221163853.png)](https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20171221/20171221163853.png)  
(画像：[https://deeplearning.hatenablog.com/entry/transformer](https://deeplearning.hatenablog.com/entry/transformer))  

+ 2017年6月に登場
+ RNNを使わない
    + 必要なのはAttentionだけ
+ 当時のSOTAを、はるかに少ない計算量で実現
    + SOTA(State of the Art): もっとも高精度であること
+ 英仏(3600万文)の学習を8GPUで3.5日で完了

【主要モジュール】  

1. Positional Encoding
    + 単語ベクトルに単語の位置を追加
1. Multi-Head Attention
    + 複数のヘッドで行うDot Product Attention
1. Feed Forward
    + 単語の位置ごとに独立処理する全結合
1. Masked Multi-Head Attention
    + 未来の単語を見ないようマスク

# DCGAN

---

# 実装演習

以下、出力結果が長い場合は一部を省略する  

## 4_1_tensorflow_codes.ipynb

+ base
    + constant
        + 実行結果  
            ![constant]({{site.baseurl}}/images/20211214.png)  
    + placeholder
        + 実行結果  
            ![place holder]({{site.baseurl}}/images/20211214_1.png)  
    + variables
        + 実行結果  
             ![variables]({{site.baseurl}}/images/20211214_2.png)  
+ 線形回帰
    + 実行結果  
        ![linear regression]({{site.baseurl}}/images/20211214_3.png)  
        ![graph]({{site.baseurl}}/images/20211214_4.png)  
    + [try]noiseの値を変更してみる
        + 0.3 → 0.1にしてみる  
            ![0.1]({{site.baseurl}}/images/20211214_7.png)  
            ![graph]({{site.baseurl}}/images/20211214_8.png)  
        + 0.3 → 0.6にしてみる  
            ![0.6]({{site.baseurl}}/images/20211214_9.png)  
            ![graph]({{site.baseurl}}/images/20211214_10.png)  
    + [try]dの数値を変更してみる
        + d = 2000にしてみる  
            ![2000]({{site.baseurl}}/images/20211214_11.png)  
            ![graph]({{site.baseurl}}/images/20211214_12.png)  
        + d = 10にしてみる  
            ![10]({{site.baseurl}}/images/20211214_13.png)  
            ![graph]({{site.baseurl}}/images/20211214_14.png)  
+ 非線形回帰
    + 実行結果
        ![nonlinear regression]({{site.baseurl}}/images/20211214_5.png)  
        ![graph]({{site.baseurl}}/images/20211214_6.png)  
    + [try]noiseの値を変更してみる
        + 0.05 → 0.5にしてみる  
            ![0.5]({{site.baseurl}}/images/20211214_15.png)  
            ![graph]({{site.baseurl}}/images/20211214_16.png)  
        + 0.05 → 0.005にしてみる  
            グラフでは、変化があまりないように見える  
            ![0.005]({{site.baseurl}}/images/20211214_17.png)  
            ![graph]({{site.baseurl}}/images/20211214_18.png)  
    + [try]dの数値を変更してみる
        + d = 100にしてみる  
            ![100]({{site.baseurl}}/images/20211214_19.png)  
            ![graph]({{site.baseurl}}/images/20211214_20.png)  
        + d = 1にしてみる
            ![1]({{site.baseurl}}/images/20211214_21.png)  
            ![graph]({{site.baseurl}}/images/20211214_22.png)  
+ [try]
    + 次の式をモデルとして回帰を行う  
        $y = 30x^2 + 0.5x + 0.2$  
    + 誤差が収束するようiters_numやlearning_rateを調整する  
        + コード  
            ```Python
            import numpy as np
            import tensorflow as tf
            import matplotlib.pyplot as plt

            iters_num = 35000
            plot_interval = 1000

            # データを生成
            n=100
            x = np.random.rand(n).astype(np.float32) * 4 - 2
            d = 30. * x ** 2 + 0.5 * x + 0.2

            #  ノイズを加える
            noise = 0.05
            d = d + noise * np.random.randn(n) 

            # モデル
            # bを使っていないことに注意.
            xt = tf.placeholder(tf.float32, [None, 4])
            dt = tf.placeholder(tf.float32, [None, 1])
            W = tf.Variable(tf.random_normal([4, 1], stddev=0.01))
            y = tf.matmul(xt,W)

            # 誤差関数 平均２乗誤差
            loss = tf.reduce_mean(tf.square(y - dt))
            optimizer = tf.train.AdamOptimizer(0.001)
            train = optimizer.minimize(loss)

            # 初期化
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            # 作成したデータをトレーニングデータとして準備
            d_train = d.reshape(-1,1)
            x_train = np.zeros([n, 4])
            for i in range(n):
                for j in range(4):
                    x_train[i, j] = x[i]**j

            #  トレーニング
            for i in range(iters_num):
                if (i+1) % plot_interval == 0:
                    loss_val = sess.run(loss, feed_dict={xt:x_train, dt:d_train}) 
                    W_val = sess.run(W)
                    print('Generation: ' + str(i+1) + '. 誤差 = ' + str(loss_val))
                sess.run(train, feed_dict={xt:x_train,dt:d_train})

            print(W_val[::-1])
                
            # 予測関数
            def predict(x):
                result = 0.
                for i in range(0,4):
                    result += W_val[i,0] * x ** i
                return result

            fig = plt.figure()
            subplot = fig.add_subplot(1,1,1)
            plt.scatter(x ,d)
            linex = np.linspace(-2,2,100)
            liney = predict(linex)
            subplot.plot(linex,liney)
            plt.show()
            ```
        + 結果  
            ![try]({{site.baseurl}}/images/20211215.png)  
            ![graph]({{site.baseurl}}/images/20211215_1.png)  
+ 分類1層(mnist)
    + [try]x, d, W, bを定義する
        + コード  
            ```Python
            x = tf.placeholder(tf.float32, [None, 784])
            d = tf.placeholder(tf.float32, [None, 10])
            W = tf.Variable(tf.random_normal([784, 10], stddev=0.01))
            b = tf.Variable(tf.zeros([10]))
            ```
        + 結果  
            ![x, d, w, b]({{site.baseurl}}/images/20211215_2.png)  
            ![graph]({{site.baseurl}}/images/20211215_3.png)  
+ 分類3層(mnist)
    + 実行結果  
        ![result]({{site.baseurl}}/images/20211214_23.png)  
        ![graph]({{site.baseurl}}/images/20211214_24.png)  
    + [try]隠れ層のサイズを変更してみる  
        + hidden_layer_size_1 = 150, hidden_layer_size_2 = 75にしてみる  
            ![150, 75]({{site.baseurl}}/images/20211215_4.png)  
            ![graph]({{site.baseurl}}/images/20211215_5.png)  
        + hidden_layer_size_1 = 2400, hidden_layer_size_2 = 1200にしてみる  
            ![2400, 1200]({{site.baseurl}}/images/20211215_6.png)  
            ![graph]({{site.baseurl}}/images/20211215_7.png)  
    + [try]optimizerを変更してみる  
        + GradientDescentOptimizer (learning_rate=0.95)  
            ![GradientDescentOptimizer]({{site.baseurl}}/images/20211215_8.png)  
            ![graph]({{site.baseurl}}/images/20211215_9.png)  
        + MomentumOptimizer (learning_rate=0.95, momentum=0.1)  
            ![MomentumOptimizer]({{site.baseurl}}/images/20211215_10.png)  
            ![graph]({{site.baseurl}}/images/20211215_11.png)  
        + AdagradOptimizer (learning_rate=0.95)  
            ![AdagradOptimizer]({{site.baseurl}}/images/20211215_12.png)  
            ![graph]({{site.baseurl}}/images/20211215_13.png)  
        + RMSPropOptimizer (learning_rate=0.001)  
            ![RMSPropOptimizer]({{site.baseurl}}/images/20211215_14.png)  
            ![graph]({{site.baseurl}}/images/20211215_15.png)  
        + AdamOptimizer  
            ![AdamOptimizer]({{site.baseurl}}/images/20211215_16.png)  
            ![graph]({{site.baseurl}}/images/20211215_17.png)  
+ 分類CNN(mnist)
    + 実行結果  
        ![result]({{site.baseurl}}/images/20211214_25.png)  
        ![graph]({{site.baseurl}}/images/20211214_26.png)  
    + [try]dropout率を0にしてみる  
        ![dropout=0]({{site.baseurl}}/images/20211214_27.png)  
        ![graph]({{site.baseurl}}/images/20211214_28.png)  
+ (メモ)
    + Firefoxにて`Matplotlib`を使ってグラフが表示できない  
        Chromeで試したら表示できた  
        アドオンが何か悪さをしている？  

## 4_3_keras_codes.ipynb  

+ keras
    + 線形回帰
        + 実行結果
            ![linear regression]({{site.baseurl}}/images/20211215_18.png)  
            ![graph]({{site.baseurl}}/images/20211215_19.png)  
    + 単純パーセプトロン
        + 実行結果
            ![Simple perceptron]({{site.baseurl}}/images/20211215_20.png)  
        + [try]np.random.seed(0)をnp.random.seed(1)に変更
            ![seed(1)]({{site.baseurl}}/images/20211215_29.png)  
        + [try]エポック数を100に変更  
            →lossの値が小さくなった  
            ![epoch=100]({{site.baseurl}}/images/20211215_30.png)  
        + [try]AND回路, XOR回路に変更
            + AND回路
                + コード  
                    ```Python  
                    X = np.array( [[0,0], [0,1], [1,0], [1,1]] )
                    T = np.array( [[0], [0], [0], [1]] )
                    ```
                + 結果：ORより精度が落ちた 
                    ![AND]({{site.baseurl}}/images/20211215_33.png)  
            + XOR回路
                + コード
                    ```Python
                    X = np.array( [[0,0], [0,1], [1,0], [1,1]] )
                    T = np.array( [[0], [1], [1], [0]] )
                    ```
                + 結果：lossの値が大きい。うまく学習できていない  
                    ![XOR]({{site.baseurl}}/images/20211215_34.png)  
        + [try]OR回路にしてバッチサイズを10に変更  
            →lossの値が大きくなった  
            ![batch_size=10]({{site.baseurl}}/images/20211215_31.png)  
        + [try]エポック数を300に変更しよう  
            →lossの値が一番小さくなった  
            ![epoch=300]({{site.baseurl}}/images/20211215_32.png)  
    + 分類(iris)
        + 実行結果
            ![iris]({{site.baseurl}}/images/20211215_21.png)  
            ![graph]({{site.baseurl}}/images/20211215_22.png)  
        + [try]中間層の活性関数をsigmoidに変更しよう
            ![sigmoid]({{site.baseurl}}/images/20211215_35.png)  
            ![graph]({{site.baseurl}}/images/20211215_36.png)  
        + [try]SGDをimportしoptimizerをSGD(lr=0.1)に変更しよう
            ![SGD]({{site.baseurl}}/images/20211215_37.png)  
            ![graph]({{site.baseurl}}/images/20211215_38.png)  
    + 分類(mnist)
        + 実行結果
        + [try]load_mnistのone_hot_labelをFalseに変更しよう (error)
            ![error]({{site.baseurl}}/images/20211215_39.png)  
        + [try]誤差関数をsparse_categorical_crossentropyに変更しよう
        + [try]Adamの引数の値を変更しよう
    + CNN分類(mnist)
        + 実行結果
            ![cnn mnist]({{site.baseurl}}/images/20211215_23.png)  
            ![graph]({{site.baseurl}}/images/20211215_24.png)  
    + cifar10
        + 実行結果
            ![cifar10]({{site.baseurl}}/images/20211215_25.png)  
            ![graph]({{site.baseurl}}/images/20211215_26.png)  
    + RNN
        + 実行結果
            ![RNN]({{site.baseurl}}/images/20211215_27.png)  
        + [try]RNNの出力ノード数を128に変更
        + [try]RNNの出力活性化関数を sigmoid に変更
        + [try]RNNの出力活性化関数を tanh に変更
        + [try]最適化方法をadamに変更
        + [try]RNNの入力 Dropout を0.5に設定
        + [try]RNNの再帰 Dropout を0.3に設定
        + [try]RNNのunrollをTrueに設定
+ (メモ)
    + 以下のコードはエラーになり動かなかったので修正
        + 分類(iris) L.35, 36など
            ```Python
            # 修正前：「KeyError: 'acc'」というエラーが出る
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            ```
            ```Python
            # 修正後：指定するkeyを変更
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            ```
        + 分類(mnist) L.31~33
            ```Python
            # 修正前：epsilonがNoneだとエラーになる
            model.compile(loss='categorical_crossentropy', 
                        optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), 
                        metrics=['accuracy'])
            ```
            ```Python
            # 修正後：`keras.backend.epsilon()`(デフォルトの値)を設定
            model.compile(loss='categorical_crossentropy', 
                        optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=keras.backend.epsilon(), decay=0.0, amsgrad=False), 
                        metrics=['accuracy'])
            ```

## lecture_chap1_exercise_public.ipynb

実行結果

### 1. データセットの準備

#### 1.1データの読み込みと単語の分割

![1.1.1]({{site.baseurl}}/images/20211219.png)  

![1.1.2]({{site.baseurl}}/images/20211219_1.png)  

![1.1.3]({{site.baseurl}}/images/20211219_2.png)  

#### 1.2単語辞書の作成

![1.2.1]({{site.baseurl}}/images/20211219_3.png)  

### 2.テンソルへの変換

#### 2.1 IDへの変換

![2.1.1]({{site.baseurl}}/images/20211219_4.png)  

### 3.モデルの構築

#### 導入：PackedSequence

![3.0.1]({{site.baseurl}}/images/20211219_5.png)  

![3.0.2]({{site.baseurl}}/images/20211219_6.png)  

![3.0.3]({{site.baseurl}}/images/20211219_7.png)  

### 4.訓練

#### 4.2学習

![4.2.1]({{site.baseurl}}/images/20211219_8.png)  
![4.2.2]({{site.baseurl}}/images/20211219_9.png)  
![4.2.3]({{site.baseurl}}/images/20211219_10.png)  

### 5.評価

![5.1.1]({{site.baseurl}}/images/20211219_11.png)  

![5.1.2]({{site.baseurl}}/images/20211219_12.png)  

![5.1.2]({{site.baseurl}}/images/20211219_13.png)  

### Beam Search

![6.1.1]({{site.baseurl}}/images/20211219_14.png)  

![6.1.2]({{site.baseurl}}/images/20211219_15.png)  
![6.1.3]({{site.baseurl}}/images/20211219_16.png)  

(メモ)  

+ 実行時にエラーが発生して止まってしまう  
    + エラー発生箇所：セル[1] L.2
        ```Python
        from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
        ```  
    + エラー内容：
        ```
        `Colab - ModuleNotFoundError: No module named 'wheel.pep425tags'
        ```  
    + 対策：事前に以下を実行しておく  
        ```Python
        !pip3 install wheel==0.34.1
        ```
    + 備考：まだ以下のエラーが出るが、一応動くようになったのでとりあえず様子見
        ```Python
        ERROR: HTTP error 403 while getting http://download.pytorch.org/whl/cu80/torch-0.4.0-cp37-cp37m-linux_x86_64.whl
        ERROR: Could not install requirement torch==0.4.0 from http://download.pytorch.org/whl/cu80/torch-0.4.0-cp37-cp37m-linux_x86_64.whl because of HTTP error 403 Client Error: Forbidden for url: http://download.pytorch.org/whl/cu80/torch-0.4.0-cp37-cp37m-linux_x86_64.whl for URL http://download.pytorch.org/whl/cu80/torch-0.4.0-cp37-cp37m-linux_x86_64.whl
        ```