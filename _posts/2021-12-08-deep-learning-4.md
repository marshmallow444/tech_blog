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

【Alpha Goの学習】  
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

## WaveNet

# 例題解説

# Appendix

# 実装演習