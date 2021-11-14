---
layout: post
title: "【ラビット・チャレンジ】深層学習 前編"
tags: ラビット・チャレンジ E資格 機械学習
---

<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

[ラビット・チャレンジ](https://ai999.careers/rabbit/)の受講レポート。  

---  

## プロローグ

### 識別と生成

| | 識別モデル<br>(discriminative, backward) | 生成モデル<br>(generative, forward) |
| --- | --- | --- |
| 目的 | データを目的のクラスに分類する | 特定のクラスのデータを生成する |
| 計算結果 | $p(C_k \| x)$ | $p(x \| C_k)$ |
| 具体的なモデル | 決定木<br>ロジスティック回帰<br>SVM<br>NN | HMM<br>ベイジアンネットワーク<br>VAE<br>GAN |
|  特徴  | 高次元→低次元<br>必要な学習データ：少 |  低次元→高次元<br>必要な学習データ：多 |

### 識別機(Clasifier)の開発アプローチ 

| | 生成モデル | 識別モデル | 識別関数 |
| --- | --- | --- | --- |
| 識別の計算 | $p(x \| C_k) \cdot p(c_k)$を推定<br>ベイズの定理より<br>$p(C_k \| x) =  \frac{p(x \| C_k) \cdot p(C_k)}{p(x)}$<br>ただし$p(x) = \sum_{k}p(x \| C_k) p(C_k)$ | $p(C_k \| x)$を推定<br>決定理論に基づき識別結果を得る<br>(閾値に基づく決定など) | 入力値$x$を直接クラスに写像(変換)する関数$f(x)$を推定 |
| モデル化の対象 | 各クラスの生起確率<br>データのクラス条件付き密度 | データがクラスに属する確率 | データの属するクラスの情報のみ<br>確率は計算されない |
| 特徴 | データを人工的に生成できる<br>確率的な識別 | 確率的な識別 | 学習量が少ない<br>決定的な識別 |


参考：「パターン認識と機械学習」(2007年)  
→生成モデルの研究が発達する前の分類方法  

### 識別器における生成モデルと識別モデル

+ 生成モデル
    + データのクラス条件付き密度の分布を推定
        + 分類結果より複雑、計算量が多い
+ 識別モデル
    + 直接データがクラスに属する確率を求める

### 識別器における識別モデルと識別関数

+ 識別モデル
    + 入力データから事後確率を推論して識別結果を決定
    + 識別結果の確率が得られる
+ 識別関数
    + 識別結果のみ得られる

### 万能近似定理と深さ

活性化関数をもつネットワークを使うことで、どんな関数でも近似できるのでは？という定理  

<br>

---

## Day1: NN

### ニューラルネットワークの全体像

入力層→中間層→出力層  

#### 確認テスト1

+ ディープラーニングは何をしようとしているか？
    + 自分の解答
        + 人間が具体的な数学モデルを直接構築するのではなく、ニューラルネットワークが入力データから特徴量を抽出することで、数学モデルを構築する
    + 模範解答
        + 明示的なプログラムの代わりに多数の中間層を持つニューラルネットワークを用いて、入力値から目的とする出力値に変換するを数学モデルを構築すること。
+ どの値の最適化が最終目標か？
    + 重み(W), バイアス(b)

#### 確認テスト2

次のネットワークを描く  

+ 入力層：2ノード1層
+ 中間層：3ノード2層
+ 出力層：1ノード1層

![test_2]({{site.baseurl}}/images/20211025.drawio.png)  

+ NNの対象とする問題の種類
    + 回帰
    + 分類

### 入力層〜中間層

$u = Wx + b$

1次関数において、  
+ $W$: 傾き
+ $b$: 切片 bias

$W, b$を学習する  

#### 確認テスト3  

![test_3]({{site.baseurl}}/images/20211025_1.png)  

#### 確認テスト

+ 以下の数式をPythonで書く  
    $u = Wx + b$

```
u = np.dot(x, W) + b
```

+ 中間層の出力を定義しているソース

```
# 2層の総入力
u2 = np.dot(z1, W2) + b2

# 2層の総出力
z2 = functions.relu(u2)
```

### 活性化関数

次の層への出力の大きさを決める<u>非線形の関数</u>  
次の層への信号のON/OFFや強弱を定める  

#### 確認テスト  

線形と非線形の違い  

+ 線形：直線  
    + 加法性と斉次性を満たす
+ 非線形：非直線  
    + 加法性と斉次性を満たさない

#### 中間層用の活性化関数

+ ReLU関数
    + 勾配消失問題の回避とスパース化
        + スパース化するとモデルの中身がシンプルになる

    $$
        f(x) = 
        \left\{
            \begin{array}{ll}
                x & (x > 0) \\  
                0 & (x \leqq 0) \\  
            \end{array}
        \right.
    $$

    ```Python
    # プログラム例
    def relu(x):
        return np.maximum(0, x)
    ```

+ シグモイド(ロジスティック)関数
    + 勾配消失問題

    $$
        f(u) = \frac{1}{1 + e^{-u}}
    $$

    ```Python
    # プログラム例
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    ```

+ ステップ関数
    + 線形分離可能なものしか学習できない

    $$
        f(x) = 
        \left\{
            \begin{array}{ll}
                1 & (x \geqq 0) \\  
                0 & (x < 0) \\  
            \end{array}
        \right.
    $$

    ```Python
    # プログラム例
    def step_function(x):
        if x > 0:
            return 1
        else:
            return 0
    ```

#### 確認テスト

ソースコードのうち、以下に該当する箇所を抜き出す  

$$
    z = f(u)
$$

```Python
# 1層の総出力
z1 = functions.relu(u1)
```

```Python
# 2層の総出力
z2 = functions.relu(u2)
```

### 出力層

+ 役割
    + 問題に対する判定結果を出力する  
+ 誤差関数
    + 訓練データを入力し、NNの出力した判定結果と期待した判定結果の誤差を求める
    + 例：二乗誤差

        $$
            E_n(w) = \frac{1}{2} \sum_{j=1}^{J} (y_j - d_j)^2 = \frac{1}{2} ||(y - d)||^2
        $$

        ```Python
        # 平均二乗誤差のコード例
        def mean_squared_error(d, y):
            return np.mean(np.square(d - y)) / 2
        ```

#### 確認テスト

+ なぜ引き算ではなく二乗するか？
    + 引き算の場合正負の符号の差が出てしまい、全体の誤差を正しく表すのに都合が悪いため
    + 二乗してそれぞれのラベルでの誤差を正の値にする
+ 上記二乗誤差の式の$\frac{1}{2}$はどういう意味をもつか？
    + 誤差逆伝搬の計算において、誤差関数の微分を用いる際の計算を簡単にするため(本質的な意味はない)

#### 出力層の活性化関数

+ ソフトマックス関数
+ 恒等写像
+ シグモイド関数

中間層との違い

+ 値の強弱
    + 中間層：しきい値の前後で信号の強弱を調整
    + 出力層：信号の大きさ(比率)はそのままに変換
+ 確率出力
    + 分類問題の場合、以下が必要
        + 出力層の出力は0~1の範囲に限定
        + 総和を1とする

出力層の種類 - 全結合NN

| | 回帰 | 二値分類 | 多クラス分類 |
| --- | --- | --- | --- |
| 活性化関数 |  恒等写像 | シグモイド関数 | ソフトマックス関数 |
| 誤差関数 | 二乗誤差 | 交差エントロピー | 交差エントロピー |

+ 恒等写像

    $$
        f(u) = u
    $$

+ シグモイド関数

    $$
        f(u) = \frac{1}{1 + e^{-u}}
    $$

    ```Python
    # シグモイド関数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    ```

+ ソフトマックス関数

    $$
        f(i, u) = \frac{e^{u_i}}{\sum_{k=1}^{K} e^{u_k}}
    $$

    ```Python
    # ソフトマックス関数
    def softmax(x): 
        if x.ndim == 2:
            # ミニバッチのときの処理
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            rerurn y.T
        
        x = x - np.max(x)   # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x))
    ```

【訓練データサンプルあたりの誤差】  

+ 二乗誤差 (実際には平均二乗誤差が用いられることが多い)
    
    $$
        E_n(W) = \frac{1}{2} \sum_{i=1}^{I} (y_n - d_n)^2
    $$

+ 交差エントロピー

    $$
        E_n(W) = - \sum_{i=1}^{I} d_i \log y_i
    $$

【学習サイクルあたりの誤差】  

$$
    E(W) = \sum_{n=1}^{N} E_n
$$

#### 確認テスト

(1) ~ (3)の数式に該当するソースコードを示し、一行ずつ処理の説明をせよ  

$$
    \overbrace{f(i, u)}^{(1)}  = \frac{\overbrace{e^{u_i}}^{(2)} }{ \underbrace{\sum_{k=1}^{K} e^{u_k}}_{(3)} }
$$

+ (1): `def softmax(x):`
+ (2): `np.exp(x)`
    + 1クラス分の確率
+ (3): `np.sum(np.exp(x))`
    + 全クラス分の確率の和  
$\space$  
+ 各行の説明

    ```Python
    def softmax(x): 
        if x.ndim == 2:
            # ミニバッチのときの処理
            x = x.T                                     # 転置
            x = x - np.max(x, axis=0)                   # オーバーフロー対策
            y = np.exp(x) / np.sum(np.exp(x), axis=0)   # softmaxの値を計算する
            return y.T                                  # 転置して返す
        
        x = x - np.max(x)                               # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x))            # softmaxの値を計算して返す
    ```

#### 確認テスト

【交差エントロピー】  

(1), (2)の数式に該当するソースコードを示し、1行ずつ処理の説明をせよ  

$$
    \overbrace{E_n(w)}^{(1)}  = \overbrace{- \sum_{i=1}^{I} d_i \log y_i}^{(2)}
$$

+ (1): `def cross_entropy_error(d, y):`
+ (2): `-np.sum(np.log(y[np.arange(batch_size), d] + 1e-7)) `  
$\space$  
+ 各行の説明  
    ```Python
    def cross_entropy_error(d, y):
        if y.ndim == 1:                 # 1次元行列の場合
            d = d.reshape(1, d.size)    # 2次元行列に変形
            y = y.reshape(1, y.size)    # 2次元行列に変形
        
        if d.size == y.size:
            # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
            d = d.argmax(axis=1)        # 配列内の最大要素のインデックスを取得
        
        batch_size = y.shape[0]         # バッチサイズを取得
        return -np.sum(np.log(y[np.arange(batch_size), d] + 1e-7)) / batch_size # 交差エントロピーを計算して返す
    ```
    + `y`: 0か1の配列
    + `batch_size`: ミニバッチの何番目？
    + `d`: d番目は0 or 1?
    + `1e-7`: 対数関数では0のとき$- \infty$に飛ぶ。これを防ぐための処理  

### 勾配降下法

+ 勾配降下法
+ 確率的勾配降下法
+ ミニバッチ勾配降下法

パラメータ$w$を最適化  
誤差を最小にする$w$を見つける  

#### 勾配降下法  

全サンプルの平均誤差  

$$
    w^{(t + 1)} = w^{(t)} - \epsilon \nabla E\\  
    \space \\  
    \nabla E = \frac{\partial E}{\partial w} = 
    \biggl[
        \frac{\partial E}{\partial w_1} \cdots \frac{\partial E}{\partial w_M}
    \biggr]
$$

→前回の値$w^{(t)}$から誤差$\epsilon \nabla E$を引いた値が今回の値$w^{(t + 1)}$  

【$\epsilon$：学習率】  
+ 大きすぎる場合：
    + 最小値にたどり着かず、発散する
+ 小さすぎる場合：
    + 収束するまでに時間がかかる
    + 局所解に陥る場合あり

【認識に必要なデータ数】  
画像分類：1クラスあたり1000~5000枚あると精度が出る  
自然言語モデル：Wikipediaの全てのデータを100~200エポック学習すると、ある程度結果が出る

#### 確率的勾配降下法 (SGD)

ランダムに抽出したサンプルの誤差   

$$
    w^{(t + 1)} = w^{(t)} - \epsilon \nabla E_n\\  
$$

メリット：  

+ データが冗長な場合の計算コスト軽減
+ 望まない局所極小解に収束するリスクを軽減
+ オンライン学習ができる

#### 確認テスト

+ オンライン学習とは？
    + 学習データを入力すると、その都度パラメータを更新する
+ バッチ学習とは？
    + 予め全データを準備しておき、一気に全ての学習データを更新する

メモリの容量には限りがあるので、オンライン学習を使うことが多い  

#### ミニバッチ勾配降下法

ミニバッチ$D_t$に属するサンプルの平均誤差  
(ミニバッチ：ランダムに分割したデータの集合)  

$$
    \begin{split}
        &w^{(t + 1)} = w^{(t)} - \epsilon \nabla E_t \\  
        &E_t = \frac{1}{N_t} \sum_{n \in D_t} E_n \\  
        &N_t = | D_t | \\  
        & (N_t: バッチ数)  
    \end{split}
$$

メリット：  

+ 計算機の計算資源を有効利用
    + スレッド並列化(CPU)やSIMD並列化(GPU)
        + **SIMD**: Single Instruction Multi Data
            + 一つの命令を同時に並列実行
    + 各バッチに対する処理を並列にできる

#### 確認テスト

以下の数式の意味を図に描いて説明せよ  

$$
    w^{(t + 1)} = w^{(t)} - \epsilon \nabla E_t
$$

![test_mini_batch]({{site.baseurl}}/images/20211027.drawio.png)  

1. 1エポック目を学習する
1. 2エポック目は、1エポック目での間違いに学習率をかけた分を修正して学習する
1. 3エポック目以降も、同様に学習していく

### 誤差勾配の計算

どう計算する？  

$$
    \nabla E = \frac{\partial E}{\partial w} = 
    \biggl[
        \frac{\partial E}{\partial w_1} \cdots \frac{\partial E}{\partial w_M}
    \biggr]
$$

【数値微分】  
プログラムで微小な数値を生成し、擬似的に微分を計算する  

$$
    \frac{\partial E}{\partial w_m} \approx \frac{E(w_m + h) - E(w_m - h)}{2h}
$$

$h$: 微小な値  
$m$番目の$w$を微小変化させた状態で、  
全ての$w$について誤差$E$を計算  

デメリット：計算量が多く、負荷が大きい  

→代わりに**誤差逆伝播法**を利用する  

### 誤差逆伝播法

算出された誤差を、出力層から順に微分し、前の層へ順に伝播  
最小限の計算で各パラメータでの微分値を<u>解析的に</u>計算する方法  

![back_propagation]({{site.baseurl}}/images/20211027_1.drawio.png)  

dとy(正解と予測結果)から誤差$E(y)$を計算する  
誤差から微分を逆算する  
→不要な再帰的計算を避けて微分を算出できる  

#### 確認テスト

すでに行った計算結果を保持するソースコードを抽出せよ  

```Python
# 出力層でのデルタ
delta2 = functions.d_mean_squared_error(d, y)
```

【誤差勾配の計算】  

$$
    \begin{array}{ll}
        E(y) = \frac{1}{2} \sum_{j=1}^{J} (y_j - d_j)^2 = \frac{1}{2} || y - d ||^2 & : 誤差関数 = 二乗誤差関数 \\  
        y = u^{(u)} & : 出力層の活性化関数 = 恒等写像 \\  
        u^{(l)} = w^{(l)} z{(l-1)}  + b^{(l)} & : 総入力の計算
    \end{array}
$$

$$
    \begin{split}
        &\frac{\partial E}{\partial w_{ji}^{(2)}} = 
        \overbrace{
            \frac{\partial E}{\partial y}
        }^{【1】}
        \overbrace{
            \frac{\partial y}{\partial u}
        }^{【2】}
        \overbrace{
            \frac{\partial u}{\partial w_{ji}^{(2)}}
        }^{【3】} \\  
        \space \\  
        &【1】\frac{\partial E(y)}{\partial y} = \frac{\partial}{\partial y} \frac{1}{2} || y  - d ||^2 = y - d \\  
        &【2】\frac{\partial y(u)}{\partial u} = \frac{\partial u}{\partial u} = 1 \\  
        &【3】\frac{\partial u(w)}{\partial w_{ji}} = \frac{\partial}{\partial w_{ji}}(w^{(l)} z^{(l-1)} + b^{(l)})
        = \frac{\partial}{\partial w_{ji}}
        \left(
            \left[
                \begin{array}{c}
                    w_{11} z_1 + \cdots + w_{1i z_i} + \cdots + w_{1I} z_I \\  
                    \vdots \\  
                    w_{j1} z_1 + \cdots + w_{ji z_i} + \cdots + w_{jI} z_I \\  
                    \vdots \\  
                    w_{J1} z_1 + \cdots + w_{Ji z_i} + \cdots + w_{JI} z_I \\  
                \end{array}
            \right] +
            \left[
                \begin{array}{c}
                    b_1 \\  
                    \vdots \\  
                    b_j \\  
                    \vdots \\  
                    b_J \\  
                \end{array}
            \right]
        \right)
        =
        \left[
                \begin{array}{c}
                    0 \\  
                    \vdots \\  
                    z_i \\  
                    \vdots \\  
                    0 \\  
                \end{array}
            \right]
    \end{split}
$$

【3】：$w_{ji}$という一つの項目について微分している。そのため、その他の行は全て0になってしまう  

$$
    \Rightarrow
    \frac{\partial E}{\partial w_{ji}^{(2)}}
    = \frac{\partial E}{\partial y} \frac{\partial y}{\partial u} \frac{\partial u}{\partial w_{ji}^{(2)}} = (y - d) \cdot 1 \cdot
    \left[
        \begin{array}{c}
            0 \\  
            \vdots \\  
            z_i \\  
            \vdots \\  
            0
        \end{array}
    \right]
    = (y_j - d_j) z_i
$$

#### 確認テスト

以下の数式に該当するソースコードを探せ  

+ $\frac{\partial E}{\partial y}$
    + `delta2 = functions.d_mean_squared_error(d, y)`
+ $\frac{\partial E}{\partial y} \frac{\partial y}{\partial u}$
    + `delta1 = np.dot(delta2, W2.T) * functions.d_sigmoid(z1)`
+ $\frac{\partial E}{\partial y} \frac{\partial y}{\partial u} \frac{\partial u}{\partial w_{ji}^{(2)}}$
    + `grad['W1'] = np.dot(x.T, delta1)`

#### ディープラーニングの開発環境  

+ ローカル (下にいくほど速い)
    + CPU ¥8,000程度
    + GPU ¥20,000程度
    + (FPGA) ¥100,000程度
        + 自分でプログラムできる計算機
        + あるプログラムだけ高速処理できる。他は遅い
    + ASIC(TPU) ¥数億
        + プログラム不可の計算機
        + あるプログラムを高速実行できるように工場で生産する
        + TPU(by Google)は機械学習用
+ クラウド
    + AWS
    + GCP
        + TPUを提供

### その他のトピック

#### 入力層の設計

+ 入力としてとり得るデータ (数値の集まり)
    + 連続する実数
    + 確率
    + フラグ値
+ 入力として取るべきでないデータ
    + 欠損値が多いデータ
    + 誤差が多いデータ
    + 出力そのもの、出力を加工した情報
        + 前段階の学習に人の手が介在することになるので、ふさわしくない
        + end-to-endが理想的
    + 連続性のないデータ
    + 無意味な数が割り当てられているデータ
+ 欠損値の扱い
    + ゼロで詰める
    + 欠損値を含む集合を除外(行を除外)
    + 入力として採用しない(列を除外)
+ データの結合
+ 数値の正規化・正則化

#### 過学習

+ 巨大なNNで発生しやすい
+ テストデータの誤差が小さくならない
+ 予防策：ドロップアウト

---

## エピローグ

### データ集合の拡張

#### 入力データ?

+ 学習データが不足するときに、人工的にデータを作り水増しする
+ 分類タスク(画像認識)に効果が高い
    + 様々な増やし方がある
        + オフセット、ノイズ、ぼかし、回転、クロップなど
    + 様々な変換を組み合わせる
+ 密度推定のためのデータは水増し不可
    + 水増しには密度の情報が必要
+ データ拡張の結果、データセット内で混同するデータが発生しないよう要注意

#### ノイズ注入によるデータ拡張

+ 中間層へのノイズ注入で、様々な抽象化レベルでのデータ拡張が可能

+ データ拡張の効果なのか？モデル性能か？
    + 見極めが重要
+ データ拡張と性能評価
    + 汎化性能が向上(しばしば劇的に)
    + ランダムなデータ拡張では再現性に注意
+ データ拡張とモデルの捉え方
    + 一般的に適用可能なデータ拡張はモデルの一部として捉える
        + 例：ノイズ付加、ドロップアウトなど
    + 特定の作業に対してのみ適用可能なデータ拡張は、入力データの事前加工として捉える
        + 例：判定対象の画像が製品全体の画像であるとき、製品の一部をクロップしたものを学習してもあまり意味がない

### CNNで扱えるデータの種類

次元間で繋がりのあるデータを扱える  

| | 1次元 | 2次元 | 3次元 |
| --- | --- | --- | --- |
| 単一チャンネル | 音声<br>[時刻、強度] | フーリエ変換した音声<br>[時刻、周波数、強度] |  CTスキャン画像<br>[x, y, z, 強度] |
| 複数チャンネル | アニメのスケルトン<br>[時刻、(腕の値、膝の値...)] | カラー画像<br>[x, y, (R, G, B)] | 動画<br>[時刻、x, y, (R, G, B)]

隣り合うデータが急に変化するのではなく、ある程度ゆるやかに変化する  

### 特徴量の転移

入力データに近い層の特徴量は、色々な画像に応用可能  
基本的な特徴量の抽出処理とタスク固有の処理を、別々のパーツに分けることができる  
ベースモデル：基本的な学習済みの重み  
+ ファインチューニング
    + ベースモデル重みを再学習
+ 転移学習
    + ベースモデルの重みを固定

| | 基本的な特徴量抽出 |  タスク固有処理 |
| --- | --- | --- | --- |
| 学習コスト(計算量) | 高 | 低 |
| 必要なデータ量 | 多 | 少 |
| | VGG(画像)<br>BERT(テキスト) |  全結合層による分類層 |

プリトレーニング(基本的な特徴量抽出)がうまくいくと、その分野の学習精度が一気に上がる  

---

## Day2

## 深層モデルのための学習テクニック

### 勾配消失問題について

誤差逆伝播法が下位層へ進んでいくにつれ、勾配が緩やかになっていく  
→勾配降下法による更新では、下位層のパラメータがほとんど変わらず、  
　訓練は最適解に収束しなくなる  

#### 確認テスト

連鎖律の原理を使い、dz / dxを求めよ  

$$
    z = t^2 \\  
    t = x + y
$$

解答：

$$
    \begin{split}
        \frac{dz}{dx} &= \frac{dt}{dx} \cdot \frac{dz}{dt} \\  
        \frac{dt}{dx} & = 1 \\  
        \frac{dz}{dt} &= 2t \\  
        \frac{dz}{dx} &= 2t \\  
        &= 2(x + y)
    \end{split}
$$

+ シグモイド関数
    + 勾配消失問題を起こしやすい
    + 微分すると最大0.25

#### 確認テスト1

シグモイド関数を微分したとき、入力値が0のときに最大値をとる  
その値として正しいのは？  

【答え】0.25  

勾配消失の解決方法:  

+ 活性化関数の選択
+ 重みの初期値設定
+ バッチ正規化

#### 活性化関数

+ ReLU関数

    ```Python
    # サンプルコード
    def relu(x):
        return np.maximum(0, x)
    ```

    $$
        f(x) = 
        \left\{
            \begin{array}{ll}
                x & (x > 0) \\  
                0 & (x \leqq 0) \\  
            \end{array}
        \right.
    $$

    + 微分結果が1か0になる
        + 勾配消失問題の回避
        + スパース化
            + 必要な部分だけ使用し、あまり役立たない部分は切り捨てられる
+ ソフトマックス
+ tanh(ハイパボリックタンジェント)
+ [Leaky ReLU](https://atmarkit.itmedia.co.jp/ait/articles/2005/13/news009.html)

#### 初期値の設定方法

重みに乱数を使う理由：  
入力に対していろんな見方をしたいため  
どういう見方をしたらうまくいくか？を求めたい  

+ Xavier
    + Xavierの初期値を設定する際の活性化関数
        + ReLU
        + シグモイド(ロジスティック関数)
        + 双曲線正接関数
        + →S字カーブ型の関数に対してうまくはたらく
    + 初期値の設定方法
        + 重みの要素を、前の層のノード数の平方根で除算した値

    ```Python
    # サンプルコード：Xavierの初期値
    network['W1'] = np.random.randn(input_layer_size, hidden_layer_size) / np.sqrt(input_layer_size)
    network['W2'] = np.random.randn(hidden_layer_size, output_layer_size) / np.sqrt(hidden_layer_size)
    ```

    + Xavierがよい理由
        + 当初は[標準正規分布](https://toukei.link/probabilitydistributions/standardnormal/)がよく用いられたが、逆誤差伝播法で勾配消失問題が発生する
        + →標準正規分布で重みを初期化したとき、各レイヤーの出力は0と1に偏る  
        + →0や1のときは微分値がほとんど0  
        + →標準偏差を小さくする(例：0.01)と、出力値のほとんどが0.5近辺になる  
        + →Xavierを使うと出力値の分布がいい感じにばらける
+ He(ヒー)
    + Heの初期値を設定する際の活性化関数
        + ReLU関数
    + 初期化の設定方法
        + 重みの要素を、前の層のノード数の平方根で除算した値に対し、$\sqrt{2}$をかけ合わせた値
            + = 正規分布の重みを$\sqrt{\frac{2}{n}}$の標準偏差の分布にする($n$は前の層のノード数)

    ```Python
    # サンプルコード：Heの初期値
    network['W1'] = np.random.randn(input_layer_size, hidden_layer_size) / np.sqrt(input_layer_size) * sqrt(2)
    network['W2'] = np.random.randn(hidden_layer_size, output_layer_size) / np.sqrt(hidden_layer_size) * sqrt(2)
    ```

    + Heがよい理由
        + 標準正規分布に基づいた重みを用いてReLU関数を通すと表現力が全くなくなる(ほとんど0)
        + →標準偏差を小さくしても、表現力はなくなっていく
        + →He初期化すると、0 ~ 1の範囲の分布が増える

#### 確認テスト

重みの初期値に0を設定すると、どのような問題が発生するか？  

【解答】  
正しい学習が行われない  
→全ての重みの値が均一に更新されるため  
　多数の重みをもつ意味がなくなる  

#### バッチ正規化

ミニバッチ単位で、入力値のデータの偏りを抑制する手法  

+ 使い所
    + 活性化関数へ値を渡す前後に、バッチ正規化の処理を孕んだ層を加える
        + バッチ正規化層への入力
            + $u^{(l)} = w^{(l)} z^{(l-1)} + b^{(l)}$
            + または$z$
+ ミニバッチのサイズ(画像の場合)
    + GPUの場合：1 ~ 64枚 
    + TPUの場合：1 ~ 256枚
    + 小さい分には問題ない
    + 8(または2)の倍数にすることが多い
        + ハードウェアの制限のため
+ メリット
    + NNの学習が安定化、スピードアップ
    + 過学習を抑制
+ 数学的記述  

    $$
        \begin{array}{ll}
            1. \quad \mu_t = \frac{1}{N_t} \sum_{i=1}^{N_t} x_{ni} & : ミニバッチの平均 \\  
            2. \quad \sigma_t^2 = \frac{1}{N_t} \sum_{i=1}^{N_t} (x_{ni} - \mu_t)^2 & : ミニバッチの分散 \\  
            3. \quad \hat x_{ni} = \frac{x_{ni} - \mu_t}{\sqrt{\sigma_t^2  + \theta}} & : ミニバッチの正規化 \\  
            4. \quad y_{ni} = \gamma x_{ni} + \beta & : 変倍・移動
        \end{array}
    $$

    (1. ~ 3.は統計的正規化、4.はNNでの扱いを良くするための調整)  

    + 処理および記号の説明
        + $\mu_t$: ミニバッチ$t$全体の平均
        + $\sigma_t^2$: ミニバッチ$t$全体の標準偏差
        + $N_t$: ミニバッチのインデックス
        + $\hat x_{ni}$: ０に値を近づける計算(0を中心とするセンタリング)と正規化を施した値
        + $\gamma$: スケーリングパラメータ
        + $\beta$: シフトパラメータ
        + $y_{ni}$: ミニバッチのインデックス値とスケーリングの積にシフトを加算した値  
            (バッチ正規化オペレーションの出力)

#### 確認テスト

一般的に考えられるバッチ正規化の効果を2点挙げよ  

+ NNの学習が安定化、スピードアップ
+ 過学習を抑制

#### 例題チャレンジ

特徴データ`data_x`, ラベルデータ`data_t`に対してミニバッチ学習を行うプログラム  
空欄(き)に当てはまるものは？  

【解答】  

`data_x[i:i_end], data_t[i:i_end]`  
→バッチサイズだけデータを取り出す処理  

E資格ではプログラムの穴埋め問題がよく出る  

### 学習率最適化手法について

学習率の決め方

初期の指針

+ 初期は大きく、徐々に小さくしていく
+ パラメータごとに学習率を可変

→<u>学習率最適化手法を利用して学習率を最適化</u>

#### モメンタム

誤差をパラメータで微分したものと学習率の積を減算した後、  
現在の重みに前回の重みを減算した値と慣性の積を加算  
株価の移動平均のような、なめらかな動きをする  

$$
    \begin{split}
        &V_t = \mu V_{t-1} -  \epsilon \nabla E \\  
        &w^{(t+1)} = w^{(t)} + V_t \\  
    \end{split}
$$

+ 慣性：$\mu$   
+ Vには基本的にwと同じ値が入る(意味合いが異なる)  

```Python
self.v[key] = self.momentum * self.v[key] - self.learning_rate * grad[key]
params[key] += self.v[key]
```

+ コード上の各変数の意味
    + `self.momentum` : $\mu$
    + `self.learning_rate`: $\epsilon$
    + `self.grad[key]` : $\nabla E$

+ メリット
    + 局所的最適解にはならず、大域的最適解となる
    + 谷間についてから最も低い位置(最適値)にいくまでの時間が早い

#### AdaGrad

誤差をパラメータで微分したものと  
再定義した学習率の積を減算する

$$
    \begin{array}{ll}
        h_0 = \theta & : 何かしらの値でhを初期化 \\  
        h_t = h_{t-1} + (\nabla E)^2 & : 計算した勾配の2乗を保持 \\  
        w^{(t+1)} = w^{(t)} - \epsilon \frac{1}{\sqrt{h_t} + \theta} \nabla E & : 現在の重みを、適応させた学習率で更新
    \end{array}
$$

```Python
self.h[key] = np.zeros_like(val)
self.h[key] += grad[key] * grad[key]
params[key] -= self.learning_rate * grad[key] / (np.sqrt(self.h[key]) + 1e-7)
```

+ コード上の各変数の意味
    + `val` : $\theta$
    + `1e-7` : 2個目の$\theta$
        + 計算がうまくいくように、適当な値を足している

+ メリット
    + 勾配の緩やかな斜面に対して、最適解に近づける
+ 課題
    + 学習率が徐々に小さくなるので、**鞍点問題**を引き起こすことがあった

#### RMSProp

AdaGradの改良版、似たような動きをする  
誤差をパラメータで微分したものと  
再定義した学習率の積減算する  

$$
    \begin{split}
        &h_t = ah_{t-1} + (1 - \alpha)(\nabla E)^2 \\  
        &w^{(t+1)} = w^{(t)} - \epsilon \frac{1}{\sqrt{h_t} + \theta} \nabla E
    \end{split}
$$

```Python
self.h[key] *= self.decay_rate
self.h[key] += (1 - self.decay_rate) * grad[key] * grad[key]
params[key] -= self.learning_rate * grad[key] / (np.sqrt(self.h[key]) + 1e-7)
```

+ コード上の各変数の意味
    + `self.decay_rate` : $\alpha$
+ メリット
    + 局所的最適解にはならず、大域的最適解となる
    + ハイパーパラメータの調整が必要な場合が少ない

#### Adam

以下をそれぞれ孕んだ最適化アルゴリズム  

+ モメンタムの、過去の勾配の指数関数的減衰平均
+ RMSPropも、過去の勾配の2乗の指数関数的減衰平均

+ メリット
    + モメンタムとRMSPropのメリットを孕む
    + 鞍点問題もクリアしやすい
    + スムーズに学習が進む

### 過学習

テスト誤差と訓練誤差とで学習曲線が乖離がすること  
ネットワークの自由度が高いと起こりやすい  

+ 入力値：少、NN：大
+ パラメータ数が適切でない
+ など

#### L1正則化、L2正則化

ネットワークの自由度を制約する  
→過学習を防ぐ  

【Weight decay(荷重減衰)】  
+ 過学習の原因：重みが大きい値をとる(と、過学習が発生することがある)
+ 過学習の解決策：誤差に対して正則化項を加算することで、重みを抑制

【数式】  

$$
    \begin{array}{ll}
        E_n(w) + \frac{1}{p} \overbrace{\lambda}^{hyper \space parameter} || x ||_p & : 誤差関数に、pノルムを加える \\  
        || x ||_p = 
        \Bigl(
            |x_1|^p  + \cdots + |x_n|^p
        \Bigr)^{\frac{1}{p}} & : pノルムの計算
    \end{array}
$$

+ L1正則化：$p = 1$の場合。**Lasso回帰**
+ L2正則化：$p = 2$の場合。**Ridge回帰**

ノルム＝距離  

例：  
点(x, 0)から点(0, y)までの距離  
+ ユークリッド距離：$\sqrt{x^2 + y^2} \quad$ ←p2ノルム  
+ マンハッタン距離：$x + y \quad \quad \quad$ ←p1ノルム  

【正則化の計算】  

$$
    \begin{split}
        &||W^{(1)}||_p = (|W_1^{(1)}|^p + \cdots + |W_n^{(1)}|^p)^{\frac{1}{p}} \\  
        &||W^{(2)}||_p = (|W_1^{(2)}|^p + \cdots + |W_n^{(2)}|^p)^{\frac{1}{p}} \\  
        &||x||_p = ||W^{(1)}||_p + ||W^{(2)}||_p \\  
        &E_n(w) + \underbrace{\frac{1}{p} \lambda ||x||_p}_{正則化項} 
    \end{split}
$$

```Python
# サンプルコード
np.sum(np.abs(network.params['W' + str(idx)]))
weight_decay += weight_decay_lambda * np.sum(np.abs(network.params['W' + str(idx)]))
loss = network.loss(x_batch, b_batch) + weight_decay
```

[![Ridge](https://image.slidesharecdn.com/random-131223004858-phpapp02/95/prml-49-638.jpg?cb=1420232764)](https://image.slidesharecdn.com/random-131223004858-phpapp02/95/prml-49-638.jpg?cb=1420232764)  
(画像：[https://www.slideshare.net/yasunoriozaki12/prml-29439402](https://www.slideshare.net/yasunoriozaki12/prml-29439402))  

[![Lasso](https://image.slidesharecdn.com/random-131223004858-phpapp02/95/prml-50-638.jpg?cb=1420232764)](https://image.slidesharecdn.com/random-131223004858-phpapp02/95/prml-50-638.jpg?cb=1420232764)  
(画像：[https://www.slideshare.net/yasunoriozaki12/prml-29439402](https://www.slideshare.net/yasunoriozaki12/prml-29439402))  

→スパース化(ReLU関数のときのように)  

#### 確認テスト

L1正則化を表しているグラフは？  

【解答】右(Lasso推定量)  

[![L1_L2](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.ap-northeast-1.amazonaws.com%2F0%2F610167%2F1b82d44f-fbca-be85-2e93-30a0f2857c51.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=4d306778e3717dafaca6dfc9c6595624)](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.ap-northeast-1.amazonaws.com%2F0%2F610167%2F1b82d44f-fbca-be85-2e93-30a0f2857c51.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=4d306778e3717dafaca6dfc9c6595624)  
(画像：[https://qiita.com/c60evaporator/items/784f0640004be4eefc51](https://qiita.com/c60evaporator/items/784f0640004be4eefc51))  

この図の意味：上の3Dグラフを上から見て、等高線を引いた感じ  
+ 右上の同心楕円：誤差関数の等高線
+ L2の円、L1の正方形(左下)：正則化項の等高線
+ 同心楕円と左下の[円 / 正方形]の交点：誤差関数と正則化項の最小値
    + 誤差関数の最小値：同心楕円の中心
    + 正則化項の最小値：[円 / 正方形]の中心

L1正則化では、$w1$方向の重みが0になる  

#### 例題チャレンジ

+ パラメータ正則化
    + L2正則化の最終的な勾配を計算するコードは？
        + `grad += rate * param` 
            + ↑「勾配」なので微分した結果
    + L1正則化の最終的な勾配を計算するコードは？
        + `x = np.sign(param)`
            + ↑あるパラメータに着目。0未満の傾きは-1, 0以上の傾きは1

#### ドロップアウト

ランダムにノードを削除して学習させる  

メリット：  
データ量を変化させずに、異なるモデルを学習させていると解釈できる  
→過学習の抑制につながる  

## 畳み込みニューラルネットワークについて

### 畳み込みニューラルネットワークの概念

次元間で繋がりのあるデータを扱える  

例：LeNetの構造図  

[![LeNet](https://cdn-ak.f.st-hatena.com/images/fotolife/k/kasuya_ug/20200214/20200214183657.png)](https://cdn-ak.f.st-hatena.com/images/fotolife/k/kasuya_ug/20200214/20200214183657.png)  
(画像：[https://buildersbox.corp-sansan.com/entry/2020/02/28/110000](https://buildersbox.corp-sansan.com/entry/2020/02/28/110000))  

データの数の変化  
(32, 32) -> (28, 28, 6) -> (14, 14, 6) -> (10, 10, 16) -> (5, 5, 16) -> (120,) -> (84,) -> (10,)  

+ 前半部分(〜S4の層)
    + 次元のつながりを保つ
    + 特徴量の抽出
+ 後半部分(C5〜の層)
    + 全結合層
    + 人間が欲しい結果を出す
+ フィルタについて
    + 例：C1の層では、6種類のフィルタを使って学習する

【畳み込み層】  

(入力値) * (フィルター) → (出力値) + (バイアス) → (活性化関数) → (出力値)  
$\Rightarrow$データの繋がりを反映させることができる  

[計算方法のイメージ]  
フィルターを左上から一定間隔ずつずらしながら、出力値を計算していく  
フィルター：重み  

[![conv](https://ainow.ai/wp-content/uploads/2021/09/8cdcf5887162a040d0a54e9861a836ef.jpg)](https://ainow.ai/wp-content/uploads/2021/09/8cdcf5887162a040d0a54e9861a836ef.jpg)  
(画像：[https://ainow.ai/2021/09/16/258469/](https://ainow.ai/2021/09/16/258469/))  

#### 畳み込み層

3次元の空間情報も学習できる  

+ 全結合層のデメリット
    + 画像の場合3次元データだが(w * h * ch)、1次元のデータとして処理される
        + →RGBの各チャンネル間の関連性が、学習へ反映されない

畳み込み層ではこの問題が解決される  

【バイアス】  

(入力値) * (フィルター)の値へ足す  
$xW + \underbrace{b}_{バイアス}$  

[![bias](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.amazonaws.com%2F0%2F70897%2F1cf58b09-85a6-01a5-cf71-b60dab16b8c8.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=c54da9b7a6618eeb11038ab89bd86117)](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.amazonaws.com%2F0%2F70897%2F1cf58b09-85a6-01a5-cf71-b60dab16b8c8.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=c54da9b7a6618eeb11038ab89bd86117)  
(画像：[https://qiita.com/nvtomo1029/items/601af18f82d8ffab551e](https://qiita.com/nvtomo1029/items/601af18f82d8ffab551e))  

【パディング】  

出力データのサイズが小さくなることを防ぐために、入力データの周りにデータを足す  
(以下の画像の例では0にしているが、0以外でもよい。一番近い値など)  

[![padding](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.amazonaws.com%2F0%2F70897%2F7c2de1fc-ca68-699f-3ac5-d927a0ae52c5.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=e81d890d74dd3a0b41f41faac5680af9)](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.amazonaws.com%2F0%2F70897%2F7c2de1fc-ca68-699f-3ac5-d927a0ae52c5.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=e81d890d74dd3a0b41f41faac5680af9)  
(画像：[https://qiita.com/nvtomo1029/items/601af18f82d8ffab551e](https://qiita.com/nvtomo1029/items/601af18f82d8ffab551e))  

【ストライド】  

フィルタを一度にずらす幅  

[![stride](https://assets.st-note.com/production/uploads/images/7872657/picture_pc_0e10907a9e23465579518ac853c34a4b.jpg)](https://assets.st-note.com/production/uploads/images/7872657/picture_pc_0e10907a9e23465579518ac853c34a4b.jpg)  
(画像：[https://note.com/ryuwryyy/n/nfd0b8ff862aa](https://note.com/ryuwryyy/n/nfd0b8ff862aa))  

【チャンネル】  

フィルタの数  
以下の画像の例では3  

[![channel](https://assets.st-note.com/production/uploads/images/7872703/picture_pc_8f512bbb99ee456c80b95841b5b7ab35.jpg)](https://assets.st-note.com/production/uploads/images/7872703/picture_pc_8f512bbb99ee456c80b95841b5b7ab35.jpg)  
(画像：[https://note.com/ryuwryyy/n/nfd0b8ff862aa](https://note.com/ryuwryyy/n/nfd0b8ff862aa))


プログラム上では、計算を高速化するために入力データの行列を変形させてから計算する  
例えば以下のような行列があったら、  
[![matrices](https://miro.medium.com/max/1258/1*OyklyL9egRmf6tOfp-8eiA.png)](https://miro.medium.com/max/1258/1*OyklyL9egRmf6tOfp-8eiA.png)  
(画像：[https://medium.com/@\_init\_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544](https://medium.com/@_init_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544))  

以下のように変形する  
[![transformed_matrix](https://miro.medium.com/max/1168/1*RLH7W_baMCmNEdXR6cvahQ.png)](https://miro.medium.com/max/1168/1*RLH7W_baMCmNEdXR6cvahQ.png)  
(画像：[https://medium.com/@\_init\_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544](https://medium.com/@_init_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544))  

→重みの行列との計算が簡単にできる形になる  

#### プーリング層

+ Max Pooling: 対象領域のMax値を取得
+ Avg. Pooling: 対象領域の平均値を取得

この層に重みはない  

#### 確認テスト

サイズ6 * 6の入力画像を、サイズ2 * 2のフィルタで畳み込んだときの出力画像のサイズは？  
なお、ストライドとパディングは1とする  

【解答】6 * 6  

公式  

$$
    \begin{split}
        &O_H = \frac{画像の高さ + 2 * パディング高さ - フィルター高さ}{ストライド} + 1 \\  
        &\space \\  
        &O_W = \frac{画像の幅 + 2 * パディング幅 - フィルター幅}{ストライド} + 1
    \end{split}
$$

### 初期のCNN

#### AlexNet

5層の畳み込み層及びプーリング層など、それに続く3層の全結合層から構成される  

[![AlexNet](https://ml4a.github.io/images/figures/alexnet.jpg)](https://ml4a.github.io/images/figures/alexnet.jpg)  
(画像：[https://ml4a.github.io/ml4a/jp/convnets/](https://ml4a.github.io/ml4a/jp/convnets/))  

過学習の防止
→サイズ4096の全結合層の出力にドロップアウトを使用  

【全結合層への変換方法】  

+ Flatten: データの形状をベクトルに変換する(1列に並べ替える)
    + 上図の場合は43,264個のデータになる
+ Global Max Pooling: 各レイヤーで一番大きいものを選ぶ
    + 上図の場合は256個のデータになる
+ Global Avg Pooling: 各レイヤーの平均を使う
    + 上図の場合は256個のデータになる

Global Max PoolingやGlobal Avg Poolingの方が精度が高い  

---

## 実装演習

### 1_1_forward_propagation.ipynb

#### 順伝播(単層・単ユニット)

+ 配列と数値の初期化方法を変えてみる  
  →中間層出力が変化する  
    + 例1:  
    `W = np.array([[0.1], [0.2]])`, `b = 0.5`の場合  
    ![fixed_params]({{site.baseurl}}/images/20211104.png)  
    + 例2:  
    `W = np.zeros(2)`, `b = np.random.rand()`の場合  
    ![zeros_rand]({{site.baseurl}}/images/20211104_1.png)  
    + 例3:  
    `W = np.random.randint(5, size=(2))`, `b = np.random.rand * 10 -5`の場合  
    ![rand_rand]({{site.baseurl}}/images/20211104_2.png)  

#### 順伝播(単層・複数ユニット)

+ 配列の初期化方法を変えてみる  
    + 例1:  
    `W = np.ones([4, 3])`の場合  
![ones]({{site.baseurl}}/images/20211104_3.png)  
    + 例2:  
    `W = np.random.rand(4, 3)`の場合  
    ![rand]({{site.baseurl}}/images/20211104_4.png)  

#### 順伝播(3層・複数ユニット)

+ デフォルトの実行結果:  
    ![default]({{site.baseurl}}/images/20211104_5.png)  
+ 各パラメータのshapeを表示してみる  
    + コード:  
    ![code]({{site.baseurl}}/images/20211104_6.png)  
    + 結果:  
    ![result]({{site.baseurl}}/images/20211104_7.png)  
+ ネットワークの初期値をランダム生成してみる
    + コード:  
    ![code]({{site.baseurl}}/images/20211104_8.png)  
    + 結果:  
    ![result]({{site.baseurl}}/images/20211104_9.png)  

#### 多クラス分類 (2-3-4ネットワーク)

+ デフォルトの実行結果:  
    ![default]({{site.baseurl}}/images/20211104_10.png)  
+ ノードの構成を変えてみる `2-3-4` → `3-5-4`
    + コード:  
        + ネットワークの初期化  
        ![code1]({{site.baseurl}}/images/20211104_11.png)  
        + 入力値  
    ![code2]({{site.baseurl}}/images/20211104_12.png)  
    + 結果:  
    ![result]({{site.baseurl}}/images/20211104_13.png)  

### 1_2_back_propagation.ipynb

逆誤差伝播を行う  
実行結果：  
![result_bp]({{site.baseurl}}/images/20211106.png)  

### 1_3_stochastic_gradient_descent.ipynb

+ デフォルトの実行結果：  
    ![default]({{site.baseurl}}/images/20211106_1.png)  
+ 確率的勾配降下法
    + 以下を試す  
        ![try]({{site.baseurl}}/images/20211106_2.png)  
        + 結果：わずかにデータのばらつきが大きくなった  
            ![result]({{site.baseurl}}/images/20211106_3.png)  
    + 上記をそのままにして、以下も試す  
        ![try2]({{site.baseurl}}/images/20211106_4.png)  
        + 結果：データのばらつきがかなり大きくなった  
            ![result2]({{site.baseurl}}/images/20211106_5.png)  

### 1_4_1_mnist_sample.ipynb

+ デフォルトの実行結果：  
    ![default]({{site.baseurl}}/images/20211106_6.png)  
    ![default_graph]({{site.baseurl}}/images/20211106_7.png)  
    + Xavierの初期値を試す  
        ![Xavier]({{site.baseurl}}/images/20211106_8.png)  
        + 結果：training setの正答率のばらつきが大きくなり、test setの正答率は上がった  
            ![result]({{site.baseurl}}/images/20211106_9.png)  
            ![graph]({{site.baseurl}}/images/20211106_10.png)  
    + Heの初期値を試す  
        ![He]({{site.baseurl}}/images/20211106_11.png)  
        + 結果：Xavierとほぼ同様の変化があった  
            ![result]({{site.baseurl}}/images/20211106_12.png)  
            ![graph]({{site.baseurl}}/images/20211106_13.png)  

### 2_1_network_modified.ipynb

1_4_1_mnist_sample.ipynbの改良版  
実行結果：  
![result]({{site.baseurl}}/images/20211107.png)
![graph]({{site.baseurl}}/images/20211107_1.png)  

### 2_2_2_vanishing_gradient_modified.ipynb

#### sigmoid - gauss

実行結果：  
![result]({{site.baseurl}}/images/20211107_2.png)  
![graph]({{site.baseurl}}/images/20211107_3.png)  

#### ReLU - gauss

実行結果：  
![result]({{site.baseurl}}/images/20211107_4.png)  
![graph]({{site.baseurl}}/images/20211107_5.png)  

#### sigmoid - Xavier

実行結果：  
![result]({{site.baseurl}}/images/20211107_6.png)  
![graph]({{site.baseurl}}/images/20211107_7.png)  

#### ReLU - He

実行結果：  
![result]({{site.baseurl}}/images/20211107_8.png)  
![graph]({{site.baseurl}}/images/20211107_9.png)  

+ [try] `hidden_size_list`の数字を変更してみる  
    + `[40, 20]`から`[1000, 500]`に変更したところ、全ての方法で精度が上がり、  
    試行回数が比較的少ないうちから精度が上がるようになった
        + `sigmoid - gauss`の正答率：
            + トレーニング：0.17 → 0.68
            + テスト：0.1135 → 0.6654  
            ![before]({{site.baseurl}}/images/20211107_3.png)  
            ![after]({{site.baseurl}}/images/20211108.png)  
        + `ReLU - gauss`の正答率：
            + トレーニング：0.95 → 0.99
            + テスト：0.9145 → 0.9584  
            ![before]({{site.baseurl}}/images/20211107_5.png)  
            ![after]({{site.baseurl}}/images/20211108_1.png)  
        + `sigmoid - Xavier`の正答率：
            + トレーニング：0.83 → 0.87
            + テスト：0.8744 → 0.8998  
            ![before]({{site.baseurl}}/images/20211107_7.png)  
            ![after]({{site.baseurl}}/images/20211108_2.png)  
        + `ReLU - He`の正答率：
            + トレーニング：0.95 → 1.0
            + テスト：0.9548 → 0.9701  
            ![before]({{site.baseurl}}/images/20211107_9.png)  
            ![after]({{site.baseurl}}/images/20211108_3.png)  
    + `[40, 20]`から`[10, 5]`に変更したところ、ほぼ全ての方法で精度が下がり、  
    試行回数が比較的多くならないと精度が上がらなくなった
        + `sigmoid - gauss`の正答率：
            + トレーニング：0.17 → 0.11
            + テスト：0.1135 → 0.1135  
            ![before]({{site.baseurl}}/images/20211107_3.png)  
            ![after]({{site.baseurl}}/images/20211108_4.png)  
        + `ReLU - gauss`の正答率：
            + トレーニング：0.95 → 0.74
            + テスト：0.9145 → 0.7329  
            ![before]({{site.baseurl}}/images/20211107_5.png)  
            ![after]({{site.baseurl}}/images/20211108_5.png)  
        + `sigmoid - Xavier`の正答率：
            + トレーニング：0.83 → 0.79
            + テスト：0.8744 → 0.7587  
            ![before]({{site.baseurl}}/images/20211107_7.png)  
            ![after]({{site.baseurl}}/images/20211108_6.png)  
        + `ReLU - He`の正答率：
            + トレーニング：0.95 → 0.95
            + テスト：0.9548 → 0.892  
            ![before]({{site.baseurl}}/images/20211107_9.png)  
            ![after]({{site.baseurl}}/images/20211108_7.png)  
+ [try]`sigmoid - He`と`ReLU - Xavier`についても試してみる
    + `sigmoid - He`  
        + `network`の初期化処理を、以下のようにした  
            ```
            hidden_size_list = [40, 20]
            network = MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, activation='sigmoid', weight_init_std='He')
            ```
        + 結果：  
            ![result]({{site.baseurl}}/images/20211108_8.png)  
            ![graph]({{site.baseurl}}/images/20211108_9.png)  
    + `ReLU - Xavier`
        + `network`の初期化処理を、以下のようにした  
            ```
            hidden_size_list = [40, 20]
            network = MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, activation='relu', weight_init_std='Xavier')
            ```
        + 結果：  
            ![result]({{site.baseurl}}/images/20211108_10.png)  
            ![graph]({{site.baseurl}}/images/20211108_11.png)  

### 2_3_batch_normalization.ipynb

+ デフォルトの実行結果：  
    ![default]({{site.baseurl}}/images/20211108_12.png)  
    ![graph]({{site.baseurl}}/images/20211108_13.png)  
+ [try]活性化関数や重みの初期値を変えてみる
    + `network`の初期化処理を、以下のようにした  
        ```
        network = MultiLayerNet(input_size=784, hidden_size_list=[40, 20], output_size=10,
                        activation='relu', weight_init_std='he', use_batchnorm=use_batchnorm)
        ```
    + 結果：正答率は上昇した  
        + トレーニング：0.8 → 0.95
        + テスト：0.8173 → 0.8981  
            ![result]({{site.baseurl}}/images/20211108_14.png)  
            ![graph]({{site.baseurl}}/images/20211108_15.png)  

### 2_4_optimizer.ipynb

+ デフォルトの実行結果：
    + SGD  
        ![SGD]({{site.baseurl}}/images/20211108_16.png)  
        ![graph]({{site.baseurl}}/images/20211108_17.png)  
    + Momentum  
        ![Momentum]({{site.baseurl}}/images/20211108_18.png)  
        ![graph]({{site.baseurl}}/images/20211108_19.png)  
    + MomentumをもとにAdaGradを作ってみよう  
        ![AdaGrad]({{site.baseurl}}/images/20211108_20.png)  
        ![graph]({{site.baseurl}}/images/20211108_21.png)  
    + RMSprop  
        ![RMSProp]({{site.baseurl}}/images/20211108_22.png)  
        ![graph]({{site.baseurl}}/images/20211108_23.png)  
    + Adam  
        ![Adam]({{site.baseurl}}/images/20211108_24.png)  
        ![graph]({{site.baseurl}}/images/20211108_25.png)  
+ [try]学習率を変えてみる
    + 0.01 → 0.1にしてみた結果：  
    MomentumとAdaGradは正答率向上、RMSpropとAdamは正答率低下。SGDは変化なし  
        + SGD   
            + トレーニング： 0.07 → 0.17
            + テスト： 0.1135 → 0.1135  
                ![before]({{site.baseurl}}/images/20211108_17.png)  
                ![after]({{site.baseurl}}/images/20211108_26.png)  
        + Momentum   
            + トレーニング： 0.1 → 0.6
            + テスト： 0.1135 → 0.573  
                ![before]({{site.baseurl}}/images/20211108_19.png)  
                ![after]({{site.baseurl}}/images/20211108_27.png)  
        + MomentumをもとにAdaGradを作ってみよう   
            + トレーニング： 0.12 → 0.76
            + テスト： 0.1135 → 0.7306  
                ![before]({{site.baseurl}}/images/20211108_21.png)  
                ![after]({{site.baseurl}}/images/20211108_27.png)  
        + RMSprop   
            + トレーニング： 0.99 → 0.21
            + テスト： 0.9421 → 0.1028  
                ![before]({{site.baseurl}}/images/20211108_23.png)  
                ![after]({{site.baseurl}}/images/20211108_28.png)  
        + Adam   
            + トレーニング： 0.95 → 0.09
            + テスト： 0.9456 → 0.1028  
                ![before]({{site.baseurl}}/images/20211108_25.png)  
                ![after]({{site.baseurl}}/images/20211108_29.png)  
+ [try]活性化関数と重みの初期か方法を変えてみる
    + 活性化関数はReLU, 初期化方法はXavierにしてみる  
        + コード
            ```
            network = MultiLayerNet(input_size=784, hidden_size_list=[40, 20], output_size=10, activation='relu', weight_init_std='xavier',
                        use_batchnorm=use_batchnorm)
            ```  
        + 結果：SGD, Momentum, AdaGradで正答率が大きく向上、他はあまり変化なし  
            + SGD   
                + トレーニング： 0.07 → 0.87
                + テスト： 0.1135 → 0.8801  
                    ![before]({{site.baseurl}}/images/20211108_17.png)  
                    ![after]({{site.baseurl}}/images/20211108_30.png)  
            + Momentum   
                + トレーニング： 0.1 → 0.93
                + テスト： 0.1135 → 0.9325  
                    ![before]({{site.baseurl}}/images/20211108_19.png)  
                    ![after]({{site.baseurl}}/images/20211108_31.png)  
            + MomentumをもとにAdaGradを作ってみよう   
                + トレーニング： 0.12 → 0.89
                + テスト： 0.1135 → 0.9296  
                    ![before]({{site.baseurl}}/images/20211108_21.png)  
                    ![after]({{site.baseurl}}/images/20211108_32.png)  
            + RMSprop   
                + トレーニング： 0.99 → 0.96
                + テスト： 0.9421 → 0.9479  
                    ![before]({{site.baseurl}}/images/20211108_23.png)  
                    ![after]({{site.baseurl}}/images/20211108_33.png)  
            + Adam   
                + トレーニング： 0.95 → 0.95
                + テスト： 0.9456 → 0.958  
                    ![before]({{site.baseurl}}/images/20211108_25.png)  
                    ![after]({{site.baseurl}}/images/20211108_34.png)  
+ [try]バッチ正規化をしてみる  
    + 結果：  
    SGD, Momentum, AdaGradは正答率が大きく向上、RMSpropとAdamはわずかに低下  
    ただ、RMSpropとAdamでも比較的少ない試行回数で正答率が向上するようになった  
        + SGD   
            + トレーニング： 0.07 → 0.71
            + テスト： 0.1135 → 0.6722  
                ![before]({{site.baseurl}}/images/20211108_17.png)  
                ![after]({{site.baseurl}}/images/20211108_35.png)  
        + Momentum   
            + トレーニング： 0.1 → 0.87
            + テスト： 0.1135 → 0.8984  
                ![before]({{site.baseurl}}/images/20211108_19.png)  
                ![after]({{site.baseurl}}/images/20211108_36.png)  
        + MomentumをもとにAdaGradを作ってみよう   
            + トレーニング： 0.12 → 0.93
            + テスト： 0.1135 → 0.8882  
                ![before]({{site.baseurl}}/images/20211108_21.png)  
                ![after]({{site.baseurl}}/images/20211108_37.png)  
        + RMSprop   
            + トレーニング： 0.99 → 0.98
            + テスト： 0.9421 → 0.9265  
                ![before]({{site.baseurl}}/images/20211108_23.png)  
                ![after]({{site.baseurl}}/images/20211108_38.png)  
        + Adam   
            + トレーニング： 0.95 → 0.95
            + テスト： 0.9456 → 0.9228  
                ![before]({{site.baseurl}}/images/20211108_25.png)  
                ![after]({{site.baseurl}}/images/20211108_39.png)  
        
### 2_5_overfitting.ipynb

#### overfitting

実行結果  
![result]({{site.baseurl}}/images/20211114.png)  
![graph]({{site.baseurl}}/images/20211114_1.png)  

#### weight decay - L2

+ 実行結果  
    ![result]({{site.baseurl}}/images/20211114_2.png)  
    ![graph]({{site.baseurl}}/images/20211114_3.png)  
+ [try] `weight_decay_lambda`の値を変更して正則化の強さを確認する
    + `weight_decay_lambda`が`0.01`の場合  
        →効果が全くない  
        ![result]({{site.baseurl}}/images/20211114_4.png)  
        ![graph]({{site.baseurl}}/images/20211114_5.png)  
    + `weight_decay_lambda`が`0.15`の場合  
        ![result]({{site.baseurl}}/images/20211114_6.png)  
        ![graph]({{site.baseurl}}/images/20211114_7.png)  
    + `weight_decay_lambda`が`0.5`の場合  
        →途中から精度が落ち、そのまま横ばいになってしまった  
        ![result]({{site.baseurl}}/images/20211114_8.png)  
        ![graph]({{site.baseurl}}/images/20211114_9.png)  

#### weight decay - L1

+ 実行結果  
    ![result]({{site.baseurl}}/images/20211114_10.png)  
    ![graph]({{site.baseurl}}/images/20211114_11.png)  
+ [try] `weight_decay_lambda`の値を変更して正則化の強さを確認する
    + `weight_decay_lambda`が`0.005`の場合  
        →正答率は早く上がったが、過学習になってしまった  
        ![result]({{site.baseurl}}/images/20211114_12.png)  
        ![graph]({{site.baseurl}}/images/20211114_13.png)  
    + `weight_decay_lambda`が`0.1`の場合  
        ![result]({{site.baseurl}}/images/20211114_14.png)  
        ![graph]({{site.baseurl}}/images/20211114_15.png)  
    + `weight_decay_lambda`が`0.15`の場合  
        →途中から精度が落ち、そのまま横ばいになってしまった  
        ![result]({{site.baseurl}}/images/20211114_16.png)  
        ![graph]({{site.baseurl}}/images/20211114_17.png)  

#### Dropout

+ 実行結果  
    ![result]({{site.baseurl}}/images/20211114_18.png)  
    ![graph]({{site.baseurl}}/images/20211114_19.png)  
+ [try] `dropout_ratio`の値を変更してみる   
    + `dropout_ratio`が`0.1`の場合  
        →正答率は上がったものの、過学習している  
        ![result]({{site.baseurl}}/images/20211114_20.png)  
        ![graph]({{site.baseurl}}/images/20211114_21.png)  
    + `dropout_ratio`が`0.2`の場合  
        →正答率は十分には上がらなかったが、トレーニングデータとテストデータの正答率の差は縮まった  
        ![result]({{site.baseurl}}/images/20211114_22.png)  
        ![graph]({{site.baseurl}}/images/20211114_23.png)  
+ [try] `optimizer`と`dropout_ratio`の値を変更してみる  
    + `optimizer`: `Momentum`, `dropout_ratio`: `0.1`の場合  
        ![result]({{site.baseurl}}/images/20211114_24.png)  
        ![graph]({{site.baseurl}}/images/20211114_25.png)  
    + `optimizer`: `AdaGrad`, `dropout_ratio`: `0.1`の場合  
        ![result]({{site.baseurl}}/images/20211114_26.png)  
        ![graph]({{site.baseurl}}/images/20211114_27.png)  
    + `optimizer`: `Adam`, `dropout_ratio`: `0.1`の場合  
        ![result]({{site.baseurl}}/images/20211114_28.png)  
        ![graph]({{site.baseurl}}/images/20211114_29.png)  
    + `optimizer`: `Momentum`, `dropout_ratio`: `0.35`の場合  
        (`SGD`の場合と違い、`dropout_ratio`が`0.2`でも過学習した。`0.35`くらいがちょうど良さそう)  
        ![result]({{site.baseurl}}/images/20211114_30.png)  
        ![graph]({{site.baseurl}}/images/20211114_31.png)  
    + `optimizer`: `AdaGrad`, `dropout_ratio`: `0.425`の場合  
        ![result]({{site.baseurl}}/images/20211114_32.png)  
        ![graph]({{site.baseurl}}/images/20211114_33.png)  
    + `optimizer`: `Adam`, `dropout_ratio`: `0.437`の場合  
        ![result]({{site.baseurl}}/images/20211114_34.png)  
        ![graph]({{site.baseurl}}/images/20211114_35.png)  

#### Dropout + L1

実行結果  
![result]({{site.baseurl}}/images/20211114_36.png)  
![graph]({{site.baseurl}}/images/20211114_37.png)  