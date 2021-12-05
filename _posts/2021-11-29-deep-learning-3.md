---
layout: post_toc
title: "【ラビット・チャレンジ】深層学習 後編 Day3"
tags: ラビット・チャレンジ E資格 機械学習
---

<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

[ラビット・チャレンジ](https://ai999.careers/rabbit/)の受講レポート。  

---  

# 前回の復習

**確認テスト**  

サイズ5×5の入力画像を、サイズ3×3のフィルタで畳み込んだときの出力画像のサイズは？  
ストライドは2, パディングは1とする  

【解答】  

3×3  

# 再帰型ニューラルネットワークについて

## 再帰型ニューラルネットワークの概念

### RNN全体像

#### RNNとは

<u>時系列データ</u>に対応可能なニューラルネットワーク  


#### 時系列データ

時間的順序を追って一定間隔ごとに観察され、  
しかも相互に統計的依存関係が認められるようなデータの系列  

例：  

+ 音声データ
+ テキストデータ

#### RNNについて

全体像  

[![RNN](http://www.net.c.dendai.ac.jp/~ogata/figure/004.jpg)](http://www.net.c.dendai.ac.jp/~ogata/figure/004.jpg)  
(画像：[http://www.net.c.dendai.ac.jp/~ogata/](http://www.net.c.dendai.ac.jp/~ogata/))

「展開」の矢印の左右の図の意味は同じ  

<br>

[![weight](https://4.bp.blogspot.com/-4sNWGgBFLkE/WKcLzxWRa9I/AAAAAAAAiGQ/sqV16vvhGH09_QyZK5sIPJ8UcEyhekp5ACLcB/s1600/OldR%251C%251CNN.png)](https://4.bp.blogspot.com/-4sNWGgBFLkE/WKcLzxWRa9I/AAAAAAAAiGQ/sqV16vvhGH09_QyZK5sIPJ8UcEyhekp5ACLcB/s1600/OldR%251C%251CNN.png)  
(画像：[http://maruyama097.blogspot.com/2017/02/lstmrnn.html](http://maruyama097.blogspot.com/2017/02/lstmrnn.html))

**RNNの数学的記述**  

$$
    u^t = W_{(in)}x^t + W \overbrace{z^{t-1}}^{*1}  + b \\  
    z^t = f(W_{(in)}x^t + Wz^{t-1} + b) \\  
    v^t = W_{(out)} z^t + c \\  
    y^t = g(W_{(out)} z^t + c) \\  
$$

$$
    \begin{split}
        W_{(in)}: &入力層から中間層への重み \\  
        W_{(out)}: &中間層から出力層への重み \\  
        *1: &前の時間の中間層の出力
    \end{split}
$$
 

活性化関数を通る前：$u / v$  
活性化関数を通った後：$z / y$  
入力側の矢印：$u, z$  
出力側の矢印：$v, y$  

```Python
u[:, t+1] = np.dot(X, W_in) + np.dot(z[:, t].reshape(1, -1), W)
z[:, t+1] = functions.sigmoid(u[:, t+1])
np.dot(z[:, t+1].reshape(1, -1), W_out) # vにあたる部分
y[:, t] = functions.sigmoid(np.dot(z[:, t+1].reshape(1, -1), W_out))
```

**確認テスト**  

RNNのネットワークには大きく分けて3つの重みがある。  
1つは入力から現在の中間層を定義する際にかけられる重み、  
1つは中間層から出力を定義する際にかけられる重みである。  
残り1つの重みについて説明せよ。  

【解答】  
中間層から中間層への重み  

**RNNの特徴**  
時系列モデルを扱うには、初期の状態と過去の時間t-1の状態を保持し、  
そこから次の時間でのtを再帰的に求める再帰構造が必要  

**演習チャレンジ**  

以下は再帰型ニューラルネットワークにおいて構文木を入力とし再帰的に文全体の表現ベクトルを得るプログラム  
ただし、ニューラルネットワークの重みパラメータはグローバル変数として定義してあるものとし、  
_activation関数は何らかの活性化関数  
木構造は再帰的な辞書で定義してあり、rootが最も外側の辞書であると仮定  
(く)に当てはまるのは？  

```Python
def traverse(node):
    if not isinstance(node, dict):
        v = node
    else:
        left = traverse(node['left'])
        right = traverse(node['right'])
        v = _activation(【(く)】)
    return v
```

【解答】  
```
W.dot(np.concatenate([left, right]))
```  
→leftとrightの特徴量が失われないようにする  

【補足】  
+ `concatenate`: 配列を連結する処理  
    + 連結したものに重みをかけることで、一旦大きくなった配列のサイズを元に戻す  
+ 構文木
    + まず隣り合った単語同士の特徴量を抽出し、次にその隣り合った特徴量同士をあわせた特徴量を抽出し、それを繰り返して最終的に文全体を表す一つの特徴量を抽出する  
    [![構文木](https://upload.wikimedia.org/wikipedia/commons/0/04/JapaneseSyntaxTreeSample1.png)](https://upload.wikimedia.org/wikipedia/commons/0/04/JapaneseSyntaxTreeSample1.png)  
    (画像：[https://ja.wikipedia.org/wiki/%E6%A7%8B%E6%96%87%E6%9C%A8](https://ja.wikipedia.org/wiki/%E6%A7%8B%E6%96%87%E6%9C%A8))

### BPTT

#### BPTTとは

Back Propagation Through Time  

#### 逆誤差伝播法の復習

計算結果(=誤差)から微分を逆算  
→不要な再帰的計算を避けて微分を算出

**確認テスト**  

連鎖律の原理を使い、$\frac{dz}{dx}$を求めよ  

$$
    z = t^2 \\ 
    t = x + y
$$

【解答】  

$$
    \frac{dz}{dx} = \frac{dz}{dt} \cdot \frac{dt}{dx} \\  
    \space \\  
    \Rightarrow
    \left\{
        \begin{array}{ll}
            \frac{dz}{dt} = 2t  \\  
            \frac{dt}{dx} = 1  \\  
        \end{array}
    \right. \\  
    \space \\  
    \begin{split}
        \Rightarrow \frac{dz}{dx} &= 2t \cdot 1 \\  
        &= 2(x + y)
    \end{split}
$$

#### BPTTの数学的記述

(本講座内で一番難しいところ！)  

$$
    \frac{\partial E}{\partial W_{(in)}} = \frac{\partial E}{\partial u^t}
    \left[
        \frac{\partial u^t}{\partial W_{(in)}}
    \right]^T
    = \delta^t[x^t]^T \\  
    \frac{\partial E}{\partial W_{(out)}} = \frac{\partial E}{\partial v^t}
    \left[
        \frac{\partial v^t}{\partial W_{(out)}}
    \right]^T
    = \delta^{out, t}[z^t]^T \\  
    \frac{\partial E}{\partial W} = \frac{\partial E}{\partial u^t}
    \left[
        \frac{\partial u^t}{\partial W}
    \right]^T
    = \delta^t[z^{t-1}]^T
$$

$$
    \frac{\partial E}{\partial b} = \frac{\partial E}{\partial u^t} \frac{\partial u^t}{\partial b} = \delta^t \\  
    \frac{\partial E}{\partial c} = \frac{\partial E}{\partial v^t} \frac{\partial v^t}{\partial c} = \delta^{out, t}
$$

[RNNについて](#rnnについて)を参照しながら理解するとよい  

**重み$W_{(in)}$について (1つ目の式)**

$$
    \frac{\partial E}{\partial W_{(in)}} = \frac{\partial E}{\partial u^t}
    \left[
        \frac{\partial u^t}{\partial W_{(in)}}
    \right]^T
    = \delta^t[x^t]^T
$$

```Python
# コード例
np.dot(X.T, delta[:, t].reshape(1, -1))
```

$$
    u^t = W_{(in)}x^t + W \overbrace{z^{t-1}}^{*1}  + b
$$

より、$W_{(in)}$は$u^t$についての式  
$\delta^t$: $E$を$u^t$まで微分したもの  
$[ \space ]^T$: RNNにおいて、時間的に遡って微分を全部やる  

**重み$W_{(out)}$について (2つ目の式)**

$$
    \frac{\partial E}{\partial W_{(out)}} = \frac{\partial E}{\partial v^t}
    \left[
        \frac{\partial v^t}{\partial W_{(out)}}
    \right]^T
    = \delta^{out, t}[z^t]^T
$$

```Python
# コード例
np.dot(z[:, t+1].reshape(-1, 1), delta_out[:, t].reshape(-1, 1))
```

$$
    v^t = W_{(out)} z^t + c \\  
$$

より、$W_{(out)}$は$v^t$についての式  
$\delta^{out, t}$: $E$を$v^t$まで微分したもの  


**重み$W$について (3つ目の式)**

$$
    \frac{\partial E}{\partial W} = \frac{\partial E}{\partial u^t}
    \left[
        \frac{\partial u^t}{\partial W}
    \right]^T
    = \delta^t[z^{t-1}]^T
$$

```Python
# コード例
np.dot(z[:, t].reshape(-1, 1), delta[:, t].reshape(1, -1))
```

**バイアス$b$について (4つ目の式)**

$$
    \frac{\partial E}{\partial b} = \frac{\partial E}{\partial u^t} \frac{\partial u^t}{\partial b} = \delta^t
    \qquad
    \frac{\partial  u^t}{\partial b} = 1
$$

**バイアス$c$について (5つ目の式)**

$$
    \frac{\partial E}{\partial c} = \frac{\partial E}{\partial v^t} \frac{\partial v^t}{\partial c} = \delta^{out, t}
    \qquad
    \frac{\partial v^t}{\partial c} = 1
$$

**$\delta$について**

$$
    \begin{split}
        \frac{\partial E}{\partial u^t} &= \frac{\partial E}{\partial v^t} \frac{\partial v^t}{\partial u^t} \\  
        &= \frac{\partial E}{\partial v^t} \frac{\partial\{W_{(out)}f(u^t) + c\}}{\partial u^t} \\  
        &= f'(u^t)W^T_{(out)} \delta^{out, t} \\  
        &= \delta^t
    \end{split}
$$

+ $v^t = W_{(out)}z^t + c$, $z^t = f(u^t)$なので、$v^t = W_{(out)}f(u^t) + c$
+ $\frac{\partial E}{\partial v^t} = \delta^{out, t}$

```Python
delta[:, t] = (np.dot(delta[:, t+1].T, W.T) + np.dot(delta_out[:, t].T, W_out.T)) * functions.d_sigmoid(u[:, t])
```

**(ちなみに、、)**  

$\delta^t$と$\delta^{t-1}$の間には関係がある  
$\delta^{t-z}$と$\delta^{t-z-1}$の間には関係がある  

$$
    \begin{split}
        \delta^{t-1} &= \frac{\partial E}{\partial u^{t-1}} = \frac{\partial E}{\partial u^t} \frac{\partial u^t}{\partial u^{t-1}} \\  
        &= \delta^t
        \left\{
            \frac{\partial u^t}{\partial z^{t-1}} \frac{\partial z^{t-1}}{\partial u^{t-1}}
        \right\} \\  
        &= \delta^t \{W f'(u^{t-1})\} \\  
        \delta^{t-z-1} &= \delta^{t-z}\{Wf'(u^{t-z-1})\}
    \end{split}
$$

**確認テスト**

RNNの図において、$y_1$を$x, z_0, z_1, w_{in}, w, w_{out}$を用いて数式で表せ  
※バイアスは任意の文字で定義  
※中間層の出力にシグモイド関数$g(x)$を作用させよ  

【解答】

$$
    z_1 = W_{in} x_1 + W z_0 + b \\  
    y_1 = g(W_{out} z_1 + c)
$$

**パラメータの更新式**  

$$
    W^{t+1}_{(in)} = W^t_{(in)} - \epsilon \frac{\partial E}{\partial W_{(in)}} = W^t_{(in)} - \epsilon \sum_{z=0}^{Tt} \delta^{t-z}[x^{t-z}]^T \\  
    W^{t+1}_{(out)} = W^t_{(out)} - \epsilon \frac{\partial E}{\partial W_{(out)}} = W^t_{(out)} - \epsilon \delta^{out, t}[z^t]^T \\  
    W^{t+1} = W^t - \epsilon \frac{\partial E}{\partial W} = W^t - \epsilon \sum_{z=0}^{Tt} \delta^{t-z}[x^{t-z-1}]^T \\  
    b^{t+1} = b^t - \epsilon \frac{\partial E}{\partial b} = b^t - \epsilon \sum_{z=0}^{Tt} \delta^{t-z} \\  
    c^{t+1} = c^t - \epsilon \frac{\partial E}{\partial c} = c^t - \epsilon \delta^{out, t}
$$

$W_{(out)}$の処理では時間的に遡らない  
中間層の前までは時間的な遡りを考慮する  
$\epsilon$: 学習率  

$$
    W^{t+1}_{(in)} = W^t_{(in)} - \epsilon \frac{\partial E}{\partial W_{(in)}} = W^t_{(in)} - \epsilon \sum_{z=0}^{Tt} \delta^{t-z}[x^{t-z}]^T
$$

```Python
# コード例
W_in -= learning_rate  * W_in_grad
```

$$
    W^{t+1}_{(out)} = W^t_{(out)} - \epsilon \frac{\partial E}{\partial W_{(out)}} = W^t_{(out)} - \epsilon \delta^{out, t}[z^t]^T 
$$

```Python
# コード例
W_out -= learning_rate * W_out_grad
```

$$
    W^{t+1} = W^t - \epsilon \frac{\partial E}{\partial W} = W^t - \epsilon \sum_{z=0}^{Tt} \delta^{t-z}[x^{t-z-1}]^T 
$$

```Python
# コード例
W -= learning_rate * W_grad
```

#### BPTTの全体像

$$
    \begin{split}
        E^t &= loss(y^t, d^t) \\  
        &= loss
        \left(
            g(W_{(out)}z^t + c), d^t
        \right) \\  
        &= loss
        \left(
            g(W_{(out)}f( \underbrace{W_{(in)} x^t + Wz^{t-1} + b}_{*2} ) + c), d^t
        \right)
    \end{split}
$$

*2について  

$$
    W_{(in)} x^t + Wz^{t-1} + b \\  
    W_{(in)} x^t + W f(u^{t-1}) + b \\  
    W_{(in)} x^t + W f(W_{(in)} x^{t-1} + Wz^{t-2} + b) + b
$$

**コード演習問題**  

BPTTを行うプログラム  
活性化関数は恒等関数  
calculate_dout関数は損失関数を出力に関して偏微分した値を返す  
(お)に当てはまるものは？  

```Python
def bptt(xs, ys, W, U, V):
    hiddens. outputs = rnn_net(xs, W, U, V)

    dW = np.zeros_like(W)
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)

    do = _calculate_do(outputs, ys)

    batch_size, n_seq = ys.shape[:2]

    for t in reversed(range(n_seq)):
        dV += np.dot(do[:, t].T, hiddens[:, t]) / batch_size
        delta_t = do[:, t].dot(V)

        for bptt_step in reversed(range(t+1)):
            dW += np.dot(delta_t.T, xs[:, bptt_step]) / batch_size
            dU += np.dot(delta_t.T, hiddens[:, bptt_step-1]) / batch_size
            delta_t = 【(お)】
    return dW, dU, dV
```

【解答】  
delta_t.dot(U)  

【解説】  
RNNでは中間層出力$h_t$が過去の中間層出力$h_{t-1}, \cdots, h_1$に依存する  
RNNにおいて喪失感すを重み$W$や$U$に関して偏微分するときは、これを考慮する必要があり  
$\frac{dh_t}{dh_{t-1}} = U$  
であることに注意すると、過去に遡るたびに$U$がかけられる

## LSTM

+ RNNの課題
    + 時系列を遡るほど、勾配が消失していく
        + **長い時系列の学習が困難**
    + 解決策
        + 構造自体を変えて解決したのが**LSTM**

**確認テスト**  

シグモイド関数を微分したとき、入力値が0の時に最大値をとる  
その値は？  

【解答】  
0.25  

**演習チャレンジ**  

RNNや深いモデルでは勾配の消失または爆発が起こる傾向にある  
勾配爆発を防ぐために勾配のクリッピングを行うという手法がある  
具体的には勾配のノルムがしきい値をこえたら  
勾配のノルムをしきい値に正規化するというもの  
以下は勾配のクリッピングを行う関数  
(さ)に当てはまるのは？  

```Python
def gradient_clipping(grad, threshold):
    norm = np.linalg.norm(grad)
    rate = threshold / norm
    if rate < 1:
        return 【(き)】
    return grad
```

【解答】  
gradient * rate  

勾配 * (しきい値 / 勾配のノルム)と計算される  

### 全体像

[![LSTM](https://agirobots.com/wp/wp-content/uploads/2020/07/Basic-LSTM-cell-1536x617.png)](https://agirobots.com/wp/wp-content/uploads/2020/07/Basic-LSTM-cell-1536x617.png)  
(画像：[https://agirobots.com/lstmgruentrance-noformula/](https://agirobots.com/lstmgruentrance-noformula/))  

基本的にやっていることはRNNと同じ  
点線(出力側→入力側)は時間的なループを表す  
**CEC**が大事  

### CEC

誤差カルーセル(Constant Error Caroucel)  
記憶機能だけをもつ  
勾配消失問題・勾配爆発問題：勾配が1であれば解決できる  

$$
    \delta^{t-z-1} = \delta^{t-z}
    \left\{
        W f'(u^{t-z-1})
    \right\}
    = 1 \\  
    \frac{\partial E}{\partial c^{t-1}}
    = \frac{\partial E}{\partial c^t} \frac{\partial c^t}{\partial c^{t-1}}
    = \frac{\partial E}{\partial c^t} \frac{\partial}{\partial c^{t-1}}\{a^t - c^{t-1}\}
    = \frac{\partial E}{\partial c^t}
$$

 **課題**  
入力データについて、時間依存度に関係なく重みが一律  
→NNの学習特性がない  
→CECの前後に学習機能をもつものを置くことで対処  

入力重み衝突：入力層→隠れ層の重み  
出力重み衝突：隠れ層→出力層の重み  

### 入力ゲートと出力ゲート

+ 役割
    + 入力ゲートと出力ゲートへの入力値の重みを、重み行列$W, U$で可変可能とする
    + →CECの課題を解決
+ 入力ゲート
    + CECに対し、入力データを「こんなふうに」覚えてくださいね、と指示する
        + 「こんなふうに」の部分、CECへの覚えさせ方をNNで学習
    + 今回の入力値と前回の中間層からの出力値を元に、今回の入力データをどう記憶するか決める
        + 今回の入力値に対する重み：$W_i$
        + 前回の出力値に対する重み：$U_i$
        + どれくらい覚えさせるか：$V_i c(t-1)$
+ 出力ゲート
    + CECから取り出したデータを「こんなふうに」利用する、と決める
        + 「こんなふうに」の部分をNNで学習
    + 今回の入力値と前回の中間層からの出力値を元に、今回の記憶したデータをどう利用するか決める
        + 今回の入力値に対する重み：$W_o$
        + 前回の出力値に対する重み：$U_o$
        + どれくらい利用するか：$V_o c(t)o(t)$


### 忘却ゲート

過去の情報が要らなくなった場合、そのタイミングで情報を忘却させる  
これがないと、不要になった情報が削除できない  
→大昔の不要なデータが影響を与える可能性が出てしまう  

$$
    c(t) = i(t) \cdot \underbrace{a(t)}_{i(t)が活性化関数を通った結果} + \underbrace{f(t)}_{forget} \cdot c(t-1)
$$

**確認テスト**  

以下の文章をLSTMに入力し、空欄に当てはまる単語を予測したい  
文中の「とても」という言葉は空欄の予測において  
なくなっても影響を及ぼさないと考えられる  
この場合、どのゲートが作用する？  
<br>
「映画おもしろかったね。ところで、とてもお腹が空いたから何か<u>【空欄】</u>。」  

【解答】  
忘却ゲート  

**演習チャレンジ**  

以下のプログラムはLSTMの順伝播を行うもの  
ただし_sigmoid関数は要素ごとにシグモイド関数を作用させる  
(け)に当てはまるのは？  

```Python
def lstm(x, prev_h, prev_c, W, U, b):
    # セルへの入力やゲートをまとめて計算し、分離
    lstm_in = _activation(x.dot(W.T) + prev_h.dot(U.T) + b)
    a, i, f, o = np.hsplit(lstm_in, 4)

    # 値を変換、セルへの入力:(-1, 1) ゲート:(0, 1)
    a = np.tanh(a)
    input_gate = _sigmoid(i)
    forget_gate = _sigmoid(f)
    output_gate = _sigmoid(o)

    # セルの状態を計算し、中間層の出力を計算
    c = 【(け)】
    h = output_gate * np.tanh(c)
    return c, h
```

【解答】  
input_gate * a + forget_gate * c  

【解説】  
新しいセルの状態 = 計算されたセルへの入力 * 入力ゲート + 1ステップ前のセルの状態 * 忘却ゲート

### 覗き穴結合

CEC自身の値に、重み行列を介して伝播可能にした構造  
→CECの保存されている過去の情報を、  
　任意のタイミングで他のノードへ伝播させたり  
　任意のタイミングで忘却させたりしたい  
→CEC自身の値は、ゲート制御に影響を与えていない  
→CECの状態も判断材料に使ってみる  

(あまり大きな改善はみられなかった)  

## GRU

パラメータを大幅に削減し、LSTMと同等またはそれ以上の精度が望めるようになった構造  
→計算負荷が低い  
(LSTMはパラメータ数が多く、計算負荷が高い)  

[![GRU](https://ichi.pro/assets/images/max/724/1*yBXV9o5q7L_CvY7quJt3WQ.png)](https://ichi.pro/assets/images/max/724/1*yBXV9o5q7L_CvY7quJt3WQ.png)  
(画像：[https://ichi.pro/lstm-oyobi-gru-no-zukai-gaido-suteppubaisuteppu-no-setsumei-75771469479713](https://ichi.pro/lstm-oyobi-gru-no-zukai-gaido-suteppubaisuteppu-no-setsumei-75771469479713))  

隠れ層の状態：  

$$
    h(t) = f
    \left(
        W_r x(t) + U_r \cdot (r(t) \cdot h(t-1)) b_h(t)
    \right)
$$

リセットゲート：  
隠れ層をどのような状態で保持するかを制御  

$$
    r(t) = W_r x(t) + U_r \cdot h(t-1) + b_h(t)
$$

更新ゲート：  
保持している値をどのように使って出力を得るか制御

$$
    z(t) = W_z x(t) + U_z \cdot h(t-1) + b_z(t)
$$

隠れ層からの出力：  

$$
    z(t) \cdot h(t-1) + (1-z(t)) \cdot h(t)
$$

## 双方向RNN

# RNNでの自然言語処理

## Seq2Seq

### 全体像

### Encoder RNN

### Decoder RNN

### HRED

### VHRED

### VAE

#### オートエンコーダー

#### VAE

## Word2Vec

## AttentionMechanism

# 実装演習