---
layout: post
title: "GitHub Pagesで数式を表示する"
---

<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

本ブログで数式を表示したくなったので、TeXを使った方法を調べてみた。  
なお、本ブログは本ブログではJekyllを使い、Markdownで記述している。(ブログ作成時の記事は[こちら](https://marshmallow444.github.io/tech_blog/2020/09/28/start-a-blog-with-github-pages.html))

## 方法1

以下のスクリプトをMarkdownファイル内に記述しておく。(MathJaxが読み込まれる)

```
<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>
```

#### 記述例

```
$\sqrt{2}$
```
→ $\sqrt{2}$

```
\\[
D = P^{-1} A P
\\]
```
→ \\[
D = P^{-1} A P
\\]


```
\begin{align}
    \frac{d}{dx} \int_a^x f(t) \\: dt = f(x) 
\end{align}
```
→
\begin{align}
    \frac{d}{dx} \int_a^x f(t) \\: dt = f(x) 
\end{align}

#### 注意事項

+ コマンドの頭以外で`\`を使用する際は、2つ記述する必要がある？
+ 改行時は`\\\\  `(バックスラッシュ4つ + 半角スペース2つ)を入力しないと改行されない

#### 参考：
+ [GitHub Pagesで数式を書く方法と主なトラブルについて](https://qiita.com/BurnEtz/items/e79999264125eb128ae7)

<br>

## 方法2

MathJax の読み込みとオプション設定をまとめてhtmlファイルとして保存しておき、使用するテーマでそれを読み込むようにする (動作未確認)

#### 参考：
+ [Github Pages で数式を ～ MathJax v3 設定のポイント](https://qiita.com/memakura/items/e4d2de379f98ad7be498)
+ [GitHub Pagesでちょっと遊んでみる(3): GitHub PagesでMathJax!!](https://pandanote.info/?p=3715)

<br>

## 方法3

[tex image link generator](https://tex-image-link-generator.herokuapp.com/)を使う  
使い方はこちら→[githubやnoteでもTeXの数式を書くぜ](https://aotamasaki.hatenablog.com/entry/2020/08/09/github%E3%82%84note%E3%81%A7%E3%82%82TeX%E3%81%AE%E6%95%B0%E5%BC%8F%E3%82%92%E6%9B%B8%E3%81%8F%E3%81%9C)

#### 具体例

入力するテキスト：  

```
\begin{align*}
\frac{d}{dx} \int_a^x f(t) \: dt = f(x)
\end{align*}
```

生成されるテキスト(Markdown用)：  

```
![\begin{align*}
\frac{d}{dx} \int_a^x f(t) \: dt = f(x)
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cfrac%7Bd%7D%7Bdx%7D+%5Cint_a%5Ex+f%28t%29+%5C%3A+dt+%3D+f%28x%29%0A%5Cend%7Balign%2A%7D%0A)
```

上記テキストをMarkdown内に張り付けると、以下のような表示になる。

![\begin{align*}
\frac{d}{dx} \int_a^x f(t) \: dt = f(x)
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cfrac%7Bd%7D%7Bdx%7D+%5Cint_a%5Ex+f%28t%29+%5C%3A+dt+%3D+f%28x%29%0A%5Cend%7Balign%2A%7D%0A)

<br>

## 備考

TeXを初めて使うので、よく分かっていないことがありそう。今後何か気づいたことがあれば追記していく。