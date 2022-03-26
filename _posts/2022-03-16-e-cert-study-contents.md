---
layout: post_toc
title: "【E資格】2022#1 参考になったコンテンツ"
tags: E資格 機械学習
---

JDLA [E資格](https://www.jdla.org/certificate/engineer/) 2022#1を受験したので、参考になった書籍・動画・サイトをまとめておく。  
今回のE資格受験についてまとめた記事は[こちら](./e-cert.html)。  

<br>

## 本ページの見方

ここでは以下の形式でまとめてある。

```markdown
・コンテンツ名
    ・(ジャンル)
        ・おすすめポイント
```

特におすすめのものには<u>下線</u>を引いてある。

<br>

## 書籍

- [<u>徹底攻略ディープラーニングE資格エンジニア問題集 第2版</u>](https://book.impress.co.jp/books/1120101184)
    - (出題範囲全般)
        - 通称「黒本」。E資格の問題集はこれしかない
        - 理解の足りない箇所を洗い出すために必須
- <u>ゼロから作るDeep Learning</u> シリーズ ([<u>1</u>](https://www.oreilly.co.jp/books/9784873117584/), [2](https://www.oreilly.co.jp/books/9784873118369/), [3](https://www.oreilly.co.jp/books/9784873119069/))
    - <u>1</u> (深層学習)
        - 最低限目を通しておいた方が良い。できれば写経も。おすすめ
        - 説明が分かりやすい
        - この本のサンプルコードとそっくりなものが、試験に出ることもある
    - 2 (自然言語処理)
        - 一通り読んでおくとよさそう (私は時間が足りず諦めてしまった)
        - 黒本の解説で理解できなかったところを、この本で補完。辞書的な使い方をした
    - 3 (深層学習：ライブラリを自作する)
        - こちらも私は辞書的な使い方をした
        - 良い本だが、1と2があれば十分かも？
    - 今後発売される[4](https://www.amazon.co.jp/dp/4873119758?tag=note0e2a-22&linkCode=ogi&th=1&psc=1)(強化学習)も分かりやすそう
- [<u>ITエンジニアのための強化学習理論入門</u>](https://www.amazon.co.jp/dp/B08D91JR4J/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)
    - (強化学習)
        - サンプルコードを使って解説されており、理解しやすかった
            - 1行ごとに説明がついているので、非エンジニアの方にも良いかも？
- [<u>史上最強図解　これならわかる！ベイズ統計学</u>](https://www.amazon.co.jp/dp/B07SDQ84MY)
    - (確率、ベイズの定理)
        - 数学が得意でない方におすすめ。イメージがつかみやすい
        - 確率やベイズの定理関連の問題がかなり解きやすくなる

<br>

## 動画

- [<u>AIcia Solid Project</u>](https://www.youtube.com/channel/UC2lJYodMaAfFeFQrGUwhlaQ)
    - (深層学習、画像認識、自然言語処理など)
        - 人間が画像や言語を認識する仕組みを、AIではどのように再現しようとしているか？という目線で説明している
        - 初心者にも分かるように、かみくだいて解説されている
        - 非常に分かりやすいので必見！
            - 特に以下のプレイリストがおすすめ
                - [Deep Learningの世界](https://youtube.com/playlist?list=PLhDAH9aTfnxKXf__soUoAEOrbLAOnVHCP)
                - [自然言語処理シリーズ](https://youtube.com/playlist?list=PLhDAH9aTfnxL4XdCRjUCC0_flR00A6tJR)
                - [CNN紹介動画](https://youtube.com/playlist?list=PLhDAH9aTfnxIGIvLciL1CtE59VGrEx4ER)
        - 今後強化学習についての動画も作成なさるとのこと。そちらも参考になりそう
- [データサイエンス研究所](https://www.youtube.com/channel/UCFDyXEywtNhdtwqC3GAkYuA/featured)
    - (機械学習など)
        - EMアルゴリズムやMCMC法の説明などがわかりやすかった
- [予備校のノリで学ぶ「大学の数学・物理」](https://www.youtube.com/c/yobinori)
    - (数学)
        - ベイズの定理などの説明が分かりやすかった

<br>

## サイト

- [【線形代数】特異値分解とは?例題付きで分かりやすく解説!!](https://nisshingeppo.com/ai/singular-value-decomposition/)
    - (特異値分解)
        - 例題があり、わかりやすい
- [特異値分解問題を解く](https://kleinblog.net/singular-value-decomposition.html)
    - (特異値分解)
        - こちらも例題があり、分かりやすい
- [情報量とエントロピー - 導出と性質](https://whyitsso.net/math/statistics/information_entropy.html)
    - (エントロピー)
        - ベン図を使った説明が分かりやすい
- [【決定版】スーパーわかりやすい最適化アルゴリズム -損失関数からAdamとニュートン法-](https://qiita.com/omiita/items/1735c1d048fe5f611f80)
    - (最適化アルゴリズム)
        - 各手法の生まれたきっかけと共に説明されており、分かりやすい
- [NegativeMindException](https://blog.negativemind.com/category/computational-photography/)
    - (GAN系)
        - 一連の手法の解説が分かりやすい
- [ニューラルネットワークが任意の関数を表現できることの視覚的証明](https://nnadl-ja.github.io/nnadl_site_ja/chap4.html)
    - (万能近似定理(普遍性定理))
        - NNの重みを実際に変えながら、出力の変化を見ることができる