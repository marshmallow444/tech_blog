---
title: Swiftのソースにドキュメントコメントを入れる話
tags:  Swift Xcode
categories:
author: marshmallow444
status: public
created_at: 2019-01-26 23:04:50 +0900
updated_at: 2019-01-26 23:31:31 +0900
published_at: 2019-01-26 23:04:50 +0900
---
### はじめに  
Objective-Cでコードを書く時にはdoxygen形式でコメントを入れていた。  
そうするとクラスやメソッドの概要などがXcodeのQuick Helpへ反映され、便利だなと思っていたのだが。  
Swiftで同様にコメントを入れたところ、それがQuick Helpに反映されない・・・  
ので、そのことについて調べてみた。  

<br>
### 結論  
Swiftではdoxygen形式でコメントを書いても、Quick Helpへは一部反映されない。  
`/`3つで始まるコメントを使うと反映される。

<br>
### どういうことなのか   
doxygen形式でも、クラスやコメントのSummaryは正しく認識されるし、ツールを使ってドキュメントを吐き出すことも可能らしい。  
しかし、  

> 一行目にメソッドの説明と:param:、:return:を使えばとりあえず問題なし  

との情報を見て試したが、`:param:`がDiscussionの欄に反映され、Parametersの欄には`No description`と表示されてしまう。どうやらObjective-Cのソースのようには認識してくれないようだ。  

`Option+Command+/`を押すと`/`3つから始まるコメントが自動生成され、この内容はQuick Helpに反映される。  
```
/// <#Description#>
///
/// - Parameter param1: <#param1 description#>
/// - Returns: <#return value description#>
```
ショートカットですぐ定型のコメントが出るので、doxygen形式よりも便利だなと思った。  
あと、Markdown形式で書けるのも良い。  
今後Swiftではこの形式のコメントを使っていこうと思う。

<br>
---

参考にしたページ：  
+ [Swiftのドキュメントコメント](https://qiita.com/Todate/items/819618dbb56e61d97453)
+ [クールなSwift用ドキュメントジェネレーターjazzyを使った](http://blog.euphonictech.com/entry/2015/03/28/215836)
+ [Swiftでヘッダドキュメント](http://d.hatena.ne.jp/tsntsumi/20140809/HeaderDocumentationInSwift)
