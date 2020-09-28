---
title: 【AutoLayout】iOSの文字サイズの設定を変えたら、レイアウトが崩れた話
tags:  iOS AutoLayout
categories:
author: marshmallow444
status: public
created_at: 2018-12-19 01:54:37 +0900
updated_at: 2018-12-21 02:27:32 +0900
published_at: 2018-12-21 02:27:32 +0900
---
## はじめに
Auto Layoutを使い始めて約1年弱なのですが、私がAuto Layoutを使って作ったアプリに、とある不具合を発見しました。その解決方法がネット等で見つけられなかったので、自力で編み出した方法をここにメモしておきます。  
何か間違いを見つけた方やもっと良い方法をご存知の方は、是非お教えください。


## どんなことが起こったか
iOSの「設定」アプリで端末の文字サイズを変更したところ、アプリのレイアウトが崩れてしまいました。Dynamic Typeには全く対応していないのに。


## こんな状況で起こった
状況によってレイアウトを切り替えたい場合ってありますよね。  
例えば、親Viewに一つボタンが乗っていて、  
通常は親Viewに対して左寄せに配置したいけど  
ある条件下では右寄せにしたい、とか。  
このアプリでもそのようなことを行っていました。

![layout.jpg]({{site.baseurl}}/images/20181219.jpg)

以下、デフォルトのレイアウトをA、そうでない方をBとします。  
<br/>
この2つのレイアウトを切り替えるために、Storyboardで以下のような設定をしてありました。

+ 一つのボタンに対し、2つのConstraintsを追加してある
	+ このConstraintsの優先度はどちらも同じ(1000)
	+ (これらのConstraintsは、両方activeになっているとコンフリクトする)
+ そのうちのデフォルトでない方は、Storyboard上にて設定のinstalledの項目をオフにしてある
+ ある条件下では、デフォルトのConstraintをinactiveにし、他方のConstraintをactiveにすることでレイアウトを切り替える (これはコード上で行う)



## こんな操作で起こった  
+ レイアウトA(デフォルト)のConstraintをinactiveにする
+ レイアウトB(非デフォルト)のConstraintをactiveにする
+ (iOSの設定アプリを開き、「一般」＞「アクセシビリティ」＞「さらに大きな文字」をオンにしておく)
+  上記画面または「画面表示と明るさ」＞「文字サイズを変更」の項目にて、文字サイズを変更する(大幅に変えると発生しやすい)

## 起こったこと(具体的に)
+ コンフリクトする2つのConstraintsのうち、
	+ レイアウトAのConstraintが、勝手にactiveに変わっている
	+ レイアウトBのConstraintが、勝手にinactiveに変わっている
+ そのため、レイアウトが崩れる！
	+ この操作ではレイアウトを変更する処理をどこにも書いていないのに・・・
	+ なんでなの？？？？

## 原因
iOSの文字サイズをある程度変えると、どうやらTrait Collectionが変わるらしいです。  
(そのため、traitCollectionDidChange:が呼ばれます。)  
多分これがレイアウトの勝手に変わる原因(?)になっていると思われます。  
<br/>

Storyboardのソースにおいて、これらのConstraintsに関連のある部分を見ていくと、以下のような箇所がありました。
<br/>

```
	// ... 略

	<constraints>
			<constraint firstItem="6Tk-OE-BBY" firstAttribute="trailing" secondItem="CF3-YR-pnl" secondAttribute="trailing" id="N9g-8d-bOB"/>
			<constraint firstItem="CF3-YR-pnl" firstAttribute="top" secondItem="6Tk-OE-BBY" secondAttribute="top" constant="27" id="VXt-i2-12s"/>
			<constraint firstItem="CF3-YR-pnl" firstAttribute="leading" secondItem="6Tk-OE-BBY" secondAttribute="leading" id="p6I-v7-ZLS"/>
	</constraints>
	<viewLayoutGuide key="safeArea" id="6Tk-OE-BBY"/>
	<variation key="default">
		 <mask key="constraints">
					<exclude reference="N9g-8d-bOB"/> // trailingのConstraintが、デフォルトでは除外されるということ？
			</mask>
	</variation>

	// ... 略
```
この `variation` というタグのところを見ると、何やらdefaultのConstantsを指定しているようです。  
自分ではそのような設定はしておらず、Xcodeが勝手に追加したものです。    
コンフリクトするConstraintsを設定した時に、自動的に追加されたものと思われます。

もしかしたらTrait Collectionが変わった時に、inactiveにしていたConstraintが何故かactiveになってしまうのかもしれません。    
そしてその際にコンフリクトしてしまったConstraintsがあった場合、元のデフォルトの状態に戻されるのかな？と推測しています。  



## 解決策
デフォルトでない方(レイアウトB)のConstraintの優先度を下げる(999などにする)ことで、この問題は起こらなくなりました。  
"Use Trait Collection"の項目もOFFにしておいた方がいいのかもしれません。  
ただこれを行うとactive / inactiveをStoryboard上で設定できなくなってしまうので、今回はやめました。これがOFFになっていなくても、不具合は直ります。


## 終わりに
なんとか不具合が起こらないようにできたものの、Trait Collectionが変わった時にConstantsのactive / inactiveが変わる理由というか、仕組みというか、その辺りがあまり理解できていません。  
なんでなの・・・？  


## 参考にしたサイト
+ [[iOS] AutoLayoutのNSConstraintを動的に変更したい](http://www.kuma-de.com/blog/2016-02-06/7119)
+ [Dynamic Type 対応について考える](https://qiita.com/MilanistaDev/items/cda50cde2993293ad6c8)
+ etc...
