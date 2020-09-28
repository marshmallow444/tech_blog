---
title: 今更ながらIFTTTを初めて使ってみた
tags:  IFTTT
categories:
author: marshmallow444
status: public
created_at: 2019-03-10 00:43:29 +0900
updated_at: 2019-03-10 00:43:29 +0900
published_at: 2019-03-10 00:43:29 +0900
---
会社でIFTTTの話が出て、面白そうだから使ってみようと思い立ち、実践してみたという話。
ごく簡単なAppletを作成してみた。

## **[IFTTT](https://ifttt.com/)** とは

IFTTT(イフト)とは「レシピ」と呼ばれる個人作成もしくは公に共有しているプロフィールを使って数あるWebサービス（Facebook、Evernote、Weather、Dropboxなど）同士で連携することができるWebサービスである。([Wikipediaより](https://ja.wikipedia.org/wiki/IFTTT))

## 連携できるサービス

https://ifttt.com/services に掲載されているものが連携可能。
以下が特に便利そう：

+ LINE
+ Gmail
+ Googleの色々なサービス(Googleカレンダー、Google Assistant、Google Sheetなど)
+ Alexa
+ Evernote
+ Dropbox
+ Facebook
+ Twitter
+ GitHub
+ Slack
+ SMS
+ Weather Underground
+ Location
+ Notification (IFTTTのモバイルアプリの機能)


## まずはアカウント登録

メールアドレスを入力してIFTTTのアカウントを作成する。
GoogleやFacebookのアカウントを使って登録することも可能。

## アプレットの作り方

下記サイトにて説明されている
https://tanuhack.com/useful-tool/other-tools/ifttt-intro2/  

自分で作らなくても、他の人が作ったアプレットを使うこともできる。
IFTTTのwebページを見ると、たくさんのアプレットが登録されている。

## とりあえず、何かアプレットを作ってみる

Google Assistantに「帰ります」と話しかけると、家族に「今から帰ります！」とLINEしてくれるアプレットを作ってみた。

#### `This`の設定

+ サービス一覧から`Google Assistant`を選択
+ `Google Assistant`と`IFTTT`を連携させる (まだ連携していない場合のみ)
	+ 連携させたいGoogleアカウントを選択してログインする
+ `Say a simple phrase`を選択
+  `What do you want to say?`に、話しかけたい内容を入力
	+ 今回は「帰ります」と入力した
+ `What do you want the Assistant to say in response?`に、Google Assistantからの返答を入力
	+ 今回は「LINEしました」と入力した (適当)
+ `Language`を`Japanese`にする

#### `That`の設定

+ サービス一覧から`LINE`を選択
+ `LINE`と`IFTTT`を連携させる (まだ連携していない場合のみ)
	+ LINEアカウントのメールアドレスとパスワードでログインする
	+ 連携が完了すると、`LINE Notify`というアカウントと自動的に友だちになる
+ `Send message`を選択
+ 送信先を選択
	+ 個人の宛先は選択できないらしく、選択肢の中に表示されない
	  受信者と`LINE Notification`のアカウントを含むグループを作成し、それを指定した
+ `Message`欄へ、LINEで送信するメッセージ本文を入力する
	+ 今回は「今から帰ります！」と入力した
+ `Create Action`ボタンを押下

最後に`Review & Finish`の画面が表示されるので、
+ Appletの名前を入力
+ `Receive notifications when this Applet runs`のON/OFFを設定する
	+ これは基本的に不要と思われる
+ `Finish`ボタンを押下

**→完了！**

この状態でAndroidスマホでGoogle Assistantを立ち上げて「帰ります」と話しかけてみたところ、無事LINEへ`[IFTTT]「今から帰ります！」`というメッセージが送信された。成功！


## 感想

とても簡単にAppletを作成することができた。今回は非常に単純なものを作ったが、GAS等と連携すればもっとできることが増えそう。
[Qiitaの`IFTTT`タグの記事一覧](https://qiita.com/tags/ifttt/items)をみてみたら面白そうな記事がいくつもあったので、今後はもっと複雑なAppletも作ってみたい。

## Qiitaで見つけたAppletの記事

以下のものが特に面白そうだった。

+ [遅くまで会社にいたら自動で「帰り遅くなる」とメールする](https://qiita.com/ikechan/items/5e4bf2b9868d0d804af5)
+ [育児ゼロアクションプロジェクト(IKZAP)ができるまで](https://qiita.com/valitoh/items/02cdbbdaa24d4c0d99a7)
+ [旅程をLINEに知らせてもらおう](https://qiita.com/koushiro/items/4c6e5dd32e7ce1f2e520)
+ [Googleアシスタントのルーティン機能を使ってみた](https://qiita.com/h-takauma/items/7604a66de3b05e6e299e)
+ [遅刻防止システムを考えて作ってみた](https://qiita.com/ito_shin_2017/items/4b4e7eb3d405bdc641fb)

## おまけ

「会社を出たら自動的にLINEでメッセージ送信」するAppletも、同じような手順で簡単に作成できる。その方が便利かな？とも思ったが、会社を出ると帰宅時以外でも問答無用でAppletが走ってしまうようなので、今回は見送った。
