---
title: iOSアプリが謎のリジェクトをくらった話
tags:  iOS iPadOS Apple
categories:
author: marshmallow444
status: public
created_at: 2020-02-15 18:22:52 +0900
updated_at: 2020-02-15 18:22:52 +0900
published_at: 2020-02-15 18:22:52 +0900
---
自分の担当したiOSアプリが謎のリジェクトをくらったので、メモを残しておく。


## Appleから指摘されたリジェクト理由   

`Your app declares support for audio in the UIBackgroundModes key in your Info.plist, but we were unable to play any audible content when the app was running in the background.`    

バックグラウンドオーディオの設定がONになっているのに、バックグラウンドでオーディオを使ってないじゃん！とのこと。


## 背景  

+ このアプリはバックグラウンドにいる状態でもオーディオをコントロールする必要があり、  
AudioUnitを動かし続ける必要がある。実際にそのような挙動になっている。
+ このアプリは既に一回リリースされており、バージョンアップ時の審査にてリジェクトを受けた。
 今回のバージョンアップでは、オーディオ周りの処理には一切変更を加えていない。
 類似アプリでも数年前から同じ処理を行っていたが、今までリジェクトを受けたことはなかった。

## 対策  

 バックグラウンドでもオーディオを使用している旨を、Appleにアピールした。
 その結果、審査が通った。(実装は一切変更していない)


## 備考  

 バックグラウンドでオーディオが使用されていることにAppleのレビュワーが気づかず
 リジェクトされる例が、しばしば発生している様子。
 ネットで同様のケースが散見された。
 例： [https://stackoverflow.com/questions/45110324/my-app-rejected-due-to-audio-in-the-uibackgroundmodes-key](https://stackoverflow.com/questions/45110324/my-app-rejected-due-to-audio-in-the-uibackgroundmodes-key)


## 所感  

iOSアプリ開発に詳しい先輩にこのことを話したところ、  
「そういうことって結構あるよね」と言われた。そういうものなのか・・・  
自分は数年間iOSアプリ開発に携わってきたが、  
自分のチームのアプリは今までリジェクトされたことがなかったので  
びっくりしてしまった。
