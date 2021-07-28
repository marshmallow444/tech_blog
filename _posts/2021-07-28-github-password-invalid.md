---
layout: post
title: "GitHubのパスワード認証が通らなかった話"
---

約1週間前まではGitHubのパスワード認証が使えていたのに、  
今日突然使えなくなっていたのでメモ。  

## 発生した問題

GitHubにソースをpushしようとしたが、エラー(403)が出て失敗した。  
以下のようなメッセージが出ていた。  

```
remote: Password authentication is temporarily disabled as part of a brownout. Please use a personal access token instead.
remote: Please see https://github.blog/2020-07-30-token-authentication-requirements-for-api-and-git-operations/ for more information.
```

## 解決方法

1. [Creating a personal access token](https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token) に書かれている手順でPersonal Access Tokenを作成する  

1. Terminalにて、以下のコマンドを実行する：  
`git config --global --add user.password 取得したトークン`
1. Keychain AccessにてGitHubの情報を探し、パスワード欄を取得したトークンで置き換える

この記事の通りだった。  
[GitHubの認証方法の新しいビッグウェーブに乗り遅れるな！](https://qiita.com/kanta_yamaoka/items/1a59892028b9c422df22)

## 備考

+ トークンの有効期限が切れたら、また上記手順を行う必要がある
+ 約1週間前に、GitHubからこんなメールが届いていた。気付かなかった・・・  

```
Hi (ユーザ名),

You recently used a password to access the repository at (リポジトリ名) with git using git/2.0 (libgit2 0.26.0).

Basic authentication using a password to Git is deprecated and will soon no longer work. Visit https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/ for more information around suggested workarounds and removal dates.

Thanks,
The GitHub Team
```