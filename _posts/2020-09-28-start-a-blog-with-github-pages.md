---
layout: post
title: "GitHub Pagesでブログを作ってみた"
tag: github_pages
---
# GitHub Pagesでブログを作ってみた  

今までお世話になっていたQrunchのサービス終了に伴い、GitHub Pagesでブログを作ってみた。  

## GitHub Pagesを使ってみた理由

+ Markdownで記事を書ける
+ `.md`ファイルをGitHubに上げると、自動でHTMLに変換して公開してくれる仕組みがある
  + Qrunchの記事が`.md`ファイルでエクスポートされたので、楽に移行できそう
+ gitでコンテンツのバージョン管理ができる
+ ブログのレイアウトのテンプレートが予めいくつか用意されている
  + →手っ取り早く始められて、取っ付きやすい
+ レイアウトを好きなようにカスタマイズできそう

## 【作り方その1】シンプルにページを作成する方法  

以下のページが分かりやすかった：  
[GitHubを使ってMarkdown文書を５ステップでホームページとして公開する方法](https://qiita.com/MahoTakara/items/3800e9dc83b530d0a050)  

このページに書かれた手順で簡単に始められる。

### ハマった点  

`GitHub Pages`の設定を`None`から`master branch`へ変更したら、Themeの変更を行う前に一旦設定を保存する必要がある。  
→`GitHub Pages`の設定を変更した後、設定を保存する前にThemeの設定を行ったら、`gh-pages`というリポジトリが作成された。そしてトップページの内容が、`gh-pages`内のファイル(自動生成されたもの)で置き換わってしまった。

## 【作り方その2】Jekyllを使ってGitHub Pagesサイトを作成する方法  

ローカルでサイトのプレビューができるようにしたくなった。    
そのためにはJekyllを使ってサイトを作成する必要があるらしいので、試してみた。  

### 手順   

以下のサイトの内容に沿って行った：      
[Jekyll を使用して GitHub Pages サイトを作成する](https://docs.github.com/ja/github/working-with-github-pages/creating-a-github-pages-site-with-jekyll#creating-your-site)  

### ハマった点  

#### 1. Jekyllサイト作成時  

##### 症状  

[サイトを作成する](https://docs.github.com/ja/free-pro-team@latest/github/working-with-github-pages/creating-a-github-pages-site-with-jekyll#%E3%82%B5%E3%82%A4%E3%83%88%E3%82%92%E4%BD%9C%E6%88%90%E3%81%99%E3%82%8B)セクションの`7`にて、  
`bundle exec jekyll 3.9.0 new .`を実行すると、`Could not locate Gemfile or .bundle/ directory`とエラーを吐く。    
試しに`jekyll 3.9.0 new .`を実行すると、`-bash: jekyll: command not found`と言われてしまった。      

##### 解決方法

(色々試したので、ちょっとうろ覚え・・・)    

+ `gem update --system`を実行([→参考サイト](http://jekyllrb-ja.github.io/docs/troubleshooting/#installation-problems))  
  + 失敗したので、`sudo gem update --system -n /usr/local/bin`を実行([→参考サイト](https://hacknote.jp/archives/19804/))  
+ 再度`gem install --user-install bundler jekyll`を実行  
+ 以下のアドバイスを実行([→参考サイト](https://talk.jekyllrb.com/t/error-upon-bundle-exec-jekyll-3-8-7-new-for-github-pages/4561))：  
  ```
  Did you first run jekyll new <project-name>?
  If not start with that. In your Gemfile add gem 'jekyll', '3.8.7'.
  Run bundle update which will download version 3.8.7 of the gem.
  Then you can run bundle exec jekyll build or bundle exec jekyll serve.
  ```   

  (この時点ではbuildもserveもまだ通らないが、大丈夫)  
+ `$ bundle update`を実行  

#### 2. ローカルでのテスト時  

##### 症状  

[サイトを作成する](https://docs.github.com/ja/free-pro-team@latest/github/working-with-github-pages/creating-a-github-pages-site-with-jekyll#%E3%82%B5%E3%82%A4%E3%83%88%E3%82%92%E4%BD%9C%E6%88%90%E3%81%99%E3%82%8B)セクションの`11`→[サイトをローカルでビルドする](https://docs.github.com/ja/free-pro-team@latest/github/working-with-github-pages/testing-your-github-pages-site-locally-with-jekyll#%E3%82%B5%E3%82%A4%E3%83%88%E3%82%92%E3%83%AD%E3%83%BC%E3%82%AB%E3%83%AB%E3%81%A7%E3%83%93%E3%83%AB%E3%83%89%E3%81%99%E3%82%8B)セクションの`3`にて、  
(Gemfileに`gem "github-pages", "~> VERSION", group: :jekyll_plugins`を追記後、)  
`bundle exec jekyll serve`に失敗する。  
`Could not find gem 'github-pages (~> 3.9.0)' in any of the gem sources listed in your Gemfile.`とのエラーを吐く。  

##### 解決方法  

+ `github-pages`のバージョンの指定に誤りがあったので修正
+ 以下のアドバイスを実行([→参考サイト](https://github.com/prose/starter/issues/44))：
  ```
  this worked for me
  specify the version for gem "github-pages" in GEMFILE:
  gem "github-pages", "~> 203", group: :jekyll_plugins
  replace the version ( 203 ) above to the one from https://pages.github.com/versions/
  then run:
  $ bundle update jekyll
  followed by :
  $ bundle install
  test site by launching it locally :
  $ bundle exec jekyll serve
  ```

#### 3. ページの公開後  

##### 症状  

ローカルではレイアウトがうまくいっていたが、GitHub Pages上ではレイアウトが崩れている(CSSが効いていない感じ)  

##### 解決方法  

+ `_config.yml`の以下の箇所を修正([→参考サイト](https://marbles.hatenablog.com/entry/2019/01/16/221417))  
  ```
  # baseurl と url を GitHub Pages のリポジトリに合わせる。
  baseurl: "/tech_blog"
  url: "https://marshmallow444.github.io"
  ```  

##### 注意

+ 上記で`baseurl`の設定をGitHub Pages用に書き換えているので、ローカルでのテスト時は`jekyll serve --baseurl ""`のコマンドを使用する必要がある(でないとレイアウトが崩れる)

## 思ったこと

+ 最初はテスト用のリポジトリを作って、そこで色々試してみるのが良い
  + 失敗してもリポジトリごと消してやり直せる
+ GitHub Pagesではページのレイアウトもカスタマイズできるようなので、今後色々と試してみたい  

## 終わりに  

Qrunch、素晴らしいサービスでした。サービス終了してしまうのがとても残念です。  
自分はQrunchで技術ブログ的なものを始めました。Qrunchのおかげで、自分の中にあった「勉強したことをアウトプットする」ことに対する心理的ハードルが下がったように思います。  
今までお世話になりました。ありがとうございました。  

## 参考  

+ [Jekyll を使用して GitHub Pages サイトを作成する](https://docs.github.com/ja/github/working-with-github-pages/creating-a-github-pages-site-with-jekyll#creating-your-site)  
+ [Jekyll を使用して GitHub Pages サイトをローカルでテストする](https://docs.github.com/ja/github/working-with-github-pages/testing-your-github-pages-site-locally-with-jekyll)  
+ [Dependency versions](https://pages.github.com/versions/)  
+ [Jekyll on macOS](https://jekyllrb.com/docs/installation/macos/)  
+ [Troubleshooting - Jekyll](http://jekyllrb-ja.github.io/docs/troubleshooting/)  
+ [gem updateでエラーが出た時の対処法](https://hacknote.jp/archives/19804/)  
+ [Error upon `bundle exec jekyll 3.8.7 new .` for Github Pages](https://talk.jekyllrb.com/t/error-upon-bundle-exec-jekyll-3-8-7-new-for-github-pages/4561)  
+ [Could not find gem github-pages #44](https://github.com/prose/starter/issues/44)  
+ [ローカル上のJekyllサイトを GitHub Pages で公開する方法（単純に静的ファイルのUpLoadのみで）](https://marbles.hatenablog.com/entry/2019/01/16/221417)  
+ [Minimal Mistakes テーマではじめる Github Pages with Jekyll](https://k11i.biz/blog/2016/08/11/starting-jekyll-with-minimal-mistakes/)  
