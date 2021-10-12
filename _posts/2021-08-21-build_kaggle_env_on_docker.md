---
layout: post
title: "Dockerを使ってKaggleの環境を構築する"
tags: Kaggle
---

10/12追記：  
VSCodeを使いたい場合は「[ローカルでDocker+VSCodeを使ったKaggleの環境構築メモ](../../10/12/kaggle-env-on-docker-with-vscode.html)」の方法がおすすめ。  
本記事の方法は、上記記事より手軽に試したい場合におすすめ。  

---

<br>

「[爆速でKaggle環境を構築する](https://qiita.com/bam6o0/items/354faa9394755a984661)」の手順に従って、Docker初心者がmac OS上にKaggleのNotebookの実行環境を構築してみたメモ。  

## 環境構築手順

1. Dockerをインストールする
1. terminalにて、以下のコマンドを実行する
    ```
    $ docker run --rm -it kaggle/python
    ```
    →これにより、Kaggle Python docker imageがPullされる

#### つまづいたところ

+ コマンド実行時に、以下のようなエラーが出て失敗した  
    ```
    Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
    ```
    Dockerを起動してから実行することで、うまくいった  
    ```
    open /Applications/Docker.app
    ```

## Notebookの実行方法

1. 以下のコマンドを実行する  
    ```
    docker run -v $PWD:/tmp/working -w=/tmp/working -p 8888:8888 --rm -it kaggle/python jupyter notebook --no-browser --ip="0.0.0.0" --notebook-dir=/tmp/working --allow-root
    ```
1. いくつかURLが出力されるので、ブラウザでそのいずれかのURLへアクセスする
1. `.ipynb`ファイルを開くと編集・実行できる

#### 1.のコマンドを短くする方法

`.bash_profile`へ、以下を追記する  
```
kaggle_jupyter() {
  docker run -v $PWD:/tmp/working -w=/tmp/working -p 8888:8888 --rm -it kaggle/python jupyter notebook --no-browser --ip="0.0.0.0" --notebook-dir=/tmp/working --allow-root
}
```
これにより、terminalで以下を実行するだけでよくなる
```
$ kaggle_jupyter
```

## メモ

+ 実行にかかった時間は、[Dockerを使わずVSCodeで実行した時](../19/run-kaggles-code-in-local-env.html)に比べると1.25倍ほどかかった。それでもKaggleのKernel上で実行するより早い。いちいちローカルにライブラリをインストールする必要がないので、一度この方法で環境構築しておくと後が楽になりそう
+ コマンド内の`--rm`の意味：クリーンアップ
    + コンテナの終了時に、自動的にコンテナをクリーンアップし、ファイルシステムを削除する
+ コマンド内の`-it`の意味：オプション`-i`と`-t`の指定
    + `-i`(=`--interactive`)：標準入力
    + `-t`(=`--tty`)：擬似端末
+ Dockerのコンテナの中にログインするには、以下のコマンドを実行する
    ```
    docker run -it (コンテナ名) bash
    ```
+ この方法はかなり多くの容量を要するので、クラウド上にこの環境を構築する手もある。もしmacの容量が足りなくなったら試してみるのもいいかも

## 参考

+ [爆速でKaggle環境を構築する](https://qiita.com/bam6o0/items/354faa9394755a984661)
+ [Docker run リファレンス](https://docs.docker.jp/engine/reference/run.html#clean-up-rm)
+ [docker run -it の「-it」とはなにか](https://qiita.com/k_uchida_____/items/8ca31226bd6d10850791)
+ [MacでDockerのインストールとチュートリアルまで](https://www.task-notes.com/entry/20191013/1570961482)
+ [kaggleやろおぜ(環境構築編)](https://scitaku.hatenablog.com/entry/2019/06/09/005657)
