# MSゴシック絶対許さんマン *MS gothic Police*

![eye_catch](https://s3-ap-northeast-1.amazonaws.com/fontkaruta2/eye_catch.jpg)

## 展示内容の紹介
8月5日（土）・6日（日）に開催された [Maker Faire Tokyo 2017](http://makezine.jp/event/mft2017/)にて、”フォント”の違いをディープラーニングで見分けるロボットアーム、**“MSゴシック絶対許さんマン”** を展示しました。  

「なんかダサい」「仕事の文書っぽくて気分が下がる」と、何かと嫌われている “MSゴシック” のフォントを自動識別し、[アーム](http://dobot.cc/dobot-magician/product-overview.html)で拾い上げて**床に捨てます**。  

本業でデータ分析や機械学習の業務を行っている会社のメンバー4人で作成し、「ナウい技術で超絶くだらないことをover killする」「”技術の無駄遣い感” を全面に押し出すこと」を目指しました！

「MSゴシック絶対許さんマン」がMSゴシックを識別して床に捨てている様子はこちらから見れます。  
（クリックでYoutubeに飛びます）  
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/qorjEbTfeR8/0.jpg)](https://www.youtube.com/watch?v=qorjEbTfeR8)


#### "フォントかるた"とは？
フォントかるたは、**テキストは全て同じで"フォント"だけが違う**というユニークなかるたです。今回はこのかるたを画像データとして取り込み、フォント識別アルゴリズムの学習データとして使わせていただきました

- [フォントかるた公式ページ](https://fontkaruta.wixsite.com/karuta)

下の画像に映っている10種類のフォントのうち、どれが「MSゴシック」かわかりますか？

![FontKaruta](https://s3-ap-northeast-1.amazonaws.com/fontkaruta2/fontkaruta_sample.jpg)

正解はこちら！

![FontKaruta_ans](https://s3-ap-northeast-1.amazonaws.com/fontkaruta2/fontkaruta_sample_ans2.jpg)

中央下段のカルタがMSゴシックでした！


## フォントを識別する方法
人間でも見分けるのが難しそうなフォントを、**深層学習**（畳み込みニューラルネットワーク）を使って機械的に識別する学習モデルを作成しました。  
48フォント分の学習データに対して、3層の畳み込み層 + [Spatial Pyramid Pooling](https://github.com/ysdyt/MSgothicPolice/blob/master/modules/layers.py)というPooling層を使って学習しています。ネットワーク構造の詳細は[こちら](https://github.com/ysdyt/MSgothicPolice/blob/master/modules/model.py)

以下がモデル全体のレイヤーとなります。以下ではinputの画像サイズが（320, 240）となっていますが、[実際のスクリプト](https://github.com/ysdyt/MSgothicPolice/blob/master/scripts/train.py)では可変サイズ（None, None）を学習するため、Predict時には任意のサイズの画像を指定することができます。

![model](https://s3-ap-northeast-1.amazonaws.com/fontkaruta2/model.png)

## データセットについて
フォントかるた製作者の許可もいただき、「フォントかるた」を卓上カメラで撮影、全48フォントに対して1000枚/fontづつデータ化しました。(train:validation:test = 700:200:100)  
データセットの半量は180度回転させた状態の画像も含む（加えて、data augumentationによってrotateした画像も含む）ため、Predict時にはかるたの向きを問いません。  
画像サイズは 縦:横 = 352:264 です（学習時はconvolutionしやすいように、resizeして320:240 にします）

※画像データセットの公開はもろもろ問題がありそうなため、行わないつもりです。ご容赦ください m(__)m

「MSゴシック」の画像データ例

![MSgothic_sample](https://s3-ap-northeast-1.amazonaws.com/fontkaruta2/img376_ms_gothic.jpg)![MSgothic_sample_reverse](https://s3-ap-northeast-1.amazonaws.com/fontkaruta2/img5_ms_gothic.jpg)


## 学習精度について
batch_size = 32, steps_per_epoch = 150, validation_steps = 100, epochs = 200 で上記データセットを学習させたときのaccとLossの推移は以下となります。

![learning_history](https://s3-ap-northeast-1.amazonaws.com/fontkaruta2/learning_history.png)

## テストデータへの当てはまり
各クラス100毎づつ用意したテストデータに対してpredictした結果です。  
最高に過学習していますがとりあえずこんな感じです。

![confusion_matrix](https://s3-ap-northeast-1.amazonaws.com/fontkaruta2/confusion_matrix.png)

## Grad-CAMによる分類結果可視化
ロボットアームを動かすデモだけでなく、Grad-CAMという手法を用いることで、学習ネットワークがフォントのどこを見て（どういった特徴量を重要視して）分類を行っているのかを可視化した説明展示も行いました。Chainerによる実装をチームメンバーの [@ywara93](https://twitter.com/ywara93) が行いました。

* [CNNは絵札のどの部分に注目してフォントを見分けているか - SlideShare](https://www.slideshare.net/YutaYoshida1/cnn-78650579)
* [suga93/fontkaruta - GitHub](https://github.com/suga93/fontkaruta_classifier)

## スクリプトの実行方法

### 環境

```
Python==3.5.2  
tensorflow-gpu==1.2.1  
Keras==2.0.6  
numpy==1.13.0  
```

その他、必要なPythonモジュールについては`requirements.txt`を参照してください。

### ファイル構成

```
MSgothicPolice                                                              
|-- README.md                                                  
|-- README_before_event.md                                     
|-- data                                                       
|   |-- test                                                   
|   |-- train                                                  
|   `-- validation                                             
|-- modules                                                    
|   |-- __init__.py                                            
|   |-- layers.py                                              
|   |-- model.py                                               
|   |-- utils.py                                               
|   `-- visualize.py                                           
|-- requirements.txt                                           
|-- result                                                                                             
`-- scripts                                                    
    `-- train.py                                               
```

### 実行

```bash
$ cd /MSgothicPolice/scripts
$ python3 train.py
```

### 出力ファイル

実行結果ファイルは`train.py`中の`result_path`に指定したパス以下に作成されます。
指定したパス以下に、`YYYYMMDD_HHMMSS`表記のディレクトリが作成され、その下に

* `hoge.hdf5`(重みファイル。lossが更新されるたびに作成される)
* `learning_history.png`(学習推移のプロット)
* `tensorboard`(tensorboard可視化用のデータ)

が作成されます。

## Web掲載

Maker Faire参加決定から当日の様子までをまとめました

* [Maker Faire Tokyo 2017 に初出展して「MSゴシック絶対許さんマン」を展示しました！- ysdyt.net](http://ysdyt.net/?p=2152)


その他、幾つかのメディアや個人ブログでも「MSゴシック許さんマン」を紹介していただきました。  

* [国内最大のMaker系フェス「Maker Faire Tokyo 2017」がお台場で開催中](https://fabcross.jp/news/2017/20170805_maker_faire_tokyo2017.html)
* [これぞモノづくり！Maker Faire Tokyo 2017に初参加した感想](http://temcee.hatenablog.com/entry/maker_faire_tokyo_2017)
* [テクノロジーの無駄遣い!?Maker Faire Tokyo2017に行ってきました!!](http://pleshe.jp/archives/2466)
	* 「この日みた中で最もテクノロジーの無駄遣い」という最高の評価をいただきました
* [Maker Faire Tokyo 2017 東８ホール – JH1LHVの雑記帳](http://jh1lhv.hatenablog.jp/entry/2017/08/10/212249)
* [頭脳集団「白金鉱業」さんの「MSゴシック絶対許さんマン」が凄かった（MFT2017展示紹介その2）](http://karaage.hatenadiary.jp/entry/2017/08/21/073000)
* [Maker Faire Tokyo 2017レポート「ポジティブなものづくり」の魅力](http://eonet.jp/zing/articles/_4100769.html)

本国Maker Faireを運営している団体「Make:」の公式webメディアでも写真付きで一分紹介されました。「親に向かってなんだそのMSゴシックは」のネタTシャツの英訳が晒されるという羞恥プレイ付き。”Font joke meets deep learning” というフレーズがイカしてる。

* [Live Updates: Maker Faire Tokyo 2017](http://makezine.com/2017/08/05/maker-faire-tokyo-2017-live-update/)
