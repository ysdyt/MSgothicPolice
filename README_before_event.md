# MSゴシック絶対許さんマン *MS gothic Police*

[Maker Faire Tokyo 2017](http://makezine.jp/event/mft2017/)への参加を目指して作成しています。  


（クリックでYoutubeに飛びます）  
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/1prXpTHX1e0/0.jpg)](https://www.youtube.com/watch?v=1prXpTHX1e0)

## 出展内容の紹介
“フォント”の違いをディープラーニングで見分けるロボットアーム、**MSゴシック絶対許さんマン**を展示します。  
「なんかダサい」「仕事の文書っぽくて気分が下がる」と何かと不遇の扱いを受ける”MSゴシック”のフォントを自動識別しアームで拾い上げてゴミ箱に捨てます。  
また当日は「フォントかるた」を使った、人間VS機械の仁義なきかるたバトルにも挑戦できます。フォントに熱い想いのある方の挑戦をお待ちしています！

### "フォントかるた"とは？
フォントかるたは、**テキストは全て同じで"フォント"だけが違う**というユニークなかるたです。今回はこのかるたを画像データとして取り込み、フォント識別アルゴリズムの学習データとして使わせていただいています（※フォントかるた作成者の許可もいただきました）

- フォントかるた公式ページ - https://fontkaruta.wixsite.com/karuta

下の画像に映っているのは全部違うフォントです。違いがわかりますか？
![FontKaruta](https://s3-ap-northeast-1.amazonaws.com/fontkaruta2/FontKaruta.png)

正解はこちら！
![FontKaruta_ans](https://s3-ap-northeast-1.amazonaws.com/fontkaruta2/FontKaruta_ans.png)

### フォントを識別する方法
人間でも見分けるのが難しそうなフォントを、**深層学習**という手法を使って、機械的に識別する学習モデルを作成します。  
最終的にはフォントかるたに収録されている48種類全てを識別できるようにする予定です（只今識別精度アップを目指してモデルの改善中です）

ちなみにロボットアームは[Dobot Magician](http://dobot.cc/dobot-magician/product-overview.html)を使用しています