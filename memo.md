
さて次の機能も実装してください。適宜ローカルのai-toolkitも参照してください。
・LRをTE1/TE2/U-netな(SDXL)ど分離して管理する方法
 アーキテクチャを自動検出して設定UIに反映できるように。
　また各コンポーネントをtrainするかどうかも選べるようにする。
・Aspect ratio bucketing および base resolutionの設定(resolutionは複数設定できるように)
・Bucketingできないほど大きかった画像をどう扱うかのオプション
　デフォルトはresブロックのbucketに縮小（Lanczosを使う）
　オプションで一部をcropしてbucketing(SDXLの場合はtime_idsのクロップ情報やtarget resolutionなどに反映)、このときどのアスペクトにいれるかはランダムでよい
　また複数res指定の時は最大のresブロックにbucketingするか、ランダムなresにbucketingするかオプションとする。
・ai-toolkitの初回ファイル確認は画像・キャプションをそれぞれ確認しておりかなり時間がかかってしまうため非合理的。フロントエンド側でデータベースに欠損がないなら所在確認は不要。したがってデータベース側には解像度情報を予め保持しておく。ここで破損している画像もはじいておく。
・バックエンド側で開けない(画像が破損しているなど)場合はbucketingの時点でskipするエラーハンドリングを。
・PILを使用している場合は、大きすぎる画像も扱えるように（参考：https://kakashibata.hatenablog.jp/entry/2022/03/27/232553）。いずれにせよbucketing時に縮小かcropされるので問題ない。
・weight dtype  (デフォルトfp16, オプションfp32, bf16, fp8e4, fp8e5)
・training(activation) dtype (デフォルトfp16, オプションbf16, fp8e4, fp8e5)
・output dtype (デフォルトfp32, オプションfp16, bf16, fp8e4, fp8e5)
・mixed precisionによるVRAM削減

他、ai-toolkitで実装している有用な機能があれば参考に。