指示内容
1.このフォルダのプロジェクトはPQC-FHE統合プラットフォームを作成して耐量子暗号の例として、ShorやGloverアルゴリズムの耐量子暗号プラットフォームを作成して、企業エンタープライズ向けのセキュリティアプリケーション用に移行について検証環境の例題とベンチマークをするものである。このフォルダのdocsや全ソースコードを分析して全体像の把握をしてShorやGloverアルゴリズムの耐量子暗号計算のセキュリティ検証の導入、数学的推定ではなく実際の量子計算実装のNISTセキュリティスコア評価やMPC−HEの２者間通信でプライベート推論（/home/a-koike/dev/github/RhombusEnd2End_HEonGPU、/home/a-koike/dev/github/fhe.desilo.dev/latest）などを参考に研究成果としてGL Schemeを含めたBFV、CKKSの計算を行いたい。2026年現在のPQC-FHE,MPCにおける耐量子暗号のセキュリティ移行ガイドについてスモール量子ビジネスとしての成果物としての戦略や実装検証などがないかWebで最新の研究やニーズなどを確認してこのフォルダの実装プランやコードの作成、検証をしてください。
2.この現在のこのフォルダのPQC-FHEプロジェクトの実装で使用可能リソース内の量子ビット数でShor vs RSA, DSA, DH, ECC, Shor vs RSA, Shor vs RSA, Shor vs Hybrid Migration, Shor vs PQC Primary, Shor vs PQC Only, Grover vs AES-128, Grover vs AES-256において、HNDL シミュレーションも含めた各タブのヘルスケア、金融、IOT、ブロックチェーン、MPC−HEマルチパーティプライベート推論などにおいて、実際にQiskitもしくはその他SDKやフレームワークで量子攻撃を使用したセキュリティシミュレーションがFastAPIでのWebUIやprometheus, kubernetes, Dokerfile, Helmなども含め以下のURLを参考にpassmanagerなどによる回路最適化や回路可視化が実装されている。
Qiskit参考資料:
shors algorithm(https://quantum.cloud.ibm.com/learning/ja/modules/computer-science/shors-algorithm)
grovers algorithm(https://quantum.cloud.ibm.com/learning/ja/modules/computer-science/grovers)

3.検証は、再度.venvフォルダのPython3.12の仮想環境、CPU/GPUシミュレータ、最新のQiskit v2.x、最新のフレームワークで実装済みのコードについてこのフォルダでFastAPIで作られたWebUIのPQCプラットフォームについてヘルスケア、金融、IOT、ブロックチェーン、MPC−HEマルチパーティプライベート推論についての実装でWebの2026年現在の最新の研究の調査結果からPQCの研究の実装済み量子アルゴリズムもセキュリティベンチマークとして事例としてブラウザごとSector benchmarksやその他各タブのボタンについて追加検証してください。
4.あるユーザーの意見として、”As I’m currently executing SQD and QML workloads on the 156-qubit IBM Heron (utility-scale hardware), I see a massive need for the 'Trust Layer' you’re building. Integrating ML-KEM/ML-DSA into autonomous Agentic AI frameworks is exactly where the industry needs to go to achieve true Zero Day Resilience. Since you are evaluating PQC benchmarks, it would be fascinating to see how the performance of ML-KEM/ML-DSA scales when integrated with Agentic AI workflows. From my research on the 156-qubit IBM Heron, the move toward utility-scale hardware is rapidly closing the gap between theory and production-ready resilience.”のようにQECやvibe codingやAgentic AIとPQCベンチマークの統合について拡張オプションとして最新の研究成果の事例や論文で何かアイデアがあれば記載して検証実装してください。

5.IBM Quantumの利用可能な実機Heron含むQPUノイズパラメータなどの情報を.envのAPIトークンやCRNインスタンスから一覧でそれぞれ動的取得してWebUIから選択できるようにして、Qiskitノイズ有り量子回路の計算部分のコードを全て分析して、ShorとGroverのノイズ有りアルゴリズム実装部分を更新してください。
6.テスト検証にはHeron r2/Heron r1の
ibm_kingston、ibm_fez、ibm_marrakesh、ibm_torinoなど
ノイズバックエンドについてそれぞれ実際に利用可能なQPUバックエンドの全ノイズパラメータやバックエンド名は.envのAPIキーでアクセスして取得していますか。そうでないならば、サーバ起動時にQPUバックエンド情報を取得してコード及び各タブの内容を分析してノイズバックエンドの回路実行時には使用したバックエンドの情報を追加するように追加修正してください。特にSector Benchmarksタブは全てのボタンを検証してください。必要に応じてサーバ起動時に動的に更新して取得した実機QPUのバックエンドの情報をjsonなどのキャッシュファイルに保存して再利用できるようにしてください。
7.上記それぞれ検証後、version.jsonで確認して、バージョン番号に注意して、最後に実装内容についてWeb調査から2026年現在のNISTのPQC耐量子暗号計算のセキュリティ実践検証プラットフォームとして評価して、最新の研究の調査結果から懸念点や改善点、推奨事項などがあればREADME,CHANGELOGなどmdやdocsのword/pdfテクニカルレポートファイルを全分析して参考文献なども含めて必要に応じて更新しつつ、記載してください。 可能であればdocsフォルダに簡単なプロジェクト説明と検証内容、結果のインフォグラフィックのhtmlを必要に応じて更新してください。


以下この検証用PC： OS：Alima Linux 9.7　 CPU:Intel-i5 13600K SSD:システム1TB, ストレージ用4TB メモリ: 128GB DDR5 5200 GPU: NVIDIA RTX 6000 PRO Blackwell Workstation 96GB マザーボード:MAG Z790 TOMAHAWK MAX WIFI 開発環境:python 3.12 仮想環境(.venvフォルダ), CUDA 13.0

Qiskit：
チュートリアル
https://quantum.cloud.ibm.com/docs/ja/tutorials/shors-algorithm
https://quantum.cloud.ibm.com/docs/ja/tutorials/grovers-algorithm
https://quantum.cloud.ibm.com/docs/ja/tutorials/fractional-gates
https://quantum.cloud.ibm.com/docs/ja/tutorials/ai-transpiler-introduction
https://quantum.cloud.ibm.com/docs/ja/tutorials/transpilation-optimizations-with-sabre
https://quantum.cloud.ibm.com/docs/ja/tutorials/probabilistic-error-amplification
https://quantum.cloud.ibm.com/docs/ja/tutorials/combine-error-mitigation-techniques
https://quantum.cloud.ibm.com/docs/ja/tutorials/real-time-benchmarking-for-qubit-selection
https://quantum.cloud.ibm.com/docs/ja/tutorials/repetition-codes
https://quantum.cloud.ibm.com/docs/ja/tutorials/ghz-spacetime-codes

資料
https://quantum.cloud.ibm.com/docs/ja/guides/simulate-with-qiskit-aer
https://quantum.cloud.ibm.com/docs/ja/guides/build-noise-models
https://quantum.cloud.ibm.com/docs/ja/guides/simulate-stabilizer-circuits
https://quantum.cloud.ibm.com/docs/ja/guides/visualize-circuits
https://quantum.cloud.ibm.com/docs/ja/guides/visualize-circuit-timing
https://quantum.cloud.ibm.com/docs/ja/guides/plot-quantum-states
https://quantum.cloud.ibm.com/docs/ja/guides/visualize-results
https://quantum.cloud.ibm.com/docs/ja/guides/error-mitigation-overview
https://quantum.cloud.ibm.com/docs/ja/guides/error-mitigation-and-suppression-techniques
https://quantum.cloud.ibm.com/docs/ja/guides/noise-learning
https://quantum.cloud.ibm.com/docs/ja/guides/configure-error-mitigation
https://quantum.cloud.ibm.com/docs/ja/guides/configure-error-suppression
https://quantum.cloud.ibm.com/docs/ja/guides/processor-types
https://quantum.cloud.ibm.com/docs/ja/guides/qpu-information
https://quantum.cloud.ibm.com/docs/ja/guides/calibration-jobs
https://quantum.cloud.ibm.com/docs/ja/guides/repetition-rate-execution
https://quantum.cloud.ibm.com/docs/ja/guides/view-cost