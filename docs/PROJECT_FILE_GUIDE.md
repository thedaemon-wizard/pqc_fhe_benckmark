# PQC-FHE Portfolio: Complete File Guide
## Week 1〜4 作成ファイル一覧と解説

---

## プロジェクト概要

**Post-Quantum Cryptography (PQC) + Fully Homomorphic Encryption (FHE)** を統合した
量子耐性暗号化システムのポートフォリオプロジェクトです。

- **総ファイル数**: 68ファイル
- **総コード量**: 約1.2MB
- **対応言語**: Python, YAML, Markdown

---

## ディレクトリ構成

```
pqc_fhe_portfolio/
├── .github/workflows/     # CI/CD設定
├── api/                   # REST/WebSocket APIサーバ
├── benchmarks/            # パフォーマンスベンチマーク
├── cli/                   # コマンドラインインターフェース
├── docs/                  # MkDocsドキュメント
│   ├── api/              # APIリファレンス
│   ├── guides/           # 運用ガイド
│   ├── security/         # セキュリティドキュメント
│   └── tutorials/        # チュートリアル
├── examples/              # 業界別デモアプリ
├── kubernetes/            # Helm Charts
├── monitoring/            # Prometheus設定
├── src/                   # ソースコード
└── tests/                 # テストスイート
```

---

# Week 1: コア実装

## 1. メインライブラリ

### `pqc_fhe_integration.py` (41KB)
**役割**: PQC + FHE統合のコアライブラリ

```python
# 主要クラス:
- PQCManager          # ML-KEM/ML-DSA鍵管理
- FHEEngine           # CKKS方式FHE演算エンジン
- HybridCryptoManager # PQC+FHE統合管理
- SecureSession       # 暗号化セッション管理
```

**機能**:
- ML-KEM-512/768/1024 鍵カプセル化
- ML-DSA-44/65/87 デジタル署名
- CKKS方式での加算・乗算・回転演算
- AES-256-GCM + PQC鍵交換によるハイブリッド暗号化

**科学的根拠**:
- NIST FIPS 203 (ML-KEM), FIPS 204 (ML-DSA)
- Cheon et al. (2017) CKKS方式

---

### `src/pqc_fhe_integration.py` (41KB)
**役割**: パッケージインストール用ソースコピー

同一内容を`src/`配下に配置し、`pip install`対応。

---

## 2. APIサーバ

### `api/server.py` (29KB)
**役割**: FastAPI REST APIサーバ

```python
# エンドポイント:
POST /api/v1/keys/generate          # PQC鍵ペア生成
POST /api/v1/keys/encapsulate       # ML-KEM鍵カプセル化
POST /api/v1/fhe/encrypt            # FHE暗号化
POST /api/v1/fhe/compute            # 暗号化演算
POST /api/v1/hybrid/encrypt         # ハイブリッド暗号化
GET  /health                        # ヘルスチェック
```

**セキュリティ機能**:
- レート制限 (100 req/min)
- APIキー認証
- CORS設定
- 入力バリデーション

---

### `api/websocket_server.py` (47KB)
**役割**: リアルタイムWebSocket APIサーバ

```python
# メッセージタイプ:
- pqc.keygen           # PQC鍵生成
- pqc.encapsulate      # 鍵カプセル化
- fhe.encrypt          # FHE暗号化
- fhe.compute          # 暗号化演算
- session.create       # セキュアセッション作成
- batch.execute        # バッチ処理
```

**特徴**:
- 双方向リアルタイム通信
- セッション管理
- バッチ処理対応
- 自動再接続

---

## 3. CLIツール

### `cli/main.py` (44KB)
**役割**: コマンドラインインターフェース

```bash
# 使用例:
pqc-fhe keygen --algorithm ml-kem-768
pqc-fhe encrypt --input data.txt --key pub.pem
pqc-fhe benchmark --iterations 1000
pqc-fhe server start --port 8080
```

**サブコマンド**:
- `keygen`: 鍵ペア生成
- `encrypt`/`decrypt`: 暗号化/復号
- `sign`/`verify`: 署名/検証
- `benchmark`: パフォーマンス測定
- `server`: APIサーバ起動

---

## 4. ベンチマーク

### `benchmarks/__init__.py` (22KB)
**役割**: パフォーマンス測定フレームワーク

```python
# 測定項目:
- PQC鍵生成時間
- カプセル化/脱カプセル化時間
- FHE暗号化/復号時間
- 暗号化演算時間
- メモリ使用量
```

**出力形式**:
- コンソール表示
- JSON/CSVエクスポート
- Prometheus形式メトリクス

---

## 5. テストスイート

### `tests/test_pqc_fhe.py` (28KB)
**役割**: 単体・統合テスト

```python
# テストカバレッジ:
- test_pqc_keygen()           # 鍵生成テスト
- test_kem_encapsulate()      # KEMテスト
- test_fhe_operations()       # FHE演算テスト
- test_hybrid_workflow()      # ハイブリッドワークフロー
- test_security_properties()  # セキュリティ特性
```

**フレームワーク**: pytest + hypothesis (プロパティベーステスト)

---

## 6. デモアプリケーション

### `examples/healthcare_demo.py` (25KB)
**役割**: 医療データプライバシー保護デモ

```python
# シナリオ:
- 患者データの暗号化保存
- 暗号化状態での統計計算
- HIPAA準拠データ共有
```

---

### `examples/financial_demo.py` (22KB)
**役割**: 金融取引セキュリティデモ

```python
# シナリオ:
- 暗号化ポートフォリオ計算
- プライバシー保護型リスク分析
- 量子耐性取引署名
```

---

### `examples/blockchain_demo.py` (29KB)
**役割**: ブロックチェーン統合デモ

```python
# シナリオ:
- 量子耐性スマートコントラクト署名
- 暗号化トランザクション処理
- PQC対応ウォレット
```

---

### `examples/iot_demo.py` (32KB)
**役割**: IoTデバイスセキュリティデモ

```python
# シナリオ:
- 軽量PQC (ML-KEM-512)
- センサーデータ暗号化集約
- エッジコンピューティング対応
```

---

## 7. インフラ設定

### `Dockerfile` (2.7KB)
**役割**: 本番用Dockerイメージ

```dockerfile
# マルチステージビルド:
- Builder: 依存関係コンパイル
- Production: 最小イメージ (python:3.11-slim)
# セキュリティ:
- 非rootユーザー実行
- 読み取り専用ファイルシステム
```

---

### `Dockerfile.gpu` (3.2KB)
**役割**: GPU対応Dockerイメージ

```dockerfile
# ベース: nvidia/cuda:12.1-runtime
# 対応: CUDA加速FHE演算
```

---

### `docker-compose.yml` (4.2KB)
**役割**: 開発環境オーケストレーション

```yaml
services:
  pqc-fhe-api:     # APIサーバ
  redis:           # セッションストア
  prometheus:      # メトリクス収集
  grafana:         # 可視化ダッシュボード
```

---

### `kubernetes/helm/pqc-fhe/` (14ファイル, 67KB)
**役割**: Kubernetes Helm Chart

| ファイル | 役割 |
|---------|------|
| `Chart.yaml` | Chartメタデータ |
| `values.yaml` | デフォルト設定値 |
| `deployment.yaml` | Podデプロイメント |
| `service.yaml` | Kubernetesサービス |
| `ingress.yaml` | Ingressルール |
| `hpa.yaml` | 水平Podオートスケーラー |
| `pdb.yaml` | Pod Disruption Budget |
| `configmap.yaml` | 設定マップ |
| `secret.yaml` | シークレット管理 |
| `networkpolicy.yaml` | ネットワークポリシー |
| `serviceaccount.yaml` | サービスアカウント |
| `servicemonitor.yaml` | Prometheus監視 |
| `NOTES.txt` | インストール後メモ |

---

### `monitoring/prometheus.yml` (2.7KB)
**役割**: Prometheus収集設定

```yaml
# メトリクス:
- pqc_keygen_duration_seconds
- fhe_operation_duration_seconds
- api_request_latency_seconds
```

---

### `.github/workflows/ci.yml` (7.4KB)
**役割**: GitHub Actions CI/CDパイプライン

```yaml
jobs:
  test:           # pytest実行
  security-scan:  # Trivy, Bandit脆弱性スキャン
  build:          # Dockerイメージビルド
  deploy-staging: # ステージングデプロイ
  deploy-prod:    # 本番デプロイ (タグ時)
```

---

## 8. プロジェクト設定

### `pyproject.toml` (4.3KB)
**役割**: Pythonパッケージ設定 (PEP 517/518)

```toml
[project]
name = "pqc-fhe-integration"
version = "1.0.0"
dependencies = [
    "liboqs-python>=0.10.0",
    "tenseal>=0.3.14",
    "fastapi>=0.109.0",
    ...
]
```

---

### `requirements.txt` (2.5KB)
**役割**: pip依存関係リスト

---

### `Makefile` (6.1KB)
**役割**: ビルド自動化

```makefile
make install    # 依存関係インストール
make test       # テスト実行
make benchmark  # ベンチマーク実行
make docs       # ドキュメントビルド
make docker     # Dockerイメージビルド
make deploy     # Kubernetesデプロイ
```

---

### `README.md` (11KB)
**役割**: プロジェクト概要・クイックスタート

---

### `SECURITY.md` (6.7KB)
**役割**: セキュリティポリシー・脆弱性報告手順

---

### `CONTRIBUTING.md` (7.2KB)
**役割**: コントリビューションガイドライン

---

### `CHANGELOG.md` (3.7KB)
**役割**: バージョン変更履歴

---

### `LICENSE` (2.3KB)
**役割**: Apache License 2.0

---

### `quickstart.py` (9.4KB)
**役割**: インタラクティブクイックスタートスクリプト

```bash
python quickstart.py
# → 対話形式でPQC/FHE機能をデモ
```

---

# Week 2: MkDocsドキュメントサイト

## 1. サイト設定

### `mkdocs.yml` (3.3KB)
**役割**: MkDocsサイト設定

```yaml
site_name: PQC-FHE Integration
theme: material
plugins:
  - search
  - mkdocstrings  # API自動ドキュメント生成
nav:
  - Home: index.md
  - Getting Started: guides/
  - Tutorials: tutorials/
  - API Reference: api/
  - Security: security/
```

---

## 2. ホーム・概要

### `docs/index.md` (5.1KB)
**役割**: ドキュメントサイトトップページ

---

### `docs/ARCHITECTURE.md` (47KB)
**役割**: システムアーキテクチャ詳細

```
内容:
- レイヤード設計図
- コンポーネント相互作用
- データフロー
- セキュリティアーキテクチャ
```

---

### `docs/API.md` (13KB)
**役割**: API概要ドキュメント

---

## 3. ガイド (guides/)

### `docs/guides/installation.md` (2.8KB)
**役割**: インストール手順

```markdown
# サポートOS: Ubuntu 22.04+, macOS 13+, Windows 11
# Python: 3.10+
pip install pqc-fhe-integration
```

---

### `docs/guides/quickstart.md` (4.2KB)
**役割**: 5分クイックスタート

---

### `docs/guides/configuration.md` (6.0KB)
**役割**: 設定オプション詳細

```yaml
# 設定ファイル例:
pqc:
  default_algorithm: ml-kem-768
fhe:
  poly_modulus_degree: 8192
  coeff_modulus_bits: [60, 40, 40, 60]
```

---

## 4. チュートリアル (tutorials/)

### `docs/tutorials/pqc_key_exchange.md` (20KB)
**役割**: PQC鍵交換チュートリアル

```python
# 学習内容:
- ML-KEM鍵ペア生成
- 鍵カプセル化/脱カプセル化
- 共有秘密導出
```

---

### `docs/tutorials/fhe_computation.md` (17KB)
**役割**: FHE演算チュートリアル

```python
# 学習内容:
- CKKS暗号化
- 暗号化加算・乗算
- ベクトル演算
- ブートストラップ
```

---

### `docs/tutorials/hybrid_workflow.md` (32KB)
**役割**: ハイブリッドワークフローチュートリアル

```python
# 学習内容:
- PQC鍵交換 + FHEデータ処理
- エンドツーエンド暗号化パイプライン
- マルチパーティ計算
```

---

### `docs/tutorials/enterprise_integration.md` (55KB)
**役割**: エンタープライズ統合ガイド

```python
# 内容:
- AWS/GCP/Azure統合
- HSM連携
- ログ・監査
- コンプライアンス対応
```

---

### `docs/tutorials/benchmarks.md` (28KB)
**役割**: ベンチマーク実行ガイド

---

### `docs/tutorials/use_cases.md` (39KB)
**役割**: 実世界ユースケース集

```
- 医療: HIPAA準拠データ分析
- 金融: プライバシー保護型リスク計算
- 政府: 機密データ処理
- IoT: エッジセキュリティ
```

---

## 5. APIリファレンス (api/)

### `docs/api/overview.md` (3.9KB)
**役割**: API概要

---

### `docs/api/pqc_manager.md` (9.6KB)
**役割**: PQCManager APIリファレンス

```python
class PQCManager:
    def generate_keypair(algorithm: str) -> KeyPair
    def encapsulate(public_key: bytes) -> Tuple[bytes, bytes]
    def decapsulate(ciphertext: bytes, secret_key: bytes) -> bytes
    def sign(message: bytes, secret_key: bytes) -> bytes
    def verify(message: bytes, signature: bytes, public_key: bytes) -> bool
```

---

### `docs/api/fhe_engine.md` (16KB)
**役割**: FHEEngine APIリファレンス

```python
class FHEEngine:
    def encrypt(data: List[float]) -> Ciphertext
    def decrypt(ciphertext: Ciphertext) -> List[float]
    def add(ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext
    def multiply(ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext
    def bootstrap(ciphertext: Ciphertext) -> Ciphertext
```

---

### `docs/api/hybrid_manager.md` (19KB)
**役割**: HybridCryptoManager APIリファレンス

---

### `docs/api/rest_api.md` (13KB)
**役割**: REST API仕様 (OpenAPI形式)

---

### `docs/api/cli.md` (8.8KB)
**役割**: CLIコマンドリファレンス

---

## 6. セキュリティドキュメント (security/)

### `docs/security/overview.md` (4.6KB)
**役割**: セキュリティ概要

---

### `docs/security/best_practices.md` (18KB)
**役割**: セキュリティベストプラクティス

```
- 鍵管理ガイドライン
- 安全なデプロイメント
- 監査ログ設定
- インシデント対応
```

---

### `docs/security/nist_compliance.md` (23KB)
**役割**: NIST準拠ドキュメント

```
対応標準:
- FIPS 203 (ML-KEM)
- FIPS 204 (ML-DSA)
- NIST SP 800-56C (鍵導出)
- NIST IR 8547 (PQC移行)
```

---

# Week 3: 高度なAPIリファレンス・セキュリティドキュメント

### `docs/api/websocket_api.md` (22KB)
**役割**: WebSocket API完全リファレンス

```javascript
// メッセージ形式:
{
  "type": "pqc.keygen",
  "request_id": "uuid",
  "payload": { "algorithm": "ml-kem-768" }
}
```

**内容**:
- 全メッセージタイプ仕様
- エラーハンドリング
- 再接続ロジック
- セッション管理

---

### `docs/security/threat_model.md` (18KB)
**役割**: 脅威モデルドキュメント

```
脅威カテゴリ:
- STRIDE分析
- 量子攻撃シナリオ
- サイドチャネル攻撃
- 中間者攻撃
- 緩和策マトリクス
```

---

# Week 4: 運用ガイド

### `docs/guides/performance_optimization.md` (62KB)
**役割**: パフォーマンス最適化ガイド

```python
# 内容:
- ベースラインメトリクス定義
- PQCアルゴリズム選択基準
- 鍵キャッシュ実装
- FHEパラメータチューニング
- ノイズバジェット管理
- メモリ最適化
- 並列処理実装
- ベンチマークフレームワーク
```

**科学的根拠**:
- Halevi & Shoup (2020): FHE最適化
- Albrecht et al. (2018): LWE Estimator
- Bernstein & Lange (2017): PQCパフォーマンス

---

### `docs/guides/deployment.md` (63KB)
**役割**: 本番デプロイメントガイド

```yaml
# 内容:
- リファレンスアーキテクチャ
- Docker本番設定
- Kubernetes完全マニフェスト
- AWS CDK (Python)
- GCP Terraform
- Prometheus/Grafana監視
- セキュリティチェックリスト
- CI/CDパイプライン
- ヘルスチェック実装
```

**セキュリティ基準**:
- NIST SP 800-53
- CIS Kubernetes Benchmark
- OWASP
- FIPS 140-3

---

### `docs/guides/migration.md` (70KB)
**役割**: PQC移行ガイド

```python
# 内容:
- NIST移行タイムライン (2024-2035)
- 移行戦略比較
- ハイブリッド移行実装
- 段階的移行フレームワーク
- データ移行マネージャー
- FHE移行評価
- ロールバック手順
- 移行テストスイート
```

**科学的根拠**:
- NIST IR 8547: PQC移行標準
- ETSI TS 103 744: ハイブリッド鍵交換
- RFC 9180: HPKE

---

# ファイルサイズ一覧

## コアコード (Python)

| ファイル | サイズ | 行数(概算) |
|---------|-------|-----------|
| `pqc_fhe_integration.py` | 41KB | ~1,200行 |
| `api/server.py` | 29KB | ~850行 |
| `api/websocket_server.py` | 47KB | ~1,400行 |
| `cli/main.py` | 44KB | ~1,300行 |
| `benchmarks/__init__.py` | 22KB | ~650行 |
| `tests/test_pqc_fhe.py` | 28KB | ~820行 |
| `examples/*.py` (4ファイル) | 108KB | ~3,200行 |

**コード合計**: ~320KB, ~9,400行

---

## ドキュメント (Markdown)

| ファイル | サイズ |
|---------|-------|
| `docs/guides/migration.md` | 70KB |
| `docs/guides/deployment.md` | 63KB |
| `docs/guides/performance_optimization.md` | 62KB |
| `docs/tutorials/enterprise_integration.md` | 55KB |
| `docs/ARCHITECTURE.md` | 47KB |
| `docs/tutorials/use_cases.md` | 39KB |
| `docs/tutorials/hybrid_workflow.md` | 32KB |
| `docs/tutorials/benchmarks.md` | 28KB |
| `docs/security/nist_compliance.md` | 23KB |
| `docs/api/websocket_api.md` | 22KB |
| その他ドキュメント | ~130KB |

**ドキュメント合計**: ~570KB

---

## インフラ設定 (YAML/Dockerfile)

| ファイル | サイズ |
|---------|-------|
| Kubernetes Helm (14ファイル) | ~67KB |
| `.github/workflows/ci.yml` | 7.4KB |
| `docker-compose.yml` | 4.2KB |
| `Dockerfile` + `Dockerfile.gpu` | 5.9KB |
| `prometheus.yml` | 2.7KB |

**インフラ設定合計**: ~87KB

---

# 技術スタック

## 暗号ライブラリ

| 用途 | ライブラリ | バージョン |
|-----|-----------|-----------|
| PQC | liboqs-python | ≥0.10.0 |
| FHE | TenSEAL | ≥0.3.14 |
| 対称暗号 | cryptography | ≥42.0.0 |

## フレームワーク

| 用途 | ライブラリ |
|-----|-----------|
| REST API | FastAPI |
| WebSocket | websockets |
| CLI | Click/Typer |
| テスト | pytest + hypothesis |
| ドキュメント | MkDocs Material |

## インフラ

| 用途 | ツール |
|-----|-------|
| コンテナ | Docker |
| オーケストレーション | Kubernetes + Helm |
| CI/CD | GitHub Actions |
| 監視 | Prometheus + Grafana |

---

# 参考文献

## NIST標準
- FIPS 203: ML-KEM (Kyber)
- FIPS 204: ML-DSA (Dilithium)
- FIPS 205: SLH-DSA (SPHINCS+)
- NIST IR 8547: PQC移行ガイドライン
- NIST SP 800-56C: 鍵導出

## 学術論文
- Cheon et al. (2017): CKKS方式
- Halevi & Shoup (2020): HElib最適化
- Albrecht et al. (2018): LWE Estimator
- Bernstein & Lange (2017): PQCパフォーマンス

## 業界標準
- ETSI TS 103 744: ハイブリッド鍵交換
- RFC 9180: HPKE
- CIS Kubernetes Benchmark
- OWASP Security Guidelines

---

*Generated: 2025-12-25*
*Project: PQC-FHE Integration Portfolio*
