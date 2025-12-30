# Changelog

All notable changes to the PQC-FHE Integration Platform.

## [2.3.4] - 2025-12-30

### Fixed
- Web UI: Healthcare example missing `addLog` function definition (JS ReferenceError)
- API Server: Healthcare demo numpy array boolean check error (ValueError)
- Ethereum RPC: Added Ankr (`rpc.ankr.com/eth`) as primary endpoint (no API key)
- Ethereum RPC: Better error handling for hex parsing and empty responses
- VitalDB: Improved numpy array handling for vital signs data
- VitalDB: Proper fallback when library not installed

### Changed
- RPC endpoint priority: Ankr > PublicNode > Cloudflare > DRPC > 1RPC
- Increased RPC timeout from 10s to 15s for reliability
- Added detailed logging for successful RPC connections

### Added
- Technical Report v2.3.4 (18 pages) with comprehensive documentation
  - Multi-platform installation guide (Debian/Ubuntu, Fedora/RHEL, Arch, macOS)
  - liboqs-python build instructions (automatic and manual options)
  - Complete References section with 12 academic citations
  - KeepTogether for tables (prevents page splits)
  - Proper code block formatting with Preformatted style

## [2.3.3] - 2025-12-30

### Fixed
- VitalDB API UTF-8 BOM handling (decode with `utf-8-sig`)
- Etherscan V1 API deprecation: Migrated to public Ethereum RPC endpoints
  - Primary: cloudflare-eth.com
  - Fallback: publicnode.com, llamarpc.com
- No API key required for blockchain data fetching

### Changed
- Blockchain data source now uses standard Ethereum JSON-RPC
- Multiple RPC endpoint fallback for reliability

## [2.3.2] - 2025-12-30

### Fixed
- VitalDB API gzip response handling (`'utf-8' codec can't decode byte 0x8b` error)
- FHE multiply TypeError when using scalar multiplication (removed erroneous relin key)
- Healthcare demo mean calculation (now correctly computes average)

### Changed
- `multiply()` method now auto-detects scalar vs ciphertext multiplication
- Healthcare demo uses proper mean formula: `sum(values * (1/n))`
- Live data fetcher handles gzip-compressed API responses

## [2.3.1] - 2025-12-30

### Added
- Live data fetching from external APIs (VitalDB, Yahoo Finance, UCI, Etherscan)
- `api/live_data_fetcher.py` module for real-time data retrieval
- LIVE/Embedded status indicators in Web UI
- Automatic fallback to embedded samples when APIs unavailable

### Changed
- Enterprise endpoints now attempt live API calls first
- Web UI shows data source status (LIVE green badge when live data)
- Sign/Verify endpoints now properly pass algorithm parameter

### Fixed
- HTTP 500 error in blockchain signing demo (algorithm parameter not passed)
- Sign endpoint tuple unpacking for signature return value
- Verify endpoint algorithm handling

## [2.3.0] - 2025-12-30

### Added
- Real data sources with full academic citations
- Healthcare: VitalDB Open Dataset integration (DOI: 10.1038/s41597-022-01411-5)
- Finance: Yahoo Finance API with real stock prices (yfinance library)
- IoT: UCI ML Repository power consumption dataset (DOI: 10.24432/C52G6F)
- Blockchain: Etherscan API with real Ethereum transaction references
- GET /enterprise/citations endpoint for all data source citations
- Data verification methods with SHA256 integrity checks
- Clinical reference ranges from AHA, WHO, CDC guidelines

### Changed
- Enterprise data module completely rewritten with citation support
- All enterprise endpoints now return data_source and verification fields
- Updated API documentation with data source information
- Technical report updated with academic references

### Fixed
- Proper attribution for all external data sources
- License compliance for CC BY-NC-SA 4.0 (VitalDB) and CC BY 4.0 (UCI)

## [2.2.0] - 2025-12-29

### Added
- Multi-algorithm PQC support (ML-KEM-512/768/1024, ML-DSA-44/65/87)
- Algorithm selection API endpoints
- Enterprise data verification examples
- Algorithm comparison endpoint
- Web UI algorithm selectors

### Changed
- PQCManager supports dynamic algorithm selection
- Updated Web UI with algorithm dropdown menus

## [2.1.2] - 2025-12-29

### Fixed
- ML-DSA-65 signature size validation (3309 bytes)
- Web UI API path corrections
- Technical report table layouts

## [2.1.0] - 2025-12-28

### Added
- RealPQCManager using liboqs-python
- Swagger UI integration guide
- Smart Default feature for API testing
- React-based Web UI with crypto simulations
- Enterprise examples (Healthcare, Finance, IoT, Blockchain)

### Removed
- MockPQCManager (replaced with real implementation)

## [2.0.0] - 2025-12-27

### Added
- FastAPI REST API server
- DESILO FHE integration with GPU support
- WebSocket server for real-time operations
- Comprehensive API documentation

### Changed
- Complete architecture redesign
- Modular component structure

## [1.0.0] - 2025-12-26

### Added
- Initial PQC-FHE integration implementation
- Basic Kyber/Dilithium support
- FHE operations (encrypt, decrypt, add, multiply)
- CLI interface
