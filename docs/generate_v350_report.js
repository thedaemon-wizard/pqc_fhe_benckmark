const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat, TableOfContents
} = require("docx");

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };
const headerColor = "1a365d";
const TW = 9360; // table width in DXA (US Letter with 1" margins)

function headerCell(text, width) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: { fill: headerColor, type: ShadingType.CLEAR },
    margins: cellMargins,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text, bold: true, color: "FFFFFF", font: "Arial", size: 20 })
    ]})]
  });
}

function dataCell(text, width) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    margins: cellMargins,
    children: [new Paragraph({ children: [new TextRun({ text, font: "Arial", size: 20 })] })]
  });
}

function makeTable(headers, data, widths) {
  return new Table({
    width: { size: TW, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ children: headers.map((h, i) => headerCell(h, widths[i])) }),
      ...data.map(row => new TableRow({
        children: row.map((cell, i) => dataCell(cell, widths[i]))
      }))
    ]
  });
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 200 }, children: [
    new TextRun({ text, bold: true, font: "Arial", size: 32, color: "2B6CB0" })
  ]});
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 160 }, children: [
    new TextRun({ text, bold: true, font: "Arial", size: 26, color: "2B6CB0" })
  ]});
}
function p(text) {
  return new Paragraph({ spacing: { after: 120 }, children: [
    new TextRun({ text, font: "Arial", size: 22 })
  ]});
}
function pb() { return new Paragraph({ children: [new PageBreak()] }); }

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: "2B6CB0" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "2B6CB0" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
    }]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
        new TextRun({ text: "PQC-FHE Technical Report v3.5.0", font: "Arial", size: 18, color: "888888" })
      ]})] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
        new TextRun({ text: "Page ", font: "Arial", size: 18, color: "888888" }),
        new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: "888888" })
      ]})] })
    },
    children: [
      // === TITLE PAGE ===
      new Paragraph({ spacing: { before: 2400 }, children: [
        new TextRun({ text: "PQC-FHE Integration Platform", font: "Arial", size: 56, color: "1a365d" })
      ]}),
      new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "2B6CB0", space: 1 } },
        spacing: { after: 200 }, children: [] }),
      new Paragraph({ spacing: { before: 200, after: 100 }, children: [
        new TextRun({ text: "Technical Report v3.5.0", font: "Arial", size: 32, color: "2B6CB0", bold: true })
      ]}),
      new Paragraph({ spacing: { after: 60 }, children: [
        new TextRun({ text: "Codename: Accurate Hardware Discovery", font: "Arial", size: 22 })
      ]}),
      p("Release Date: 2026-03-29"),
      p("Classification: Enterprise Technical Report"),
      p("Author: PQC-FHE Integration Library"),
      p("Based on: 2026 Q1 Latest Post-Quantum Cryptography Research"),
      pb(),

      // === TABLE OF CONTENTS ===
      new Paragraph({ spacing: { before: 200, after: 200 }, children: [
        new TextRun({ text: "Table of Contents", font: "Arial", size: 32, bold: true, color: "2B6CB0" })
      ]}),
      new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
      pb(),

      // === 1. EXECUTIVE SUMMARY ===
      h1("1. Executive Summary"),
      p("PQC-FHE Integration Platform v3.5.0 is an accuracy-focused release that corrects IBM Quantum hardware data and adds benchmark result persistence. The platform now correctly distinguishes between IBM Heron r1 (ibm_torino, 133 qubits, Dec 2023) and Heron r2 (ibm_fez/kingston/marrakesh, 156 qubits, Jul 2024), expanding KNOWN_PROCESSORS from 3 to 6 backends."),
      p("This release also introduces BenchmarkResultsManager, which automatically saves benchmark results as timestamped JSON files and circuit diagrams as PNG files for later retrieval. Four new API endpoints provide access to saved results and Prometheus-compatible metrics, and Button 5 (All-Sector Circuit Comparison) now displays circuit diagrams alongside benchmark data."),
      p("The 2026 PQC research landscape section has been updated with NIST IR 8547 migration timeline (2030 deprecated, 2035 disallowed), HQC selection as the 5th NIST PQC algorithm (March 2025), and compressed CRQC estimates showing RSA-2048 attacks may require as few as 100K physical qubits using QLDPC codes."),
      p("Infrastructure verification has been completed: Docker image (420MB) builds successfully with liboqs 0.14.0, Docker Compose monitoring stack (API + Prometheus + Grafana) runs end-to-end, Prometheus scrapes /metrics endpoint (UP status, 1ms), Grafana connects to Prometheus data source, and Helm chart validates with lint and template rendering (8 Kubernetes manifests)."),

      h2("Key Changes Summary"),
      makeTable(
        ["Category", "Change", "Impact"],
        [
          ["IBM QPU Data", "ibm_torino: Heron R2 -> Heron r1", "Correct processor type (133Q)"],
          ["New Backends", "ibm_fez/kingston/marrakesh added", "3 Heron r2 backends (156Q each)"],
          ["HERON_R2_FALLBACK", "ibm_torino -> ibm_fez", "Correct fallback reference"],
          ["Result Persistence", "BenchmarkResultsManager", "JSON/PNG file storage"],
          ["New API Endpoints", "4 endpoints added (90 total routes)", "Results/diagrams/metrics"],
          ["Prometheus /metrics", "Zero-dependency exposition format", "HTTP stats, memory, uptime, GC"],
          ["Docker Image", "420MB multi-stage build", "liboqs 0.14.0, python:3.12-slim"],
          ["Docker Compose", "API + Prometheus + Grafana", "Monitoring stack verified"],
          ["Helm Chart", "8 manifests validated", "Deployment, Service, Ingress, HPA, RBAC"],
          ["Circuit Diagrams", "Button 5 shows diagrams", "All-Sector visualization"],
          ["NIST IR 8547", "2030/2035 timeline added", "Migration urgency context"],
          ["Test Coverage", "221 -> 238 tests", "17 new tests, all passing"],
        ],
        [2500, 3500, 3360]
      ),

      h2("Key Metrics"),
      makeTable(
        ["Metric", "Value"],
        [
          ["Total Test Cases", "238 (17 new in v3.5.0)"],
          ["KNOWN_PROCESSORS", "6 backends (3 new Heron r2)"],
          ["API Routes", "90 total (4 new: results, diagrams, metrics)"],
          ["Docker Image Size", "420MB (multi-stage build)"],
          ["Prometheus Metrics", "/metrics — uptime, memory, HTTP stats, GC, app info"],
          ["Helm Manifests", "8 (Deployment, Service, Ingress, HPA, NetworkPolicy, RBAC)"],
          ["Circuit Diagram Storage", "data/circuit_diagrams/ (auto-saved PNG)"],
          ["Benchmark Result Storage", "data/benchmark_results/ (auto-saved JSON)"],
          ["Academic References", "40+ citations"],
          ["IBM QPU Processors Supported", "Heron r1, Heron r2, Eagle r3, Nighthawk r1"],
        ],
        [4680, 4680]
      ),
      pb(),

      // === 2. IBM QPU HARDWARE CORRECTION ===
      h1("2. IBM Quantum Hardware Data Correction"),
      p("The previous versions (v3.3.0-v3.4.0) incorrectly classified ibm_torino as a Heron R2 processor with 156 qubits. Web research confirms that ibm_torino is actually a Heron r1 processor with 133 qubits, released in December 2023. The true Heron r2 processors (156 qubits, released July 2024) are ibm_fez, ibm_kingston, and ibm_marrakesh."),

      h2("2.1 Processor Type Corrections"),
      makeTable(
        ["Backend", "Previous (Incorrect)", "Corrected (v3.5.0)", "Source"],
        [
          ["ibm_torino", "Heron R2, 156Q", "Heron r1, 133Q", "IBM Quantum Platform"],
          ["ibm_fez", "(not listed)", "Heron r2, 156Q", "IBM Quantum Platform"],
          ["ibm_kingston", "(not listed)", "Heron r2, 156Q", "IBM Open Plan 2026"],
          ["ibm_marrakesh", "(not listed)", "Heron r2, 156Q", "IBM Quantum Platform"],
          ["ibm_brisbane", "Eagle r3, 127Q", "Eagle r3, 127Q (unchanged)", "IBM Quantum Docs"],
          ["ibm_sherbrooke", "Eagle r3, 127Q", "Eagle r3, 127Q (unchanged)", "IBM Quantum Docs"],
        ],
        [2000, 2200, 2800, 2360]
      ),

      h2("2.2 Noise Parameter Corrections"),
      p("The ibm_torino noise parameters have been updated to reflect Heron r1 specifications, which have shorter coherence times and higher error rates compared to Heron r2:"),
      makeTable(
        ["Parameter", "Previous (r2 specs)", "Corrected (r1 specs)", "Heron r2 (ibm_fez)"],
        [
          ["T1 (us)", "250.0", "160.0", "250.0"],
          ["T2 (us)", "150.0", "100.0", "150.0"],
          ["1Q Error", "2.4e-4", "3.0e-4", "2.4e-4"],
          ["2Q Error", "3.8e-3", "5.0e-3", "3.8e-3"],
          ["Readout Error", "1.2e-2", "1.5e-2", "1.2e-2"],
          ["Num Qubits", "156", "133", "156"],
        ],
        [2000, 2200, 2500, 2660]
      ),

      h2("2.3 Dynamic Discovery Flow"),
      p("The existing 3-tier fallback chain remains unchanged but now uses corrected data at each level:"),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [
        new TextRun({ text: "Tier 1 (API): QiskitRuntimeService fetches real-time processor_type, num_qubits, T1/T2, gate errors via .env token", font: "Arial", size: 22 })
      ]}),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [
        new TextRun({ text: "Tier 2 (JSON Cache): data/ibm_backends_cache.json stores last successful API response (regenerated with 6 backends)", font: "Arial", size: 22 })
      ]}),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [
        new TextRun({ text: "Tier 3 (Hardcoded): KNOWN_PROCESSORS expanded from 3 to 6 entries with corrected specs", font: "Arial", size: 22 })
      ]}),

      h2("2.4 HERON_R2_FALLBACK Correction"),
      p("The HERON_R2_FALLBACK constant (used when unknown backend names are requested) has been corrected from ibm_torino to ibm_fez. This ensures that fallback noise parameters use actual Heron r2 specifications rather than r1 specs incorrectly labeled as r2."),
      pb(),

      // === 3. BENCHMARK RESULTS PERSISTENCE ===
      h1("3. Benchmark Results Persistence"),
      p("v3.5.0 introduces BenchmarkResultsManager, a new class that automatically saves benchmark results and circuit diagrams to disk for later retrieval and display in the Web UI."),

      h2("3.1 BenchmarkResultsManager"),
      p("The manager provides persistent storage with the following methods:"),
      makeTable(
        ["Method", "Description", "Storage"],
        [
          ["save_benchmark_result()", "Save benchmark JSON with timestamp", "data/benchmark_results/"],
          ["save_circuit_diagram_png()", "Save base64 diagram as PNG", "data/circuit_diagrams/"],
          ["list_results(limit)", "List saved results (newest first)", "metadata only"],
          ["get_result(filename)", "Load specific result (path-traversal safe)", "full JSON"],
          ["list_diagrams()", "List saved PNG diagrams", "metadata only"],
          ["ensure_dirs()", "Create directories if missing", "called at startup"],
        ],
        [3000, 3800, 2560]
      ),

      h2("3.2 Automatic Result Saving"),
      p("Both run_sector_circuit_benchmark() and run_all_sectors() now automatically save their results to timestamped JSON files in data/benchmark_results/. Circuit diagram endpoints (shor-diagram, grover-diagram, ecc-diagram) also persist generated PNG files to data/circuit_diagrams/."),

      h2("3.3 New API Endpoints"),
      makeTable(
        ["Endpoint", "Method", "Description"],
        [
          ["/benchmarks/results", "GET", "List saved benchmark result files (newest first)"],
          ["/benchmarks/results/{filename}", "GET", "Retrieve specific result (path-traversal protection)"],
          ["/benchmarks/diagrams", "GET", "List saved circuit diagram PNG files"],
        ],
        [3500, 1200, 4660]
      ),

      h2("3.4 Web UI Updates"),
      p("Button 5 (All-Sector Circuit Comparison) now calls fetchCircuitDiagrams() after benchmark execution, displaying Shor, Grover, and ECC circuit diagrams in a 3-column grid layout identical to Button 4. The noise backend dropdown label has been corrected from 'IBM Heron R2 (156Q, Fallback Profile)' to 'IBM Heron r2 (156Q, ibm_fez Specs)'."),
      pb(),

      // === 4. 2026 PQC RESEARCH LANDSCAPE ===
      h1("4. 2026 PQC Research Landscape Update"),
      p("The following research developments from 2025-2026 have been incorporated into the platform documentation and validation assessments:"),

      h2("4.1 NIST IR 8547 Migration Timeline"),
      p("NIST Information Report 8547 (November 2024) establishes a binding timeline for PQC migration across US federal systems:"),
      makeTable(
        ["Year", "Status", "Requirement"],
        [
          ["2024-2029", "Active Migration", "Begin PQC evaluation, planning, and hybrid deployment"],
          ["2030", "Deprecated", "RSA, ECDSA, ECDH, DSA, DH deprecated; PQC preferred"],
          ["2035", "Disallowed", "Classical-only algorithms prohibited in federal systems"],
        ],
        [2000, 2500, 4860]
      ),

      h2("4.2 HQC: 5th NIST PQC Algorithm"),
      p("In March 2025, NIST selected HQC (Hamming Quasi-Cyclic) as the 5th PQC standardized algorithm. HQC is a code-based KEM that provides diversification against the lattice monoculture risk inherent in ML-KEM, ML-DSA, FN-DSA, and CKKS (all lattice-based). Standardization is expected by 2027."),

      h2("4.3 CRQC Estimate Compression"),
      p("The estimated physical qubit count required for a Cryptographically Relevant Quantum Computer (CRQC) to break RSA-2048 has been dramatically compressed over the past 5 years:"),
      makeTable(
        ["Year", "Estimate", "Technology", "Source"],
        [
          ["2021", "~20M physical qubits", "Surface codes", "Gidney & Ekera (2021)"],
          ["2025", "~1M physical qubits", "Magic state cultivation", "Gidney (May 2025)"],
          ["2026", "~100K physical qubits", "QLDPC codes", "Iceberg Quantum (Feb 2026)"],
        ],
        [1500, 2500, 2500, 2860]
      ),

      h2("4.4 Hybrid TLS Default Deployment"),
      p("ML-KEM + X25519 hybrid key exchange is now deployed by default across major platforms: Google Chrome, Mozilla Firefox, Cloudflare, and Akamai (as of February 2026). This validates the hybrid migration strategy implemented in the platform."),

      h2("4.5 AI-Assisted Side-Channel Attacks"),
      p("Research published in 2026 demonstrates single-trace key recovery attacks on ML-KEM implementations using machine learning techniques. This underscores the critical importance of constant-time implementations and highlights that PQC algorithm security alone is insufficient without side-channel countermeasures."),

      h2("4.6 MOZAIK: MPC+FHE IoT Platform"),
      p("MOZAIK (January 2026) is an open-source platform combining MPC and FHE for privacy-preserving machine learning on IoT devices. This validates the MPC-HE sector benchmarks implemented in the platform, which simulate similar privacy-preserving computation scenarios."),
      pb(),

      // === 5. PLATFORM VALIDATION ===
      h1("5. Platform Validation and Limitations"),

      h2("5.1 Small-Scale Circuit Validity"),
      p("The platform uses 8-24 qubit circuits for Shor, Grover, and ECC demonstrations. This approach is standard across the quantum computing research community. IBM Quantum Learning tutorials use N=15 factoring (8 qubits) for Shor, Google Cirq tutorials use 4-8 qubit Grover search, and all major academic papers use small-scale demonstrations with extrapolation models."),

      h2("5.2 Extrapolation Model Sources"),
      makeTable(
        ["Algorithm", "Demo Scale", "Full-Scale Estimate", "Source"],
        [
          ["Shor vs RSA-2048", "N=15,21,35 (8-12Q)", "~4,000+ logical qubits", "Gidney 2021/2025"],
          ["Grover vs AES-128", "4-16 qubits", "2,953 logical qubits", "Grassl et al. 2016"],
          ["ECC DLog P-256", "GF(2^4) (12Q)", "2,330 logical qubits", "Roetteler et al. 2017"],
        ],
        [2500, 2200, 2500, 2160]
      ),

      h2("5.3 Known Limitations"),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [
        new TextRun({ text: "Scale gap: RSA-2048 requires ~4,000+ logical qubits; max physical qubits available is 156 (no error correction)", font: "Arial", size: 22 })
      ]}),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [
        new TextRun({ text: "Extrapolation uncertainty: Resource estimates depend on theoretical models; implementation overhead not included", font: "Arial", size: 22 })
      ]}),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [
        new TextRun({ text: "Lattice monoculture: ML-KEM, ML-DSA, FN-DSA, and CKKS all rely on lattice assumptions; HQC mitigates but is not yet standardized", font: "Arial", size: 22 })
      ]}),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [
        new TextRun({ text: "Side-channel: Platform evaluates risks but does not implement hardware-level countermeasures", font: "Arial", size: 22 })
      ]}),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [
        new TextRun({ text: "AI-assisted attacks: ML-KEM single-trace key recovery demonstrated in 2026; constant-time implementation is critical", font: "Arial", size: 22 })
      ]}),
      pb(),

      // === 6. INFRASTRUCTURE & MONITORING ===
      h1("6. Infrastructure & Monitoring Verification"),
      p("v3.5.0 includes comprehensive infrastructure verification covering Docker containerization, Prometheus monitoring, Grafana visualization, and Kubernetes orchestration via Helm charts."),

      h2("6.1 Docker Containerization"),
      makeTable(
        ["Component", "Detail", "Status"],
        [
          ["Docker Image", "pqc-fhe-api:v3.5.0 (420MB)", "Build successful"],
          ["Base Image", "python:3.12-slim (multi-stage)", "OK"],
          ["liboqs Build", "v0.14.0 with pkg-config + OpenSSL", "OK"],
          ["Health Check", "/health endpoint, 30s interval", "OK"],
          ["Non-root User", "appuser (security best practice)", "OK"],
          ["GPU Dockerfile", "Dockerfile.gpu (CUDA 12.2)", "Available"],
        ],
        [3120, 3120, 3120]
      ),

      h2("6.2 Prometheus Monitoring"),
      p("A zero-dependency /metrics endpoint was added to api/server.py using the Prometheus exposition format (text/plain; version=0.0.4). The endpoint uses thread-safe HTTP middleware to track request statistics without requiring the prometheus_client library."),
      makeTable(
        ["Metric", "Type", "Description"],
        [
          ["process_uptime_seconds", "gauge", "Time since server started"],
          ["process_resident_memory_bytes", "gauge", "RSS memory usage"],
          ["python_gc_objects_collected", "gauge", "GC collected objects per generation"],
          ["http_requests_total", "counter", "Total HTTP requests (excluding /metrics)"],
          ["http_requests_by_status_total", "counter", "Requests by HTTP status code"],
          ["http_requests_by_method_total", "counter", "Requests by HTTP method"],
          ["http_request_duration_seconds_sum", "counter", "Total request processing time"],
          ["pqc_fhe_ciphertexts_stored", "gauge", "Current stored ciphertexts"],
          ["pqc_fhe_info", "info", "Application version and metadata"],
        ],
        [3120, 1560, 4680]
      ),

      h2("6.3 Docker Compose Monitoring Stack"),
      p("The docker-compose.yml defines a monitoring profile that runs API, Prometheus, and Grafana on a shared Docker bridge network (pqc-fhe-network). Prometheus scrapes the API /metrics endpoint via Docker internal DNS (api:8000). Grafana connects to Prometheus via pqc-fhe-prometheus:9090."),
      makeTable(
        ["Service", "Port", "Network", "Verification"],
        [
          ["API (pqc-fhe-api)", "8000", "pqc-fhe-network", "Health: healthy, /metrics: UP"],
          ["Prometheus", "9090", "pqc-fhe-network", "Target UP, 1ms scrape time"],
          ["Grafana", "${GRAFANA_PORT:-3000}", "pqc-fhe-network", "Data source connected, dashboard created"],
        ],
        [2340, 2340, 2340, 2340]
      ),

      h2("6.4 Kubernetes Helm Chart"),
      p("The Helm chart (kubernetes/helm/pqc-fhe/) passed lint validation and template rendering, producing 8 Kubernetes manifests:"),
      makeTable(
        ["Manifest", "Kind", "Key Configuration"],
        [
          ["Deployment", "apps/v1", "GPU resources, livenessProbe, readinessProbe"],
          ["Service", "v1", "ClusterIP, port 8000"],
          ["Ingress", "networking.k8s.io/v1", "TLS, path-based routing"],
          ["HPA", "autoscaling/v2", "CPU/Memory targets, min/max replicas"],
          ["NetworkPolicy", "networking.k8s.io/v1", "Ingress/Egress rules"],
          ["ServiceAccount", "v1", "Dedicated service account"],
          ["Role", "rbac.authorization.k8s.io/v1", "ConfigMap/Secret read access"],
          ["RoleBinding", "rbac.authorization.k8s.io/v1", "Role to ServiceAccount binding"],
        ],
        [2340, 3510, 3510]
      ),
      pb(),

      // === 7. TEST COVERAGE ===
      h1("7. Test Coverage (v3.5.0)"),
      p("v3.5.0 adds 17 new tests, bringing the total from 221 to 238. All tests pass (0 failures, 80 deprecation warnings from qiskit-ibm-runtime IBMFractionalTranslationPlugin):"),

      h2("7.1 TestIBMQuantumV350 (10 tests)"),
      makeTable(
        ["Test", "Verification"],
        [
          ["test_ibm_torino_is_heron_r1", "ibm_torino = Heron r1, 133Q"],
          ["test_ibm_fez_is_heron_r2", "ibm_fez = Heron r2, 156Q, CZ gates"],
          ["test_ibm_kingston_is_heron_r2", "ibm_kingston = Heron r2, 156Q"],
          ["test_ibm_marrakesh_is_heron_r2", "ibm_marrakesh = Heron r2, 156Q"],
          ["test_heron_r1_inferior_to_r2", "r1 T1 < r2 T1, r1 errors > r2 errors"],
          ["test_heron_r2_fallback_is_ibm_fez", "HERON_R2_FALLBACK.name = ibm_fez"],
          ["test_known_processors_count", "6 backends total"],
          ["test_all_heron_backends_have_cz", "All 4 Heron processors use CZ"],
          ["test_list_backends_includes_new", "fez/kingston/marrakesh in list"],
          ["test_noise_params_for_new_backends", "T1=250, num_qubits=156 for all r2"],
        ],
        [4680, 4680]
      ),

      h2("7.2 TestBenchmarkResultsSaving (7 tests)"),
      makeTable(
        ["Test", "Verification"],
        [
          ["test_ensure_dirs_creates_directories", "RESULTS_DIR and DIAGRAMS_DIR created"],
          ["test_save_and_load_result", "JSON round-trip correctness"],
          ["test_list_results", "Metadata list format correct"],
          ["test_path_traversal_prevention", "../../../etc/passwd blocked"],
          ["test_save_circuit_diagram_png", "PNG file written from base64 data URI"],
          ["test_save_circuit_diagram_invalid_uri", "None returned for invalid input"],
          ["test_list_diagrams", "Diagram metadata list format correct"],
        ],
        [4680, 4680]
      ),
      pb(),

      // === 8. VERSION HISTORY ===
      h1("8. Version History"),
      makeTable(
        ["Version", "Date", "Codename", "Key Features"],
        [
          ["3.5.0", "2026-03-29", "Accurate Hardware Discovery", "IBM QPU correction, BenchmarkResultsManager, Prometheus /metrics, Docker 420MB, Helm validated, 238 tests"],
          ["3.4.0", "2026-03-29", "Dynamic QPU Discovery", "3-tier fallback, singleton, JSON cache, least_busy"],
          ["3.3.0", "2026-03-28", "IBM QPU Noise", "IBM Quantum noise integration, FHE bootstrap optimization"],
          ["3.2.0", "2026-03-19", "Research Accuracy", "BKZ fix, GPU acceleration, circuit visualization, 25 endpoints"],
          ["3.1.0", "2026-03-18", "Circuit Verification", "Qiskit AerSimulator, sector benchmarks"],
          ["3.0.0", "2026-03-18", "Quantum Threat", "Shor/Grover simulator, security scoring, MPC-HE"],
        ],
        [1200, 1600, 2800, 3760]
      ),
      pb(),

      // === 9. REFERENCES ===
      h1("9. References"),
      ...[
        "[1] NIST FIPS 203: ML-KEM Standard (August 2024)",
        "[2] NIST FIPS 204: ML-DSA Standard (August 2024)",
        "[3] NIST FIPS 205: SLH-DSA Standard (August 2024)",
        "[4] NIST IR 8547: Transition to Post-Quantum Cryptography Standards (November 2024)",
        "[5] NIST IR 8545: HQC Selection as 5th PQC Algorithm (March 2025)",
        "[6] Gidney & Ekera (2021): How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits",
        "[7] Gidney (May 2025): Magic state cultivation - RSA-2048 with ~1M physical qubits",
        "[8] Pinnacle/Iceberg Quantum (Feb 2026): QLDPC codes - RSA-2048 with ~100K physical qubits",
        "[9] IBM Quantum Docs: Processor Types - https://quantum.cloud.ibm.com/docs/guides/processor-types",
        "[10] IBM Quantum Open Plan Updates (March 2026): Heron r3, Flamingo, ibm_kingston specs",
        "[11] IBM Quantum Learning: Shor's Algorithm - https://quantum.cloud.ibm.com/learning/ja/modules/computer-science/shors-algorithm",
        "[12] IBM Quantum Learning: Grover's Algorithm - https://quantum.cloud.ibm.com/learning/ja/modules/computer-science/grovers",
        "[13] Roetteler et al. (2017): Quantum resource estimates for computing elliptic curve discrete logarithms",
        "[14] Grassl et al. (2016): Applying Grover's algorithm to AES",
        "[15] MOZAIK (Jan 2026): Open-source MPC+FHE IoT Platform",
        "[16] Security Boulevard: Enterprise PQC Migration Guide 2026",
        "[17] Microsoft Majorana 1 (Feb 2025): Topological qubits for quantum computing",
        "[18] MDPI: HNDL Temporal Cybersecurity Risk Model - Mosca theorem formalization",
        "[19] Dutch team / van Hoof et al. (Oct 2025): Quantum sieve exponent 0.257",
        "[20] Berzati et al. (CHES 2025): ML-KEM SPA key recovery in 30 seconds",
        "[21] Prometheus Exposition Format: https://prometheus.io/docs/instrumenting/exposition_formats/",
        "[22] Docker Multi-Stage Builds: https://docs.docker.com/build/building/multi-stage/",
        "[23] Helm Chart Best Practices: https://helm.sh/docs/chart_best_practices/",
        "[24] IBM Quantum Docs - Build Noise Models: https://quantum.cloud.ibm.com/docs/guides/build-noise-models",
        "[25] IBM Quantum Docs - QPU Information: https://quantum.cloud.ibm.com/docs/guides/qpu-information",
      ].map(ref => p(ref)),
    ]
  }]
});

const outPath = "/home/a-koike/.claude-worktrees/pqc_fhe_benckmark/goofy-stonebraker/docs/PQC_FHE_Technical_Report_v3.5.0_Enterprise.docx";
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outPath, buffer);
  console.log("Report generated:", outPath);
  console.log("Size:", (buffer.length / 1024).toFixed(1), "KB");
});
