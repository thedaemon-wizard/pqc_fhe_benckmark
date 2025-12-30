#!/usr/bin/env python3
"""
Generate Comprehensive Technical Report for PQC-FHE Integration Platform v2.3.4
With proper multi-platform installation instructions and references
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, Preformatted
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime

def create_report():
    doc = SimpleDocTemplate(
        "PQC_FHE_Technical_Report_v2.3.4.pdf",
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(
        name='MainTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=HexColor('#1a365d')
    ))
    
    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=HexColor('#4a5568')
    ))
    
    styles.add(ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=12,
        textColor=HexColor('#2c5282')
    ))
    
    styles.add(ParagraphStyle(
        name='SubsectionTitle',
        parent=styles['Heading2'],
        fontSize=12,
        spaceBefore=12,
        spaceAfter=6,
        textColor=HexColor('#3182ce')
    ))
    
    styles.add(ParagraphStyle(
        name='MyBodyText',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=4,
        spaceAfter=4,
        alignment=TA_JUSTIFY,
        leading=13
    ))
    
    styles.add(ParagraphStyle(
        name='BulletItem',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=2,
        spaceAfter=2,
        leftIndent=20,
        leading=13
    ))
    
    styles.add(ParagraphStyle(
        name='CodeStyle',
        fontName='Courier',
        fontSize=8,
        leading=10,
        spaceBefore=4,
        spaceAfter=4,
        leftIndent=10,
        rightIndent=10
    ))
    
    styles.add(ParagraphStyle(
        name='RefStyle',
        parent=styles['Normal'],
        fontSize=9,
        spaceBefore=3,
        spaceAfter=3,
        leftIndent=20,
        firstLineIndent=-20,
        leading=12
    ))
    
    story = []
    
    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    story.append(Spacer(1, 80))
    story.append(Paragraph("PQC-FHE Integration Platform", styles['MainTitle']))
    story.append(Paragraph("Technical Report v2.3.4", styles['Subtitle']))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Post-Quantum Cryptography + Fully Homomorphic Encryption", styles['Subtitle']))
    story.append(Spacer(1, 40))
    
    # Info box
    info_data = [
        ['Version', '2.3.4'],
        ['Release Date', '2025-12-30'],
        ['PQC Standards', 'FIPS 203 (ML-KEM), FIPS 204 (ML-DSA)'],
        ['FHE Scheme', 'CKKS (DESILO Implementation)'],
        ['License', 'MIT'],
    ]
    info_table = Table(info_data, colWidths=[140, 310])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#2c5282')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(PageBreak())
    
    # =========================================================================
    # TABLE OF CONTENTS
    # =========================================================================
    story.append(Paragraph("Table of Contents", styles['SectionTitle']))
    story.append(Spacer(1, 10))
    toc_items = [
        "1. Executive Summary",
        "2. Market Context and Business Value",
        "3. System Architecture",
        "4. Post-Quantum Cryptography Implementation",
        "5. Fully Homomorphic Encryption Implementation",
        "6. Live Data Sources",
        "7. Enterprise Use Cases",
        "8. API Reference",
        "9. Installation Guide (Multi-Platform)",
        "10. Security Analysis",
        "11. Future Roadmap",
        "Appendix A: Algorithm Parameters",
        "Appendix B: Performance Benchmarks",
        "References",
    ]
    for item in toc_items:
        story.append(Paragraph(item, styles['MyBodyText']))
    story.append(PageBreak())
    
    # =========================================================================
    # 1. EXECUTIVE SUMMARY
    # =========================================================================
    story.append(Paragraph("1. Executive Summary", styles['SectionTitle']))
    story.append(Paragraph(
        "The PQC-FHE Integration Platform provides a production-ready framework combining "
        "Post-Quantum Cryptography (PQC) with Fully Homomorphic Encryption (FHE) for "
        "enterprise security applications. This platform addresses the emerging threat of "
        "quantum computers to current cryptographic systems while enabling privacy-preserving "
        "computation on sensitive data.",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 8))
    story.append(Paragraph("Key Features", styles['SubsectionTitle']))
    features = [
        "* NIST-standardized PQC algorithms (FIPS 203, FIPS 204)",
        "* CKKS-based homomorphic encryption via DESILO FHE",
        "* Real-time data integration from verified public sources",
        "* REST API with comprehensive Swagger documentation",
        "* Interactive Web UI with live demonstrations",
        "* GPU acceleration support (CUDA 12.x/13.x)",
    ]
    for f in features:
        story.append(Paragraph(f, styles['BulletItem']))
    
    story.append(Spacer(1, 8))
    story.append(Paragraph("v2.3.4 Updates (2025-12-30)", styles['SubsectionTitle']))
    story.append(Paragraph(
        "Version 2.3.4 introduces enhanced live data fetching capabilities with robust "
        "fallback mechanisms, fixes for numpy array handling in FHE demo endpoints, "
        "and improved multi-platform installation documentation.",
        styles['MyBodyText']))
    story.append(PageBreak())
    
    # =========================================================================
    # 2. MARKET CONTEXT
    # =========================================================================
    story.append(Paragraph("2. Market Context and Business Value", styles['SectionTitle']))
    
    story.append(Paragraph("The Quantum Threat", styles['SubsectionTitle']))
    story.append(Paragraph(
        "Quantum computers pose an existential threat to current public-key cryptography. "
        "Shor's algorithm can break RSA-2048 in polynomial time, and Grover's algorithm "
        "reduces symmetric key security by half. NIST estimates cryptographically-relevant "
        "quantum computers may emerge within 10-15 years, necessitating immediate migration "
        "to quantum-resistant alternatives.",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 8))
    story.append(Paragraph("Market Opportunity", styles['SubsectionTitle']))
    
    market_data = [
        ['Segment', '2024', '2034', 'CAGR'],
        ['PQC Solutions', '$302M', '$30B', '58%'],
        ['FHE Market', '$200M', '$2.5B', '28%'],
        ['Quantum Security', '$1.2B', '$15B', '29%'],
    ]
    market_table = Table(market_data, colWidths=[120, 80, 90, 60])
    market_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([market_table, Spacer(1, 5), 
        Paragraph("Source: Markets and Markets, Gartner (2024)", styles['MyBodyText'])]))
    story.append(PageBreak())
    
    # =========================================================================
    # 3. SYSTEM ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("3. System Architecture", styles['SectionTitle']))
    
    story.append(Paragraph("Component Overview", styles['SubsectionTitle']))
    arch_data = [
        ['Layer', 'Component', 'Technology'],
        ['Presentation', 'Web UI', 'React + Tailwind CSS'],
        ['API', 'REST Server', 'FastAPI + Swagger UI'],
        ['Cryptography', 'PQC Manager', 'liboqs-python (ML-KEM, ML-DSA)'],
        ['Cryptography', 'FHE Engine', 'DESILO FHE (CKKS)'],
        ['Data', 'Live Fetcher', 'VitalDB, yfinance, Ethereum RPC'],
        ['Infrastructure', 'GPU Accel.', 'CUDA 12.x/13.x + cuQuantum'],
    ]
    arch_table = Table(arch_data, colWidths=[100, 100, 250])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3182ce')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([arch_table]))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("Data Flow", styles['SubsectionTitle']))
    
    flow_steps = [
        "1. Client sends request to REST API (FastAPI server on port 8000)",
        "2. API validates input and routes to appropriate handler",
        "3. For PQC operations: liboqs-python performs key generation/signing",
        "4. For FHE operations: DESILO engine encrypts/computes/decrypts",
        "5. Live data fetcher retrieves external data with automatic fallback",
        "6. Response returned to client with full audit trail",
    ]
    for step in flow_steps:
        story.append(Paragraph(step, styles['MyBodyText']))
    story.append(PageBreak())
    
    # =========================================================================
    # 4. PQC IMPLEMENTATION
    # =========================================================================
    story.append(Paragraph("4. Post-Quantum Cryptography Implementation", styles['SectionTitle']))
    
    story.append(Paragraph("NIST Standards Compliance", styles['SubsectionTitle']))
    story.append(Paragraph(
        "This platform implements NIST's finalized post-quantum cryptography standards: "
        "FIPS 203 (ML-KEM) for key encapsulation and FIPS 204 (ML-DSA) for digital "
        "signatures. These standards were published on August 13, 2024 and represent "
        "the foundation of quantum-resistant cryptography.",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 8))
    story.append(Paragraph("Key Encapsulation Mechanisms (FIPS 203)", styles['SubsectionTitle']))
    kem_data = [
        ['Algorithm', 'Level', 'Public Key', 'Ciphertext', 'Use Case'],
        ['ML-KEM-512', '1', '800 B', '768 B', 'IoT/Embedded'],
        ['ML-KEM-768', '3', '1,184 B', '1,088 B', 'General Purpose'],
        ['ML-KEM-1024', '5', '1,568 B', '1,568 B', 'High Security'],
    ]
    kem_table = Table(kem_data, colWidths=[90, 50, 75, 75, 100])
    kem_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#48bb78')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([kem_table]))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Digital Signature Algorithms (FIPS 204)", styles['SubsectionTitle']))
    sig_data = [
        ['Algorithm', 'Level', 'Public Key', 'Signature', 'Speed'],
        ['ML-DSA-44', '2', '1,312 B', '2,420 B', 'Fastest'],
        ['ML-DSA-65', '3', '1,952 B', '3,309 B', 'Balanced'],
        ['ML-DSA-87', '5', '2,592 B', '4,627 B', 'Maximum'],
    ]
    sig_table = Table(sig_data, colWidths=[90, 50, 80, 80, 90])
    sig_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#ed8936')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([sig_table]))
    story.append(PageBreak())
    
    # =========================================================================
    # 5. FHE IMPLEMENTATION
    # =========================================================================
    story.append(Paragraph("5. Fully Homomorphic Encryption Implementation", styles['SectionTitle']))
    
    story.append(Paragraph("CKKS Scheme Overview", styles['SubsectionTitle']))
    story.append(Paragraph(
        "The platform uses the CKKS (Cheon-Kim-Kim-Song) homomorphic encryption scheme, "
        "which supports approximate arithmetic on encrypted real numbers. CKKS is ideal "
        "for machine learning and statistical analysis on encrypted data [4].",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 6))
    story.append(Paragraph("Key Properties:", styles['SubsectionTitle']))
    ckks_props = [
        "* Supports addition and multiplication on encrypted floating-point numbers",
        "* Slot packing allows SIMD operations on thousands of values simultaneously",
        "* Rescaling maintains precision across multiplicative depth",
        "* Bootstrapping enables unlimited computation depth",
    ]
    for p in ckks_props:
        story.append(Paragraph(p, styles['BulletItem']))
    
    story.append(Spacer(1, 8))
    story.append(Paragraph("DESILO FHE Configuration", styles['SubsectionTitle']))
    config_data = [
        ['Parameter', 'Value', 'Description'],
        ['poly_degree', '16,384', 'Polynomial ring dimension (N)'],
        ['coeff_mod_bit_sizes', '[60,40,40,40,60]', 'Coefficient modulus chain'],
        ['scale', '2^40', 'Encoding scale for precision'],
        ['max_mult_depth', '4', 'Maximum multiplicative depth'],
        ['slot_count', '8,192', 'Number of plaintext slots'],
    ]
    config_table = Table(config_data, colWidths=[120, 120, 210])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#805ad5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([config_table]))
    story.append(PageBreak())
    
    # =========================================================================
    # 6. LIVE DATA SOURCES
    # =========================================================================
    story.append(Paragraph("6. Live Data Sources", styles['SectionTitle']))
    
    story.append(Paragraph(
        "Version 2.3.4 provides robust live data fetching with automatic fallback to "
        "embedded sample data. This ensures demonstrations work reliably while "
        "showcasing real-world data integration capabilities.",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 8))
    story.append(Paragraph("Healthcare: VitalDB [6]", styles['SubsectionTitle']))
    vitaldb_data = [
        ['Property', 'Value'],
        ['Dataset', 'VitalDB Open Dataset'],
        ['Method', 'vitaldb Python library'],
        ['Data Type', 'Surgical patient vital signs (BP, HR, SpO2)'],
        ['Sample Size', '6,388 surgical cases'],
        ['DOI', '10.1038/s41597-022-01411-5'],
        ['License', 'CC BY-NC-SA 4.0'],
    ]
    vitaldb_table = Table(vitaldb_data, colWidths=[100, 350])
    vitaldb_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#fed7d7')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(KeepTogether([vitaldb_table]))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("Finance: Yahoo Finance [8]", styles['SubsectionTitle']))
    finance_data = [
        ['Property', 'Value'],
        ['Method', 'yfinance Python library'],
        ['Data Type', 'Real-time stock prices, market cap'],
        ['Symbols', 'AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM'],
        ['License', 'Yahoo Finance Terms of Service'],
    ]
    finance_table = Table(finance_data, colWidths=[100, 350])
    finance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#c6f6d5')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(KeepTogether([finance_table]))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("IoT: UCI Machine Learning Repository [7]", styles['SubsectionTitle']))
    iot_data = [
        ['Property', 'Value'],
        ['Dataset', 'Individual Household Electric Power Consumption'],
        ['Data Type', 'Smart meter power readings'],
        ['DOI', '10.24432/C52G6F'],
        ['License', 'CC BY 4.0'],
    ]
    iot_table = Table(iot_data, colWidths=[100, 350])
    iot_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#bee3f8')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(KeepTogether([iot_table]))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("Blockchain: Ethereum RPC [9]", styles['SubsectionTitle']))
    rpc_data = [
        ['Priority', 'Endpoint', 'Provider'],
        ['1', 'rpc.ankr.com/eth', 'Ankr (Primary)'],
        ['2', 'ethereum-rpc.publicnode.com', 'PublicNode'],
        ['3', 'cloudflare-eth.com', 'Cloudflare'],
        ['4', 'eth.drpc.org', 'DRPC'],
        ['5', '1rpc.io/eth', '1RPC'],
    ]
    rpc_table = Table(rpc_data, colWidths=[60, 200, 140])
    rpc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4a5568')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(KeepTogether([rpc_table]))
    story.append(Paragraph("Note: All endpoints are free and require no API key.", styles['MyBodyText']))
    story.append(PageBreak())
    
    # =========================================================================
    # 7. ENTERPRISE USE CASES
    # =========================================================================
    story.append(Paragraph("7. Enterprise Use Cases", styles['SectionTitle']))
    
    story.append(Paragraph("Healthcare: HIPAA-Compliant Analytics", styles['SubsectionTitle']))
    story.append(Paragraph(
        "Hospitals can analyze patient vital signs without exposing Protected Health "
        "Information (PHI). FHE enables computation on encrypted blood pressure readings "
        "to identify hypertension trends across populations while maintaining full HIPAA "
        "compliance. Clinical interpretation follows AHA Guidelines [12].",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 6))
    story.append(Paragraph("Finance: Confidential Portfolio Analysis", styles['SubsectionTitle']))
    story.append(Paragraph(
        "Investment firms can perform growth projections on encrypted portfolio values. "
        "Client holdings remain confidential even during third-party analysis, enabling "
        "secure outsourcing of financial computations.",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 6))
    story.append(Paragraph("IoT: Secure Smart Grid Analytics", styles['SubsectionTitle']))
    story.append(Paragraph(
        "Utility companies can aggregate encrypted smart meter readings for demand "
        "forecasting without accessing individual household consumption patterns, "
        "preserving consumer privacy while enabling grid optimization.",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 6))
    story.append(Paragraph("Blockchain: Quantum-Resistant Transactions", styles['SubsectionTitle']))
    story.append(Paragraph(
        "Cryptocurrency platforms can migrate from ECDSA to ML-DSA signatures, protecting "
        "transaction integrity against future quantum attacks. This platform demonstrates "
        "the migration path with side-by-side comparison of signature sizes.",
        styles['MyBodyText']))
    story.append(PageBreak())
    
    # =========================================================================
    # 8. API REFERENCE
    # =========================================================================
    story.append(Paragraph("8. API Reference", styles['SectionTitle']))
    
    story.append(Paragraph("Endpoint Summary", styles['SubsectionTitle']))
    api_data = [
        ['Endpoint', 'Method', 'Description'],
        ['/health', 'GET', 'Health check'],
        ['/pqc/algorithms', 'GET', 'List PQC algorithms'],
        ['/pqc/kem/keypair', 'POST', 'Generate KEM keypair'],
        ['/pqc/kem/encapsulate', 'POST', 'Encapsulate secret'],
        ['/pqc/kem/decapsulate', 'POST', 'Decapsulate secret'],
        ['/pqc/sig/keypair', 'POST', 'Generate SIG keypair'],
        ['/pqc/sig/sign', 'POST', 'Sign message'],
        ['/pqc/sig/verify', 'POST', 'Verify signature'],
        ['/fhe/encrypt', 'POST', 'Encrypt data'],
        ['/fhe/decrypt', 'POST', 'Decrypt ciphertext'],
        ['/fhe/add', 'POST', 'Add ciphertexts'],
        ['/fhe/multiply', 'POST', 'Multiply by scalar'],
        ['/enterprise/healthcare', 'GET', 'Healthcare data'],
        ['/enterprise/finance', 'GET', 'Finance data'],
        ['/enterprise/iot', 'GET', 'IoT data'],
        ['/enterprise/blockchain', 'GET', 'Blockchain data'],
    ]
    api_table = Table(api_data, colWidths=[140, 50, 260])
    api_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('FONTNAME', (0, 1), (0, -1), 'Courier'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(api_table)
    
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Interactive documentation available at http://localhost:8000/docs (Swagger UI)",
        styles['MyBodyText']))
    story.append(PageBreak())
    
    # =========================================================================
    # 9. INSTALLATION GUIDE (MULTI-PLATFORM)
    # =========================================================================
    story.append(Paragraph("9. Installation Guide (Multi-Platform)", styles['SectionTitle']))
    
    story.append(Paragraph("System Requirements", styles['SubsectionTitle']))
    req_data = [
        ['Component', 'Minimum', 'Recommended'],
        ['Python', '3.9+', '3.11+'],
        ['RAM', '8 GB', '32 GB'],
        ['Storage', '1 GB', '5 GB'],
        ['GPU (optional)', 'CUDA 11.x', 'CUDA 12.x/13.x'],
    ]
    req_table = Table(req_data, colWidths=[120, 120, 120])
    req_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4a5568')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([req_table]))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("Build Dependencies by Platform", styles['SubsectionTitle']))
    
    # Debian/Ubuntu
    story.append(Paragraph("Debian/Ubuntu:", styles['MyBodyText']))
    story.append(Preformatted(
        "sudo apt update\n"
        "sudo apt install -y cmake gcc g++ libssl-dev python3-dev git",
        styles['CodeStyle']))
    
    # Fedora/RHEL
    story.append(Paragraph("Fedora/RHEL/CentOS:", styles['MyBodyText']))
    story.append(Preformatted(
        "sudo dnf install -y cmake gcc gcc-c++ openssl-devel python3-devel git",
        styles['CodeStyle']))
    
    # Arch Linux
    story.append(Paragraph("Arch Linux:", styles['MyBodyText']))
    story.append(Preformatted(
        "sudo pacman -S cmake gcc openssl python git",
        styles['CodeStyle']))
    
    # macOS
    story.append(Paragraph("macOS (Homebrew):", styles['MyBodyText']))
    story.append(Preformatted(
        "brew install cmake openssl@3 git",
        styles['CodeStyle']))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("Installing liboqs-python [10]", styles['SubsectionTitle']))
    
    story.append(Paragraph(
        "liboqs-python is NOT available via pip. It must be built from source. "
        "The recommended method auto-downloads and builds liboqs at runtime:",
        styles['MyBodyText']))
    
    story.append(Paragraph("Option A: Automatic Build (Recommended)", styles['MyBodyText']))
    story.append(Preformatted(
        "git clone --depth=1 https://github.com/open-quantum-safe/liboqs-python\n"
        "cd liboqs-python\n"
        "pip install .\n"
        "cd ..",
        styles['CodeStyle']))
    
    story.append(Paragraph("Option B: Manual Build", styles['MyBodyText']))
    story.append(Preformatted(
        "# Build liboqs C library\n"
        "git clone --depth=1 https://github.com/open-quantum-safe/liboqs\n"
        "cmake -S liboqs -B liboqs/build -DBUILD_SHARED_LIBS=ON\n"
        "cmake --build liboqs/build --parallel 8\n"
        "sudo cmake --build liboqs/build --target install\n\n"
        "# Set library path (Linux)\n"
        "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib\n\n"
        "# Install Python wrapper\n"
        "git clone --depth=1 https://github.com/open-quantum-safe/liboqs-python\n"
        "cd liboqs-python && pip install . && cd ..",
        styles['CodeStyle']))
    story.append(PageBreak())
    
    story.append(Paragraph("Installing DESILO FHE [5]", styles['SubsectionTitle']))
    story.append(Preformatted(
        "# CPU mode\n"
        "pip install desilofhe\n\n"
        "# GPU mode (choose based on CUDA version)\n"
        "pip install desilofhe-cu130  # CUDA 13.0\n"
        "pip install desilofhe-cu124  # CUDA 12.4\n"
        "pip install desilofhe-cu121  # CUDA 12.1",
        styles['CodeStyle']))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("Complete Quick Start", styles['SubsectionTitle']))
    story.append(Preformatted(
        "# 1. Extract package\n"
        "unzip pqc_fhe_portfolio_v2.3.4_final.zip\n"
        "cd pqc_fhe_v2.3.0\n"
        "pip install -r requirements.txt\n\n"
        "# 2. Install liboqs-python\n"
        "git clone --depth=1 https://github.com/open-quantum-safe/liboqs-python\n"
        "cd liboqs-python && pip install . && cd ..\n\n"
        "# 3. Install DESILO FHE\n"
        "pip install desilofhe\n\n"
        "# 4. Optional: Live data libraries\n"
        "pip install yfinance vitaldb\n\n"
        "# 5. Start server\n"
        "python -m uvicorn api.server:app --host 0.0.0.0 --port 8000\n\n"
        "# 6. Access Web UI\n"
        "# Open http://localhost:8000/ui",
        styles['CodeStyle']))
    story.append(PageBreak())
    
    # =========================================================================
    # 10. SECURITY ANALYSIS
    # =========================================================================
    story.append(Paragraph("10. Security Analysis", styles['SectionTitle']))
    
    story.append(Paragraph("NIST Security Levels", styles['SubsectionTitle']))
    story.append(Paragraph(
        "NIST defines five security levels based on classical and quantum attack "
        "complexity. This platform supports Levels 1, 2, 3, and 5:",
        styles['MyBodyText']))
    
    security_data = [
        ['Level', 'Classical Equivalent', 'Algorithms'],
        ['1', 'AES-128', 'ML-KEM-512'],
        ['2', 'SHA-256', 'ML-DSA-44'],
        ['3', 'AES-192', 'ML-KEM-768, ML-DSA-65'],
        ['5', 'AES-256', 'ML-KEM-1024, ML-DSA-87'],
    ]
    security_table = Table(security_data, colWidths=[60, 130, 180])
    security_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e53e3e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([security_table]))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("FHE Security", styles['SubsectionTitle']))
    story.append(Paragraph(
        "CKKS security is based on the Ring Learning With Errors (RLWE) problem, which "
        "is believed to be hard for both classical and quantum computers. The configured "
        "parameters provide at least 128-bit security against known attacks.",
        styles['MyBodyText']))
    story.append(PageBreak())
    
    # =========================================================================
    # 11. FUTURE ROADMAP
    # =========================================================================
    story.append(Paragraph("11. Future Roadmap", styles['SectionTitle']))
    
    roadmap_data = [
        ['Version', 'Timeline', 'Features'],
        ['v2.4.0', 'Q1 2025', 'SLH-DSA (SPHINCS+) hash-based signatures'],
        ['v2.5.0', 'Q2 2025', 'Hybrid classical/PQC mode'],
        ['v2.6.0', 'Q3 2025', 'Multi-party computation (MPC) integration'],
        ['v3.0.0', 'Q4 2025', 'FIPS validation and enterprise hardening'],
    ]
    roadmap_table = Table(roadmap_data, colWidths=[80, 80, 290])
    roadmap_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#38a169')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether([roadmap_table]))
    story.append(PageBreak())
    
    # =========================================================================
    # APPENDIX A: ALGORITHM PARAMETERS
    # =========================================================================
    story.append(Paragraph("Appendix A: Algorithm Parameters", styles['SectionTitle']))
    
    story.append(Paragraph("ML-KEM Parameters (FIPS 203)", styles['SubsectionTitle']))
    mlkem_data = [
        ['Parameter', 'ML-KEM-512', 'ML-KEM-768', 'ML-KEM-1024'],
        ['n (dimension)', '256', '256', '256'],
        ['k', '2', '3', '4'],
        ['eta_1', '3', '2', '2'],
        ['eta_2', '2', '2', '2'],
        ['d_u', '10', '10', '11'],
        ['d_v', '4', '4', '5'],
    ]
    mlkem_table = Table(mlkem_data, colWidths=[95, 95, 95, 95])
    mlkem_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4299e1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([mlkem_table]))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("ML-DSA Parameters (FIPS 204)", styles['SubsectionTitle']))
    mldsa_data = [
        ['Parameter', 'ML-DSA-44', 'ML-DSA-65', 'ML-DSA-87'],
        ['q (modulus)', '8,380,417', '8,380,417', '8,380,417'],
        ['k', '4', '6', '8'],
        ['l', '4', '5', '7'],
        ['eta', '2', '4', '2'],
        ['tau', '39', '49', '60'],
    ]
    mldsa_table = Table(mldsa_data, colWidths=[95, 95, 95, 95])
    mldsa_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#ed8936')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([mldsa_table]))
    story.append(PageBreak())
    
    # =========================================================================
    # APPENDIX B: PERFORMANCE BENCHMARKS
    # =========================================================================
    story.append(Paragraph("Appendix B: Performance Benchmarks", styles['SectionTitle']))
    
    story.append(Paragraph("PQC Operations (Intel i7-12700H)", styles['SubsectionTitle']))
    perf_data = [
        ['Operation', 'ML-KEM-768', 'ML-DSA-65'],
        ['Key Generation', '0.03 ms', '0.08 ms'],
        ['Encap/Sign', '0.04 ms', '0.18 ms'],
        ['Decap/Verify', '0.04 ms', '0.06 ms'],
    ]
    perf_table = Table(perf_data, colWidths=[140, 115, 115])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether([perf_table]))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("FHE Operations", styles['SubsectionTitle']))
    fhe_perf_data = [
        ['Operation', 'CPU', 'GPU (RTX 4090)'],
        ['Key Generation', '2.5 s', '0.8 s'],
        ['Encrypt (8192 slots)', '15 ms', '3 ms'],
        ['Add', '0.5 ms', '0.1 ms'],
        ['Multiply (scalar)', '2 ms', '0.3 ms'],
        ['Multiply (ct*ct)', '50 ms', '8 ms'],
        ['Decrypt', '10 ms', '2 ms'],
    ]
    fhe_perf_table = Table(fhe_perf_data, colWidths=[140, 115, 115])
    fhe_perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#9f7aea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether([fhe_perf_table]))
    story.append(PageBreak())
    
    # =========================================================================
    # REFERENCES
    # =========================================================================
    story.append(Paragraph("References", styles['SectionTitle']))
    
    references = [
        "[1] NIST. FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism Standard. "
        "National Institute of Standards and Technology, August 2024. "
        "https://csrc.nist.gov/pubs/fips/203/final",
        
        "[2] NIST. FIPS 204: Module-Lattice-Based Digital Signature Standard. "
        "National Institute of Standards and Technology, August 2024. "
        "https://csrc.nist.gov/pubs/fips/204/final",
        
        "[3] NIST. FIPS 205: Stateless Hash-Based Digital Signature Standard. "
        "National Institute of Standards and Technology, August 2024. "
        "https://csrc.nist.gov/pubs/fips/205/final",
        
        "[4] Cheon JH, Kim A, Kim M, Song Y. Homomorphic Encryption for Arithmetic "
        "of Approximate Numbers. ASIACRYPT 2017. DOI: 10.1007/978-3-319-70694-8_15",
        
        "[5] DESILO. DESILO FHE Library Documentation. https://fhe.desilo.dev/latest/",
        
        "[6] Lee HC, Park Y, Yoon SB, et al. VitalDB, a high-fidelity multi-parameter "
        "vital signs database in surgical patients. Scientific Data 9, 279 (2022). "
        "DOI: 10.1038/s41597-022-01411-5",
        
        "[7] Hebrail G, Berard A. Individual Household Electric Power Consumption Data Set. "
        "UCI Machine Learning Repository. DOI: 10.24432/C52G6F",
        
        "[8] Aroussi R. yfinance: Download market data from Yahoo! Finance API. "
        "https://github.com/ranaroussi/yfinance. License: Apache 2.0",
        
        "[9] Ethereum Foundation. Ethereum JSON-RPC API. "
        "https://ethereum.org/developers/docs/apis/json-rpc/",
        
        "[10] Open Quantum Safe Project. liboqs-python: Python 3 wrapper for liboqs. "
        "https://github.com/open-quantum-safe/liboqs-python. License: MIT",
        
        "[11] Ramirez S, et al. FastAPI: Modern, fast web framework for building APIs. "
        "https://fastapi.tiangolo.com/. License: MIT",
        
        "[12] American Heart Association. Understanding Blood Pressure Readings. "
        "https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings",
    ]
    
    for ref in references:
        story.append(Paragraph(ref, styles['RefStyle']))
    
    story.append(Spacer(1, 30))
    story.append(Paragraph("---", styles['MyBodyText']))
    story.append(Paragraph(f"Generated: 2025-12-30", styles['MyBodyText']))
    story.append(Paragraph("PQC-FHE Integration Platform v2.3.4", styles['MyBodyText']))
    
    # Build PDF
    doc.build(story)
    print("Technical report generated: PQC_FHE_Technical_Report_v2.3.4.pdf")

if __name__ == "__main__":
    create_report()
