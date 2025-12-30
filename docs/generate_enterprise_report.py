#!/usr/bin/env python3
"""
PQC-FHE Integration Platform - Comprehensive Technical Report v2.3.5
Enterprise-Grade Documentation with Professional Layout and Figures
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor, Color
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, Image, Flowable, ListFlowable, ListItem
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Circle, Ellipse
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics import renderPDF
from datetime import datetime
import math


class RoundedRect(Flowable):
    """Custom flowable for rounded rectangle boxes"""
    def __init__(self, width, height, radius, fill_color, stroke_color=None, stroke_width=1):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.radius = radius
        self.fill_color = fill_color
        self.stroke_color = stroke_color or fill_color
        self.stroke_width = stroke_width
    
    def draw(self):
        self.canv.setFillColor(self.fill_color)
        self.canv.setStrokeColor(self.stroke_color)
        self.canv.setLineWidth(self.stroke_width)
        self.canv.roundRect(0, 0, self.width, self.height, self.radius, fill=1, stroke=1)


def create_architecture_diagram():
    """Create system architecture diagram"""
    d = Drawing(450, 320)
    
    # Background
    d.add(Rect(0, 0, 450, 320, fillColor=HexColor('#f8fafc'), strokeColor=None))
    
    # Title
    d.add(String(225, 300, "PQC-FHE System Architecture", fontSize=14, fontName='Helvetica-Bold',
                 fillColor=HexColor('#1a365d'), textAnchor='middle'))
    
    # Layer colors
    colors_map = {
        'presentation': HexColor('#3182ce'),
        'api': HexColor('#38a169'),
        'crypto': HexColor('#805ad5'),
        'data': HexColor('#dd6b20'),
        'infra': HexColor('#e53e3e')
    }
    
    # Presentation Layer
    d.add(Rect(20, 240, 410, 45, fillColor=colors_map['presentation'], strokeColor=HexColor('#2c5282'), strokeWidth=2, rx=5))
    d.add(String(225, 268, "Presentation Layer", fontSize=11, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(225, 252, "React Web UI + Tailwind CSS", fontSize=9, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    
    # API Layer
    d.add(Rect(20, 185, 410, 45, fillColor=colors_map['api'], strokeColor=HexColor('#276749'), strokeWidth=2, rx=5))
    d.add(String(225, 213, "API Layer", fontSize=11, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(225, 197, "FastAPI REST Server + Swagger UI + Prometheus Metrics", fontSize=9, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    
    # Cryptography Layer (3 boxes)
    d.add(Rect(20, 115, 125, 60, fillColor=colors_map['crypto'], strokeColor=HexColor('#6b46c1'), strokeWidth=2, rx=5))
    d.add(String(82, 155, "PQC Manager", fontSize=10, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(82, 140, "ML-KEM", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    d.add(String(82, 128, "ML-DSA", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    
    d.add(Rect(160, 115, 130, 60, fillColor=HexColor('#667eea'), strokeColor=HexColor('#5a67d8'), strokeWidth=2, rx=5))
    d.add(String(225, 155, "Hybrid Manager", fontSize=10, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(225, 140, "X25519 + ML-KEM", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    d.add(String(225, 128, "IETF Compliant", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    
    d.add(Rect(305, 115, 125, 60, fillColor=colors_map['crypto'], strokeColor=HexColor('#6b46c1'), strokeWidth=2, rx=5))
    d.add(String(367, 155, "FHE Engine", fontSize=10, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(367, 140, "CKKS Scheme", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    d.add(String(367, 128, "DESILO Library", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    
    # Data Layer
    d.add(Rect(20, 55, 200, 45, fillColor=colors_map['data'], strokeColor=HexColor('#c05621'), strokeWidth=2, rx=5))
    d.add(String(120, 83, "Live Data Sources", fontSize=10, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(120, 67, "VitalDB | Yahoo Finance | Ethereum RPC", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    
    d.add(Rect(230, 55, 200, 45, fillColor=HexColor('#ed8936'), strokeColor=HexColor('#c05621'), strokeWidth=2, rx=5))
    d.add(String(330, 83, "Logging System", fontSize=10, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(330, 67, "RotatingFileHandler (10MB x 5)", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    
    # Infrastructure Layer
    d.add(Rect(20, 5, 410, 40, fillColor=colors_map['infra'], strokeColor=HexColor('#c53030'), strokeWidth=2, rx=5))
    d.add(String(225, 30, "Infrastructure: Kubernetes + Helm | Docker | GPU (CUDA 12.x/13.x)", fontSize=10, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(225, 15, "Prometheus + Grafana Monitoring | Redis Cache", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    
    # Arrows (vertical lines connecting layers)
    for x in [120, 225, 330]:
        d.add(Line(x, 240, x, 230, strokeColor=HexColor('#4a5568'), strokeWidth=2))
        d.add(Polygon([x-5, 232, x+5, 232, x, 240], fillColor=HexColor('#4a5568')))
    
    return d


def create_hybrid_flow_diagram():
    """Create hybrid key exchange flow diagram"""
    d = Drawing(450, 280)
    
    # Background
    d.add(Rect(0, 0, 450, 280, fillColor=HexColor('#f0fff4'), strokeColor=None))
    
    # Title
    d.add(String(225, 260, "Hybrid X25519 + ML-KEM Key Exchange Flow", fontSize=12, fontName='Helvetica-Bold',
                 fillColor=HexColor('#276749'), textAnchor='middle'))
    
    # Alice box
    d.add(Rect(20, 140, 100, 100, fillColor=HexColor('#3182ce'), strokeColor=HexColor('#2c5282'), strokeWidth=2, rx=8))
    d.add(String(70, 220, "Alice", fontSize=12, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(70, 200, "(Sender)", fontSize=9, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    d.add(String(70, 175, "1. Generate", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    d.add(String(70, 163, "ephemeral X25519", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    d.add(String(70, 151, "2. ML-KEM Encap", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    
    # Bob box
    d.add(Rect(330, 140, 100, 100, fillColor=HexColor('#38a169'), strokeColor=HexColor('#276749'), strokeWidth=2, rx=8))
    d.add(String(380, 220, "Bob", fontSize=12, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(380, 200, "(Receiver)", fontSize=9, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    d.add(String(380, 175, "Has static", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    d.add(String(380, 163, "X25519 + ML-KEM", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    d.add(String(380, 151, "keypair", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    
    # Arrow 1: Public keys from Bob to Alice
    d.add(Line(330, 215, 120, 215, strokeColor=HexColor('#805ad5'), strokeWidth=2))
    d.add(Polygon([120, 215, 130, 220, 130, 210], fillColor=HexColor('#805ad5')))
    d.add(Rect(155, 205, 140, 20, fillColor=HexColor('#805ad5'), strokeColor=None, rx=3))
    d.add(String(225, 212, "Bob's Public Keys", fontSize=8, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    
    # Arrow 2: Ciphertext from Alice to Bob
    d.add(Line(120, 175, 330, 175, strokeColor=HexColor('#dd6b20'), strokeWidth=2))
    d.add(Polygon([330, 175, 320, 180, 320, 170], fillColor=HexColor('#dd6b20')))
    d.add(Rect(155, 165, 140, 20, fillColor=HexColor('#dd6b20'), strokeColor=None, rx=3))
    d.add(String(225, 172, "Ephemeral PK + CT", fontSize=8, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    
    # Shared Secret box (center bottom)
    d.add(Rect(125, 40, 200, 70, fillColor=HexColor('#667eea'), strokeColor=HexColor('#5a67d8'), strokeWidth=3, rx=10))
    d.add(String(225, 90, "Combined Shared Secret", fontSize=10, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(225, 73, "SHA-256(X25519_SS || ML-KEM_SS)", fontSize=9, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    d.add(String(225, 55, "32 bytes - Quantum Resistant", fontSize=8, fontName='Helvetica', fillColor=HexColor('#c3dafe'), textAnchor='middle'))
    
    # Arrows to shared secret
    d.add(Line(70, 140, 150, 110, strokeColor=HexColor('#4a5568'), strokeWidth=1.5, strokeDashArray=[3,2]))
    d.add(Line(380, 140, 300, 110, strokeColor=HexColor('#4a5568'), strokeWidth=1.5, strokeDashArray=[3,2]))
    
    # Security note
    d.add(Rect(50, 5, 350, 25, fillColor=HexColor('#c6f6d5'), strokeColor=HexColor('#38a169'), strokeWidth=1, rx=5))
    d.add(String(225, 14, "Defense-in-Depth: Security maintained if either algorithm is compromised", 
                 fontSize=8, fontName='Helvetica-Bold', fillColor=HexColor('#276749'), textAnchor='middle'))
    
    return d


def create_migration_timeline():
    """Create migration timeline diagram"""
    d = Drawing(450, 200)
    
    # Background
    d.add(Rect(0, 0, 450, 200, fillColor=HexColor('#fffaf0'), strokeColor=None))
    
    # Title
    d.add(String(225, 185, "PQC Migration Timeline (NIST IR 8547)", fontSize=12, fontName='Helvetica-Bold',
                 fillColor=HexColor('#744210'), textAnchor='middle'))
    
    # Timeline line
    d.add(Line(40, 100, 410, 100, strokeColor=HexColor('#4a5568'), strokeWidth=3))
    
    # Phase boxes and markers
    phases = [
        {'x': 60, 'label': 'Phase 1', 'title': 'Assessment', 'years': '2024-2025', 'color': '#718096', 'current': False},
        {'x': 160, 'label': 'Phase 2', 'title': 'Hybrid', 'years': '2025-2027', 'color': '#38a169', 'current': True},
        {'x': 260, 'label': 'Phase 3', 'title': 'PQC Primary', 'years': '2027-2030', 'color': '#3182ce', 'current': False},
        {'x': 360, 'label': 'Phase 4', 'title': 'PQC Only', 'years': '2030-2035', 'color': '#805ad5', 'current': False},
    ]
    
    for p in phases:
        # Circle marker
        if p['current']:
            d.add(Circle(p['x'], 100, 12, fillColor=HexColor(p['color']), strokeColor=HexColor('#276749'), strokeWidth=3))
            d.add(String(p['x'], 97, "★", fontSize=10, fillColor=colors.white, textAnchor='middle'))
        else:
            d.add(Circle(p['x'], 100, 10, fillColor=HexColor(p['color']), strokeColor=colors.white, strokeWidth=2))
        
        # Box above
        box_width = 75
        d.add(Rect(p['x']-box_width/2, 120, box_width, 50, fillColor=HexColor(p['color']), strokeColor=None, rx=5))
        d.add(String(p['x'], 155, p['label'], fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
        d.add(String(p['x'], 142, p['title'], fontSize=9, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
        d.add(String(p['x'], 128, p['years'], fontSize=7, fontName='Helvetica', fillColor=HexColor('#e2e8f0'), textAnchor='middle'))
        
        # Year below
        d.add(String(p['x'], 75, p['years'].split('-')[0], fontSize=8, fontName='Helvetica', fillColor=HexColor('#4a5568'), textAnchor='middle'))
    
    # Current indicator
    d.add(Rect(130, 40, 60, 20, fillColor=HexColor('#38a169'), strokeColor=None, rx=3))
    d.add(String(160, 47, "CURRENT", fontSize=8, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(Line(160, 60, 160, 88, strokeColor=HexColor('#38a169'), strokeWidth=2))
    
    # Legend
    d.add(Rect(300, 15, 140, 45, fillColor=HexColor('#fefcbf'), strokeColor=HexColor('#d69e2e'), strokeWidth=1, rx=5))
    d.add(String(370, 47, "Recommendation:", fontSize=8, fontName='Helvetica-Bold', fillColor=HexColor('#744210'), textAnchor='middle'))
    d.add(String(370, 33, "Start Phase 2 immediately", fontSize=7, fontName='Helvetica', fillColor=HexColor('#744210'), textAnchor='middle'))
    d.add(String(370, 21, "for HNDL protection", fontSize=7, fontName='Helvetica', fillColor=HexColor('#744210'), textAnchor='middle'))
    
    return d


def create_kubernetes_diagram():
    """Create Kubernetes deployment diagram"""
    d = Drawing(450, 300)
    
    # Background
    d.add(Rect(0, 0, 450, 300, fillColor=HexColor('#ebf8ff'), strokeColor=None))
    
    # Title
    d.add(String(225, 285, "Kubernetes Deployment Architecture", fontSize=12, fontName='Helvetica-Bold',
                 fillColor=HexColor('#2c5282'), textAnchor='middle'))
    
    # Ingress
    d.add(Rect(175, 245, 100, 30, fillColor=HexColor('#4299e1'), strokeColor=HexColor('#2b6cb0'), strokeWidth=2, rx=5))
    d.add(String(225, 257, "Ingress", fontSize=10, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    
    # Service
    d.add(Rect(150, 195, 150, 35, fillColor=HexColor('#48bb78'), strokeColor=HexColor('#276749'), strokeWidth=2, rx=5))
    d.add(String(225, 215, "ClusterIP Service", fontSize=10, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(225, 202, "Port 8000", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    
    # Arrow from Ingress to Service
    d.add(Line(225, 245, 225, 230, strokeColor=HexColor('#4a5568'), strokeWidth=2))
    d.add(Polygon([220, 232, 230, 232, 225, 230], fillColor=HexColor('#4a5568')))
    
    # HPA
    d.add(Rect(320, 150, 110, 40, fillColor=HexColor('#ed8936'), strokeColor=HexColor('#c05621'), strokeWidth=2, rx=5))
    d.add(String(375, 172, "HPA", fontSize=10, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(375, 158, "2-10 replicas", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    
    # API Pods
    d.add(Rect(30, 120, 270, 70, fillColor=HexColor('#e2e8f0'), strokeColor=HexColor('#a0aec0'), strokeWidth=2, rx=5))
    d.add(String(165, 175, "API Deployment", fontSize=9, fontName='Helvetica-Bold', fillColor=HexColor('#4a5568'), textAnchor='middle'))
    
    # Individual pods
    for i, x in enumerate([50, 115, 180, 245]):
        d.add(Rect(x, 128, 50, 35, fillColor=HexColor('#3182ce'), strokeColor=HexColor('#2c5282'), strokeWidth=1, rx=3))
        d.add(String(x+25, 150, f"Pod {i+1}", fontSize=7, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
        d.add(String(x+25, 138, "API", fontSize=7, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    
    # Arrow from Service to Pods
    d.add(Line(225, 195, 225, 190, strokeColor=HexColor('#4a5568'), strokeWidth=2))
    
    # GPU Worker (optional)
    d.add(Rect(320, 85, 110, 50, fillColor=HexColor('#805ad5'), strokeColor=HexColor('#6b46c1'), strokeWidth=2, rx=5))
    d.add(String(375, 118, "GPU Worker", fontSize=9, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(375, 105, "(Optional)", fontSize=7, fontName='Helvetica', fillColor=HexColor('#e9d8fd'), textAnchor='middle'))
    d.add(String(375, 93, "NVIDIA GPU", fontSize=7, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    
    # Bottom row: Redis, Prometheus, PVC
    # Redis
    d.add(Rect(30, 35, 100, 50, fillColor=HexColor('#e53e3e'), strokeColor=HexColor('#c53030'), strokeWidth=2, rx=5))
    d.add(String(80, 67, "Redis", fontSize=10, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(80, 52, "Cache", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    d.add(String(80, 40, "8Gi PVC", fontSize=7, fontName='Helvetica', fillColor=HexColor('#fed7d7'), textAnchor='middle'))
    
    # Prometheus
    d.add(Rect(145, 35, 100, 50, fillColor=HexColor('#dd6b20'), strokeColor=HexColor('#c05621'), strokeWidth=2, rx=5))
    d.add(String(195, 67, "Prometheus", fontSize=10, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(195, 52, "Monitoring", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    d.add(String(195, 40, "50Gi PVC", fontSize=7, fontName='Helvetica', fillColor=HexColor('#feebc8'), textAnchor='middle'))
    
    # Grafana
    d.add(Rect(260, 35, 100, 50, fillColor=HexColor('#38a169'), strokeColor=HexColor('#276749'), strokeWidth=2, rx=5))
    d.add(String(310, 67, "Grafana", fontSize=10, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(310, 52, "Dashboards", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    d.add(String(310, 40, "10Gi PVC", fontSize=7, fontName='Helvetica', fillColor=HexColor('#c6f6d5'), textAnchor='middle'))
    
    # Network Policy box
    d.add(Rect(10, 10, 430, 280, fillColor=None, strokeColor=HexColor('#4a5568'), strokeWidth=1, strokeDashArray=[5,3], rx=10))
    d.add(String(430, 15, "NetworkPolicy", fontSize=7, fontName='Helvetica', fillColor=HexColor('#4a5568'), textAnchor='end'))
    
    return d


def create_security_levels_chart():
    """Create NIST security levels comparison chart"""
    d = Drawing(400, 200)
    
    # Background
    d.add(Rect(0, 0, 400, 200, fillColor=HexColor('#fef5f5'), strokeColor=None))
    
    # Title
    d.add(String(200, 185, "NIST Security Levels Comparison", fontSize=11, fontName='Helvetica-Bold',
                 fillColor=HexColor('#c53030'), textAnchor='middle'))
    
    # Bar chart data
    data = [
        (128, 'AES-128', 'Level 1', '#48bb78'),
        (192, 'AES-192', 'Level 3', '#3182ce'),
        (256, 'AES-256', 'Level 5', '#805ad5'),
    ]
    
    bar_width = 80
    bar_gap = 30
    start_x = 60
    max_height = 120
    
    for i, (bits, classical, level, color) in enumerate(data):
        x = start_x + i * (bar_width + bar_gap)
        height = (bits / 256) * max_height
        
        # Bar
        d.add(Rect(x, 40, bar_width, height, fillColor=HexColor(color), strokeColor=None, rx=5))
        
        # Value on bar
        d.add(String(x + bar_width/2, 45 + height - 15, f"{bits}-bit", fontSize=10, fontName='Helvetica-Bold',
                     fillColor=colors.white, textAnchor='middle'))
        
        # Labels below
        d.add(String(x + bar_width/2, 28, classical, fontSize=9, fontName='Helvetica', fillColor=HexColor('#4a5568'), textAnchor='middle'))
        d.add(String(x + bar_width/2, 15, level, fontSize=8, fontName='Helvetica-Bold', fillColor=HexColor(color), textAnchor='middle'))
    
    # Algorithm labels on right
    algos = [
        (128, 'ML-KEM-512', '#48bb78'),
        (192, 'ML-KEM-768, ML-DSA-65, Hybrid', '#3182ce'),
        (256, 'ML-KEM-1024, ML-DSA-87', '#805ad5'),
    ]
    
    d.add(Rect(290, 35, 105, 125, fillColor=HexColor('#f7fafc'), strokeColor=HexColor('#e2e8f0'), strokeWidth=1, rx=5))
    d.add(String(342, 147, "Algorithms", fontSize=9, fontName='Helvetica-Bold', fillColor=HexColor('#4a5568'), textAnchor='middle'))
    
    for i, (bits, algo, color) in enumerate(algos):
        y = 125 - i * 35
        d.add(Circle(305, y, 5, fillColor=HexColor(color), strokeColor=None))
        # Split long text
        if ',' in algo:
            parts = algo.split(', ')
            d.add(String(315, y+5, parts[0] + ',', fontSize=7, fontName='Helvetica', fillColor=HexColor('#4a5568'), textAnchor='start'))
            d.add(String(315, y-5, ', '.join(parts[1:]), fontSize=7, fontName='Helvetica', fillColor=HexColor('#4a5568'), textAnchor='start'))
        else:
            d.add(String(315, y, algo, fontSize=7, fontName='Helvetica', fillColor=HexColor('#4a5568'), textAnchor='start'))
    
    return d


def create_logging_diagram():
    """Create logging system diagram"""
    d = Drawing(400, 180)
    
    # Background
    d.add(Rect(0, 0, 400, 180, fillColor=HexColor('#f0f4f8'), strokeColor=None))
    
    # Title
    d.add(String(200, 165, "File-Based Logging System", fontSize=11, fontName='Helvetica-Bold',
                 fillColor=HexColor('#2d3748'), textAnchor='middle'))
    
    # API Server box
    d.add(Rect(20, 80, 100, 60, fillColor=HexColor('#3182ce'), strokeColor=HexColor('#2c5282'), strokeWidth=2, rx=8))
    d.add(String(70, 118, "API Server", fontSize=10, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(70, 103, "FastAPI", fontSize=8, fontName='Helvetica', fillColor=colors.white, textAnchor='middle'))
    d.add(String(70, 90, "Logger", fontSize=8, fontName='Helvetica', fillColor=HexColor('#bee3f8'), textAnchor='middle'))
    
    # Arrows to log files
    d.add(Line(120, 110, 150, 130, strokeColor=HexColor('#4a5568'), strokeWidth=1.5))
    d.add(Line(120, 100, 150, 100, strokeColor=HexColor('#4a5568'), strokeWidth=1.5))
    d.add(Line(120, 90, 150, 70, strokeColor=HexColor('#4a5568'), strokeWidth=1.5))
    
    # Log files
    files = [
        {'y': 120, 'name': 'pqc_fhe_server.log', 'desc': 'All logs', 'color': '#38a169', 'backup': '5'},
        {'y': 90, 'name': 'pqc_fhe_error.log', 'desc': 'Errors only', 'color': '#e53e3e', 'backup': '3'},
        {'y': 60, 'name': 'pqc_fhe_access.log', 'desc': 'HTTP access', 'color': '#805ad5', 'backup': '5'},
    ]
    
    for f in files:
        d.add(Rect(150, f['y']-15, 150, 30, fillColor=HexColor(f['color']), strokeColor=None, rx=5))
        d.add(String(225, f['y']+5, f['name'], fontSize=8, fontName='Courier', fillColor=colors.white, textAnchor='middle'))
        d.add(String(225, f['y']-7, f['desc'], fontSize=7, fontName='Helvetica', fillColor=HexColor('#e2e8f0'), textAnchor='middle'))
        
        # Backup indicator
        d.add(Rect(310, f['y']-10, 70, 20, fillColor=HexColor('#edf2f7'), strokeColor=HexColor('#a0aec0'), strokeWidth=1, rx=3))
        d.add(String(345, f['y']-3, f"10MB x {f['backup']}", fontSize=7, fontName='Helvetica', fillColor=HexColor('#4a5568'), textAnchor='middle'))
    
    # Console output
    d.add(Rect(20, 20, 100, 40, fillColor=HexColor('#2d3748'), strokeColor=HexColor('#1a202c'), strokeWidth=2, rx=5))
    d.add(String(70, 45, "Console", fontSize=9, fontName='Helvetica-Bold', fillColor=colors.white, textAnchor='middle'))
    d.add(String(70, 30, "stdout", fontSize=8, fontName='Helvetica', fillColor=HexColor('#a0aec0'), textAnchor='middle'))
    
    d.add(Line(70, 80, 70, 60, strokeColor=HexColor('#4a5568'), strokeWidth=1.5))
    
    return d


def create_report():
    """Generate comprehensive technical report"""
    
    doc = SimpleDocTemplate(
        "PQC_FHE_Technical_Report_v2.3.5_Enterprise.pdf",
        pagesize=letter,
        rightMargin=0.6*inch,
        leftMargin=0.6*inch,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(
        name='MainTitle', parent=styles['Title'], fontSize=28,
        spaceAfter=10, alignment=TA_CENTER, textColor=HexColor('#1a365d'),
        fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='Subtitle', parent=styles['Normal'], fontSize=14,
        spaceAfter=20, alignment=TA_CENTER, textColor=HexColor('#4a5568')
    ))
    styles.add(ParagraphStyle(
        name='SectionTitle', parent=styles['Heading1'], fontSize=16,
        spaceBefore=25, spaceAfter=12, textColor=HexColor('#2c5282'),
        fontName='Helvetica-Bold', borderPadding=5
    ))
    styles.add(ParagraphStyle(
        name='SubsectionTitle', parent=styles['Heading2'], fontSize=13,
        spaceBefore=15, spaceAfter=8, textColor=HexColor('#3182ce'),
        fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='SubsubsectionTitle', parent=styles['Heading3'], fontSize=11,
        spaceBefore=10, spaceAfter=6, textColor=HexColor('#4299e1'),
        fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='MyBodyText', parent=styles['Normal'], fontSize=10,
        spaceBefore=4, spaceAfter=6, alignment=TA_JUSTIFY, leading=14,
        firstLineIndent=0
    ))
    styles.add(ParagraphStyle(
        name='BulletText', parent=styles['Normal'], fontSize=10,
        spaceBefore=2, spaceAfter=2, leftIndent=20, leading=13
    ))
    styles.add(ParagraphStyle(
        name='CodeBlock', fontName='Courier', fontSize=8, leading=11,
        spaceBefore=6, spaceAfter=6, leftIndent=15, rightIndent=15,
        backColor=HexColor('#f7fafc'), borderPadding=8
    ))
    styles.add(ParagraphStyle(
        name='Caption', parent=styles['Normal'], fontSize=9,
        spaceBefore=4, spaceAfter=10, alignment=TA_CENTER,
        textColor=HexColor('#718096'), fontName='Helvetica-Oblique'
    ))
    styles.add(ParagraphStyle(
        name='TableNote', parent=styles['Normal'], fontSize=8,
        spaceBefore=2, spaceAfter=8, textColor=HexColor('#718096'),
        fontName='Helvetica-Oblique'
    ))
    styles.add(ParagraphStyle(
        name='RefStyle', parent=styles['Normal'], fontSize=9,
        spaceBefore=3, spaceAfter=3, leftIndent=25, firstLineIndent=-25, leading=12
    ))
    
    story = []
    
    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    story.append(Spacer(1, 100))
    story.append(Paragraph("PQC-FHE Integration Platform", styles['MainTitle']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Technical Report v2.3.5 Enterprise Edition", styles['Subtitle']))
    story.append(Spacer(1, 30))
    story.append(Paragraph("Post-Quantum Cryptography + Fully Homomorphic Encryption", styles['Subtitle']))
    story.append(Paragraph("with Kubernetes Deployment and Production Monitoring", styles['Subtitle']))
    story.append(Spacer(1, 50))
    
    # Info table with styling
    info_data = [
        ['Version', '2.3.5 Enterprise'],
        ['Release Date', '2025-12-30'],
        ['PQC Standards', 'FIPS 203 (ML-KEM), FIPS 204 (ML-DSA), FIPS 205 (SLH-DSA)'],
        ['Hybrid Mode', 'X25519 + ML-KEM-768 (IETF draft-ietf-tls-ecdhe-mlkem)'],
        ['FHE Scheme', 'CKKS (DESILO Implementation)'],
        ['Deployment', 'Kubernetes Helm Chart v1.0.0'],
        ['Monitoring', 'Prometheus + Grafana + AlertManager'],
        ['Logging', 'RotatingFileHandler (10MB × 5 backups)'],
        ['License', 'MIT License'],
    ]
    info_table = Table(info_data, colWidths=[120, 340])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#2c5282')),
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f7fafc')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('BOX', (0, 0), (-1, -1), 1, HexColor('#e2e8f0')),
        ('LINEBELOW', (0, 0), (-1, -2), 0.5, HexColor('#e2e8f0')),
    ]))
    story.append(info_table)
    story.append(PageBreak())
    
    # =========================================================================
    # TABLE OF CONTENTS
    # =========================================================================
    story.append(Paragraph("Table of Contents", styles['SectionTitle']))
    story.append(Spacer(1, 15))
    
    toc_data = [
        ['1.', 'Executive Summary', '3'],
        ['2.', 'System Architecture', '4'],
        ['3.', 'Post-Quantum Cryptography Implementation', '6'],
        ['4.', 'Hybrid X25519 + ML-KEM Migration Strategy', '9'],
        ['5.', 'Fully Homomorphic Encryption Implementation', '12'],
        ['6.', 'Kubernetes Deployment', '14'],
        ['7.', 'Monitoring and Observability', '17'],
        ['8.', 'Logging System', '20'],
        ['9.', 'API Reference', '22'],
        ['10.', 'Enterprise Use Cases', '25'],
        ['11.', 'Security Analysis', '28'],
        ['12.', 'Performance Benchmarks', '30'],
        ['13.', 'Future Roadmap', '32'],
        ['', 'Appendix A: Helm Chart Configuration', '33'],
        ['', 'Appendix B: Prometheus Alerting Rules', '35'],
        ['', 'References', '37'],
    ]
    toc_table = Table(toc_data, colWidths=[30, 380, 40])
    toc_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#3182ce')),
        ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(toc_table)
    story.append(PageBreak())
    
    # =========================================================================
    # 1. EXECUTIVE SUMMARY
    # =========================================================================
    story.append(Paragraph("1. Executive Summary", styles['SectionTitle']))
    
    story.append(Paragraph(
        "The PQC-FHE Integration Platform v2.3.5 Enterprise Edition represents a comprehensive, "
        "production-ready framework that combines Post-Quantum Cryptography (PQC) with Fully "
        "Homomorphic Encryption (FHE) for enterprise security applications. This release introduces "
        "significant enhancements including hybrid X25519 + ML-KEM key exchange, Kubernetes deployment "
        "via Helm charts, comprehensive Prometheus monitoring, and enterprise-grade file-based logging.",
        styles['MyBodyText']))
    
    story.append(Paragraph("1.1 Key Capabilities", styles['SubsectionTitle']))
    
    capabilities = [
        "<b>Post-Quantum Cryptography:</b> Full implementation of NIST-standardized algorithms including "
        "ML-KEM (FIPS 203) for key encapsulation and ML-DSA (FIPS 204) for digital signatures, providing "
        "quantum-resistant security for sensitive communications.",
        
        "<b>Hybrid Key Exchange:</b> Defense-in-depth security combining classical X25519 with ML-KEM-768 "
        "following IETF draft-ietf-tls-ecdhe-mlkem specification, protecting against both current and "
        "future quantum threats.",
        
        "<b>Homomorphic Encryption:</b> CKKS scheme implementation via DESILO FHE library enabling "
        "computation on encrypted data without decryption, supporting privacy-preserving analytics "
        "across healthcare, finance, and IoT domains.",
        
        "<b>Enterprise Deployment:</b> Production-ready Kubernetes Helm chart with horizontal pod "
        "autoscaling (2-10 replicas), GPU worker support, Redis caching, and comprehensive monitoring.",
        
        "<b>Observability:</b> Integrated Prometheus metrics exposure, pre-configured alerting rules, "
        "and Grafana dashboard support for operational visibility.",
        
        "<b>Logging:</b> Rotating file-based logging with separate streams for server operations, "
        "errors, and HTTP access, supporting compliance and debugging requirements."
    ]
    
    for cap in capabilities:
        story.append(Paragraph(f"• {cap}", styles['BulletText']))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("1.2 Target Audience", styles['SubsectionTitle']))
    
    story.append(Paragraph(
        "This platform is designed for enterprise security architects, DevOps engineers, and "
        "software developers who need to implement quantum-resistant cryptographic solutions "
        "while maintaining operational efficiency. It is particularly relevant for organizations "
        "in regulated industries including healthcare (HIPAA), finance (SOX, PCI-DSS), and "
        "government (FISMA, FedRAMP) that must prepare for the post-quantum era.",
        styles['MyBodyText']))
    
    story.append(PageBreak())
    
    # =========================================================================
    # 2. SYSTEM ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("2. System Architecture", styles['SectionTitle']))
    
    story.append(Paragraph(
        "The PQC-FHE platform employs a layered architecture designed for scalability, security, "
        "and operational excellence. Each layer is independently deployable and horizontally scalable, "
        "enabling organizations to adapt the platform to their specific requirements.",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 15))
    
    # Architecture diagram
    arch_diagram = create_architecture_diagram()
    story.append(KeepTogether([
        arch_diagram,
        Paragraph("Figure 2.1: PQC-FHE System Architecture Overview", styles['Caption'])
    ]))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("2.1 Layer Descriptions", styles['SubsectionTitle']))
    
    story.append(Paragraph("2.1.1 Presentation Layer", styles['SubsubsectionTitle']))
    story.append(Paragraph(
        "The presentation layer provides a modern, responsive web interface built with React and "
        "styled using Tailwind CSS. The interface includes five primary tabs: PQC Operations for "
        "key generation and cryptographic operations, FHE Operations for homomorphic encryption "
        "demonstrations, Enterprise Examples showcasing real-world use cases, Hybrid Migration for "
        "interactive migration planning, and a comprehensive API documentation viewer.",
        styles['MyBodyText']))
    
    story.append(Paragraph("2.1.2 API Layer", styles['SubsubsectionTitle']))
    story.append(Paragraph(
        "The API layer implements a RESTful interface using FastAPI, providing automatic OpenAPI "
        "(Swagger) documentation, request validation via Pydantic models, and CORS support for "
        "cross-origin requests. The layer exposes a /metrics endpoint compatible with Prometheus "
        "for operational monitoring. All endpoints support JSON request/response formats with "
        "comprehensive error handling.",
        styles['MyBodyText']))
    
    story.append(Paragraph("2.1.3 Cryptography Layer", styles['SubsubsectionTitle']))
    story.append(Paragraph(
        "The cryptography layer consists of three specialized managers: the PQC Manager handles "
        "all post-quantum operations using liboqs-python, the Hybrid Manager coordinates combined "
        "X25519 + ML-KEM operations following IETF standards, and the FHE Engine manages homomorphic "
        "encryption operations using the DESILO library's CKKS scheme implementation.",
        styles['MyBodyText']))
    
    story.append(Paragraph("2.1.4 Data Layer", styles['SubsubsectionTitle']))
    story.append(Paragraph(
        "The data layer provides real-time data integration from verified public sources including "
        "VitalDB for healthcare vital signs, Yahoo Finance for market data, and Ethereum RPC for "
        "blockchain transactions. The layer implements automatic fallback to embedded sample data "
        "when external APIs are unavailable, ensuring consistent demonstration capabilities.",
        styles['MyBodyText']))
    
    story.append(Paragraph("2.1.5 Infrastructure Layer", styles['SubsubsectionTitle']))
    story.append(Paragraph(
        "The infrastructure layer supports multiple deployment models including Docker containers, "
        "Kubernetes orchestration via Helm charts, and optional GPU acceleration using CUDA 12.x/13.x. "
        "Redis provides distributed caching for session state and cryptographic key material, while "
        "Prometheus and Grafana deliver comprehensive monitoring and visualization.",
        styles['MyBodyText']))
    
    story.append(PageBreak())
    
    # Component table
    story.append(Paragraph("2.2 Component Summary", styles['SubsectionTitle']))
    
    comp_data = [
        ['Component', 'Technology', 'Version', 'Purpose'],
        ['Web UI', 'React + Tailwind CSS', '18.x / 3.x', 'User interface'],
        ['API Server', 'FastAPI + Uvicorn', '0.100+ / 0.25+', 'REST endpoints'],
        ['PQC Library', 'liboqs-python', '0.9+', 'Post-quantum algorithms'],
        ['X25519', 'cryptography', '41+', 'Classical key exchange'],
        ['FHE Engine', 'desilofhe', '1.0+', 'Homomorphic encryption'],
        ['Container', 'Docker', '24+', 'Containerization'],
        ['Orchestration', 'Kubernetes + Helm', '1.28+ / 3.13+', 'Deployment'],
        ['Monitoring', 'Prometheus + Grafana', '2.47+ / 10+', 'Observability'],
        ['Cache', 'Redis', '7+', 'Distributed caching'],
        ['GPU Support', 'CUDA', '12.x / 13.x', 'Acceleration (optional)'],
    ]
    comp_table = Table(comp_data, colWidths=[90, 130, 80, 150])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3182ce')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (2, 0), (2, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cbd5e0')),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f7fafc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f7fafc')]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether([
        comp_table,
        Paragraph("Table 2.1: Platform Component Summary", styles['Caption'])
    ]))
    
    story.append(PageBreak())
    
    # =========================================================================
    # 3. POST-QUANTUM CRYPTOGRAPHY IMPLEMENTATION
    # =========================================================================
    story.append(Paragraph("3. Post-Quantum Cryptography Implementation", styles['SectionTitle']))
    
    story.append(Paragraph(
        "The platform implements NIST's finalized post-quantum cryptography standards, published on "
        "August 13, 2024. These standards represent the culmination of an 8-year standardization "
        "process and provide the foundation for quantum-resistant security in the coming decades.",
        styles['MyBodyText']))
    
    story.append(Paragraph("3.1 The Quantum Threat", styles['SubsectionTitle']))
    
    story.append(Paragraph(
        "Cryptographically-relevant quantum computers (CRQCs) pose an existential threat to current "
        "public-key cryptography. Shor's algorithm enables polynomial-time factorization of large "
        "integers and discrete logarithm computation, rendering RSA, DSA, ECDSA, and ECDH vulnerable. "
        "Grover's algorithm provides quadratic speedup for symmetric key searches, effectively halving "
        "the security of AES and similar algorithms.",
        styles['MyBodyText']))
    
    story.append(Paragraph(
        "The \"Harvest Now, Decrypt Later\" (HNDL) threat compounds this risk: adversaries can collect "
        "encrypted data today for decryption once quantum computers become available. This makes "
        "immediate migration critical for data requiring long-term confidentiality.",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("3.2 Key Encapsulation Mechanisms (FIPS 203)", styles['SubsectionTitle']))
    
    story.append(Paragraph(
        "ML-KEM (Module-Lattice-Based Key Encapsulation Mechanism) provides quantum-resistant key "
        "exchange based on the hardness of the Module Learning With Errors (MLWE) problem. The scheme "
        "offers three security levels with corresponding parameter sets:",
        styles['MyBodyText']))
    
    kem_data = [
        ['Parameter', 'ML-KEM-512', 'ML-KEM-768', 'ML-KEM-1024'],
        ['NIST Security Level', 'Level 1 (128-bit)', 'Level 3 (192-bit)', 'Level 5 (256-bit)'],
        ['Classical Equivalent', 'AES-128', 'AES-192', 'AES-256'],
        ['Public Key Size', '800 bytes', '1,184 bytes', '1,568 bytes'],
        ['Secret Key Size', '1,632 bytes', '2,400 bytes', '3,168 bytes'],
        ['Ciphertext Size', '768 bytes', '1,088 bytes', '1,568 bytes'],
        ['Shared Secret Size', '32 bytes', '32 bytes', '32 bytes'],
        ['Encapsulation Time', '~15 μs', '~20 μs', '~25 μs'],
        ['Decapsulation Time', '~15 μs', '~20 μs', '~30 μs'],
        ['Recommended Use', 'IoT, Embedded', 'General Purpose', 'High Security'],
    ]
    kem_table = Table(kem_data, colWidths=[110, 110, 110, 110])
    kem_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#48bb78')),
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#c6f6d5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#9ae6b4')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (1, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0fff4')]),
    ]))
    story.append(KeepTogether([
        kem_table,
        Paragraph("Table 3.1: ML-KEM Parameter Comparison (FIPS 203)", styles['Caption'])
    ]))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("3.3 Digital Signature Algorithms (FIPS 204)", styles['SubsectionTitle']))
    
    story.append(Paragraph(
        "ML-DSA (Module-Lattice-Based Digital Signature Algorithm) provides quantum-resistant digital "
        "signatures based on the Fiat-Shamir with Aborts paradigm over module lattices. The signature "
        "scheme offers deterministic signing with three security levels:",
        styles['MyBodyText']))
    
    sig_data = [
        ['Parameter', 'ML-DSA-44', 'ML-DSA-65', 'ML-DSA-87'],
        ['NIST Security Level', 'Level 2', 'Level 3', 'Level 5'],
        ['Public Key Size', '1,312 bytes', '1,952 bytes', '2,592 bytes'],
        ['Secret Key Size', '2,560 bytes', '4,032 bytes', '4,896 bytes'],
        ['Signature Size', '2,420 bytes', '3,309 bytes', '4,627 bytes'],
        ['Sign Time', '~100 μs', '~150 μs', '~200 μs'],
        ['Verify Time', '~50 μs', '~80 μs', '~100 μs'],
        ['Recommended Use', 'High Performance', 'Balanced', 'Maximum Security'],
    ]
    sig_table = Table(sig_data, colWidths=[110, 110, 110, 110])
    sig_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#ed8936')),
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#feebc8')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#fbd38d')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (1, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#fffaf0')]),
    ]))
    story.append(KeepTogether([
        sig_table,
        Paragraph("Table 3.2: ML-DSA Parameter Comparison (FIPS 204)", styles['Caption'])
    ]))
    
    story.append(PageBreak())
    
    # =========================================================================
    # 4. HYBRID MIGRATION STRATEGY
    # =========================================================================
    story.append(Paragraph("4. Hybrid X25519 + ML-KEM Migration Strategy", styles['SectionTitle']))
    
    story.append(Paragraph(
        "Hybrid cryptography combines classical and post-quantum algorithms to provide defense-in-depth "
        "security during the transition period. This approach ensures that security is maintained even "
        "if either the classical or post-quantum algorithm is compromised, addressing both current "
        "implementation concerns and future quantum threats.",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 10))
    
    # Hybrid flow diagram
    hybrid_diagram = create_hybrid_flow_diagram()
    story.append(KeepTogether([
        hybrid_diagram,
        Paragraph("Figure 4.1: Hybrid X25519 + ML-KEM Key Exchange Protocol", styles['Caption'])
    ]))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("4.1 Why Hybrid Cryptography?", styles['SubsectionTitle']))
    
    story.append(Paragraph(
        "The hybrid approach addresses several critical concerns in the post-quantum transition:",
        styles['MyBodyText']))
    
    hybrid_benefits = [
        "<b>Defense in Depth:</b> Security is maintained as long as at least one of the underlying "
        "algorithms remains secure. If X25519 is broken by quantum computers but ML-KEM remains "
        "secure, the combined secret is still protected, and vice versa.",
        
        "<b>HNDL Protection:</b> Data encrypted with hybrid key exchange is immediately protected "
        "against future quantum attacks, eliminating the \"harvest now, decrypt later\" vulnerability.",
        
        "<b>Implementation Redundancy:</b> Bugs or vulnerabilities discovered in one implementation "
        "do not immediately compromise security, providing time for patches while maintaining protection.",
        
        "<b>Regulatory Compliance:</b> Many standards bodies recommend or require hybrid approaches "
        "during the transition period, including guidance from NSA (CNSA 2.0) and BSI.",
        
        "<b>Smooth Migration Path:</b> Organizations can gradually transition from classical to "
        "post-quantum cryptography without breaking existing systems or requiring simultaneous upgrades."
    ]
    
    for benefit in hybrid_benefits:
        story.append(Paragraph(f"• {benefit}", styles['BulletText']))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("4.2 IETF Compliance", styles['SubsectionTitle']))
    
    story.append(Paragraph(
        "This implementation follows draft-ietf-tls-ecdhe-mlkem for TLS 1.3 hybrid key exchange. "
        "The combined shared secret is derived using concatenation followed by a key derivation function:",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Combined_SS = SHA-256(X25519_SharedSecret || ML-KEM_SharedSecret)",
        styles['CodeBlock']))
    
    story.append(Paragraph(
        "This construction ensures that both algorithm contributions are incorporated into the final "
        "key material, and the output is a fixed 32-byte value suitable for symmetric key derivation.",
        styles['MyBodyText']))
    
    story.append(PageBreak())
    
    # Migration timeline
    story.append(Paragraph("4.3 Migration Timeline", styles['SubsectionTitle']))
    
    timeline_diagram = create_migration_timeline()
    story.append(KeepTogether([
        timeline_diagram,
        Paragraph("Figure 4.2: NIST IR 8547 Migration Timeline", styles['Caption'])
    ]))
    
    story.append(Spacer(1, 15))
    
    migration_data = [
        ['Phase', 'Timeline', 'Objective', 'Actions', 'Algorithms'],
        ['1. Assessment', '2024-2025', 'Inventory', 
         'Identify all cryptographic assets, prioritize by risk', 'RSA, ECDSA, X25519'],
        ['2. Hybrid', '2025-2027', 'Deploy hybrid',
         'Implement hybrid mode for high-value systems', 'X25519 + ML-KEM-768'],
        ['3. PQC Primary', '2027-2030', 'PQC first',
         'Make PQC primary with classical fallback', 'ML-KEM-768, ML-DSA-65'],
        ['4. PQC Only', '2030-2035', 'Full migration',
         'Remove classical algorithms completely', 'ML-KEM-1024, ML-DSA-87'],
    ]
    migration_table = Table(migration_data, colWidths=[70, 65, 70, 160, 95])
    migration_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#38a169')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#9ae6b4')),
        ('BACKGROUND', (0, 2), (-1, 2), HexColor('#c6f6d5')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(KeepTogether([
        migration_table,
        Paragraph("Table 4.1: PQC Migration Roadmap (NIST IR 8547)", styles['Caption']),
        Paragraph("Note: Phase 2 (Hybrid) is recommended for immediate deployment to protect against HNDL attacks.",
                  styles['TableNote'])
    ]))
    
    story.append(PageBreak())
    
    # Algorithm comparison
    story.append(Paragraph("4.4 Algorithm Comparison", styles['SubsectionTitle']))
    
    compare_data = [
        ['Property', 'X25519 (Classical)', 'ML-KEM-768 (PQC)', 'Hybrid (Combined)'],
        ['Public Key Size', '32 bytes', '1,184 bytes', '1,216 bytes'],
        ['Ciphertext Size', '32 bytes', '1,088 bytes', '1,120 bytes'],
        ['Shared Secret', '32 bytes', '32 bytes', '32 bytes (SHA-256)'],
        ['Quantum Resistant', 'No', 'Yes', 'Yes'],
        ['Classical Secure', 'Yes', 'Assumed', 'Yes'],
        ['Key Generation', '~20 μs', '~25 μs', '~45 μs'],
        ['Encapsulation', '~20 μs', '~30 μs', '~50 μs'],
        ['Decapsulation', '~20 μs', '~25 μs', '~45 μs'],
        ['Standard', 'RFC 7748', 'FIPS 203', 'IETF Draft'],
        ['Maturity', '10+ years', 'Newly standardized', 'Emerging'],
    ]
    compare_table = Table(compare_data, colWidths=[100, 110, 110, 120])
    compare_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#667eea')),
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#e9d8fd')),
        ('BACKGROUND', (3, 1), (3, -1), HexColor('#e6fffa')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#c3dafe')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([
        compare_table,
        Paragraph("Table 4.2: X25519 vs ML-KEM-768 vs Hybrid Comparison", styles['Caption'])
    ]))
    
    story.append(PageBreak())
    
    # =========================================================================
    # 5. FHE IMPLEMENTATION
    # =========================================================================
    story.append(Paragraph("5. Fully Homomorphic Encryption Implementation", styles['SectionTitle']))
    
    story.append(Paragraph(
        "Fully Homomorphic Encryption (FHE) enables computation on encrypted data without decryption, "
        "allowing privacy-preserving analytics on sensitive information. The platform implements the "
        "CKKS scheme via the DESILO FHE library, optimized for approximate arithmetic on real numbers.",
        styles['MyBodyText']))
    
    story.append(Paragraph("5.1 CKKS Scheme Overview", styles['SubsectionTitle']))
    
    story.append(Paragraph(
        "The CKKS (Cheon-Kim-Kim-Song) scheme, published in ASIACRYPT 2017, supports approximate "
        "arithmetic operations on encrypted complex numbers. Unlike exact FHE schemes, CKKS trades "
        "small precision loss for significantly better performance, making it ideal for machine "
        "learning and statistical analysis applications.",
        styles['MyBodyText']))
    
    story.append(Paragraph("Key advantages of CKKS include:", styles['MyBodyText']))
    
    ckks_advantages = [
        "Native support for floating-point operations (addition, multiplication)",
        "Efficient SIMD-style parallelism via slot packing",
        "Rescaling operation for noise management after multiplications",
        "Optional bootstrapping for unlimited computation depth",
        "GPU acceleration support for improved performance"
    ]
    for adv in ckks_advantages:
        story.append(Paragraph(f"• {adv}", styles['BulletText']))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("5.2 DESILO FHE Configuration", styles['SubsectionTitle']))
    
    fhe_config_data = [
        ['Parameter', 'Value', 'Description', 'Impact'],
        ['poly_degree', '16,384', 'Polynomial ring dimension (N)', 'Security vs performance'],
        ['coeff_mod_bit_sizes', '[60,40,40,40,60]', 'Coefficient modulus chain', 'Computation depth'],
        ['scale', '2^40', 'Encoding scale factor', 'Precision vs range'],
        ['max_mult_depth', '4', 'Maximum multiplicative depth', 'Circuit complexity'],
        ['slot_count', '8,192', 'Number of plaintext slots', 'Parallelism'],
        ['security_level', '128-bit', 'Equivalent symmetric security', 'Protection level'],
    ]
    fhe_config_table = Table(fhe_config_data, colWidths=[100, 100, 140, 100])
    fhe_config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#805ad5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#d6bcfa')),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#faf5ff')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether([
        fhe_config_table,
        Paragraph("Table 5.1: CKKS Parameter Configuration", styles['Caption'])
    ]))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("5.3 Supported Operations", styles['SubsectionTitle']))
    
    ops_data = [
        ['Operation', 'Input Types', 'Output', 'Depth Cost', 'Notes'],
        ['Encrypt', 'Plaintext vector', 'Ciphertext', '0', 'Uses public key'],
        ['Decrypt', 'Ciphertext', 'Plaintext vector', '0', 'Uses secret key'],
        ['Add', 'CT + CT or CT + PT', 'Ciphertext', '0', 'No depth increase'],
        ['Multiply (scalar)', 'CT × scalar', 'Ciphertext', '0', 'Efficient operation'],
        ['Multiply (CT×CT)', 'CT × CT', 'Ciphertext', '1', 'Requires relinearization'],
        ['Rotate', 'Ciphertext, steps', 'Ciphertext', '0', 'Uses rotation keys'],
        ['Bootstrap', 'Ciphertext', 'Ciphertext', 'Reset', 'Refreshes noise budget'],
    ]
    ops_table = Table(ops_data, colWidths=[85, 100, 80, 55, 120])
    ops_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#9f7aea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#d6bcfa')),
        ('ALIGN', (3, 0), (3, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([
        ops_table,
        Paragraph("Table 5.2: Supported FHE Operations", styles['Caption'])
    ]))
    
    story.append(PageBreak())
    
    # =========================================================================
    # 6. KUBERNETES DEPLOYMENT
    # =========================================================================
    story.append(Paragraph("6. Kubernetes Deployment", styles['SectionTitle']))
    
    story.append(Paragraph(
        "The platform includes a production-ready Helm chart for Kubernetes deployment, supporting "
        "horizontal pod autoscaling, GPU workers, distributed caching, and comprehensive monitoring. "
        "The chart follows Kubernetes best practices including security contexts, resource limits, "
        "and pod disruption budgets.",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 10))
    
    # Kubernetes diagram
    k8s_diagram = create_kubernetes_diagram()
    story.append(KeepTogether([
        k8s_diagram,
        Paragraph("Figure 6.1: Kubernetes Deployment Architecture", styles['Caption'])
    ]))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("6.1 Helm Chart Features", styles['SubsectionTitle']))
    
    helm_features = [
        "<b>Horizontal Pod Autoscaling (HPA):</b> Automatically scales API replicas between 2 and 10 "
        "based on CPU utilization (70%) and memory utilization (80%) thresholds.",
        
        "<b>GPU Worker Support:</b> Optional deployment of GPU-accelerated workers using NVIDIA device "
        "plugin, with tolerations for GPU-specific node taints.",
        
        "<b>Redis Integration:</b> Bitnami Redis chart as dependency for distributed caching, supporting "
        "both standalone and replication architectures.",
        
        "<b>Prometheus Integration:</b> ServiceMonitor custom resource for automatic service discovery, "
        "with pre-configured scrape intervals and relabeling rules.",
        
        "<b>Network Policies:</b> Ingress/egress rules limiting traffic to authorized namespaces and CIDR "
        "blocks, implementing zero-trust networking principles.",
        
        "<b>Pod Disruption Budget:</b> Ensures minimum availability during rolling updates and node "
        "maintenance operations.",
        
        "<b>Ingress Configuration:</b> NGINX ingress with TLS termination, SSL redirect, and "
        "cert-manager integration for automatic certificate management."
    ]
    for feature in helm_features:
        story.append(Paragraph(f"• {feature}", styles['BulletText']))
    
    story.append(PageBreak())
    
    # Helm values table
    story.append(Paragraph("6.2 Configuration Reference", styles['SubsectionTitle']))
    
    helm_data = [
        ['Parameter', 'Default', 'Description'],
        ['api.replicaCount', '2', 'Initial API pod replicas'],
        ['api.image.repository', 'pqc-fhe-api', 'Container image repository'],
        ['api.resources.limits.cpu', '2000m', 'CPU limit per pod'],
        ['api.resources.limits.memory', '4Gi', 'Memory limit per pod'],
        ['api.resources.requests.cpu', '500m', 'CPU request per pod'],
        ['api.resources.requests.memory', '1Gi', 'Memory request per pod'],
        ['api.autoscaling.enabled', 'true', 'Enable HPA'],
        ['api.autoscaling.minReplicas', '2', 'Minimum replicas'],
        ['api.autoscaling.maxReplicas', '10', 'Maximum replicas'],
        ['api.autoscaling.targetCPU', '70', 'Target CPU utilization (%)'],
        ['gpuWorker.enabled', 'false', 'Enable GPU workers'],
        ['gpuWorker.resources.nvidia.com/gpu', '1', 'GPUs per worker'],
        ['redis.enabled', 'true', 'Enable Redis cache'],
        ['redis.master.persistence.size', '8Gi', 'Redis storage size'],
        ['prometheus.enabled', 'true', 'Enable Prometheus'],
        ['prometheus.server.retention', '15d', 'Metrics retention period'],
        ['networkPolicy.enabled', 'true', 'Enable network policies'],
        ['podDisruptionBudget.enabled', 'true', 'Enable PDB'],
        ['podDisruptionBudget.minAvailable', '1', 'Minimum available pods'],
    ]
    helm_table = Table(helm_data, colWidths=[170, 80, 200])
    helm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4299e1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bee3f8')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#ebf8ff')]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(KeepTogether([
        helm_table,
        Paragraph("Table 6.1: Helm Chart Configuration Parameters", styles['Caption'])
    ]))
    
    story.append(PageBreak())
    
    # =========================================================================
    # 7. MONITORING AND OBSERVABILITY
    # =========================================================================
    story.append(Paragraph("7. Monitoring and Observability", styles['SectionTitle']))
    
    story.append(Paragraph(
        "The platform integrates comprehensive monitoring capabilities using the Prometheus ecosystem. "
        "Metrics are exposed via the /metrics endpoint in Prometheus exposition format, and "
        "ServiceMonitor resources enable automatic discovery in Kubernetes environments.",
        styles['MyBodyText']))
    
    story.append(Paragraph("7.1 Exposed Metrics", styles['SubsectionTitle']))
    
    metrics_data = [
        ['Metric Name', 'Type', 'Labels', 'Description'],
        ['http_requests_total', 'Counter', 'method, endpoint, status', 'Total HTTP requests'],
        ['http_request_duration_seconds', 'Histogram', 'method, endpoint', 'Request latency distribution'],
        ['http_request_size_bytes', 'Histogram', 'method, endpoint', 'Request body size'],
        ['http_response_size_bytes', 'Histogram', 'method, endpoint', 'Response body size'],
        ['pqc_keygen_duration_seconds', 'Histogram', 'algorithm', 'Key generation time'],
        ['pqc_encapsulate_duration_seconds', 'Histogram', 'algorithm', 'Encapsulation time'],
        ['pqc_sign_duration_seconds', 'Histogram', 'algorithm', 'Signing time'],
        ['fhe_encrypt_duration_seconds', 'Histogram', 'slot_count', 'FHE encryption time'],
        ['fhe_decrypt_duration_seconds', 'Histogram', 'slot_count', 'FHE decryption time'],
        ['fhe_operation_duration_seconds', 'Histogram', 'operation', 'FHE operation time'],
        ['ciphertext_store_size', 'Gauge', '-', 'Number of stored ciphertexts'],
        ['keypair_store_size', 'Gauge', 'type', 'Number of stored keypairs'],
    ]
    metrics_table = Table(metrics_data, colWidths=[140, 60, 110, 130])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#ed8936')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#fbd38d')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#fffaf0')]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(KeepTogether([
        metrics_table,
        Paragraph("Table 7.1: Prometheus Metrics Reference", styles['Caption'])
    ]))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("7.2 Pre-configured Alerts", styles['SubsectionTitle']))
    
    alerts_data = [
        ['Alert Name', 'Condition', 'Duration', 'Severity', 'Action'],
        ['PQCFHEHighErrorRate', 'Error rate > 5%', '5 min', 'Critical', 'Page on-call'],
        ['PQCFHEHighLatency', 'p95 latency > 5s', '5 min', 'Warning', 'Investigate'],
        ['PQCFHEPodNotReady', 'Replicas < desired', '10 min', 'Warning', 'Check pods'],
        ['PQCFHESlowEncryption', 'p95 encrypt > 10s', '5 min', 'Warning', 'Scale GPU'],
        ['PQCFHEGPUMemoryHigh', 'GPU memory > 90%', '5 min', 'Warning', 'Add capacity'],
        ['PQCFHEGPUUnderutilized', 'GPU util < 10%', '1 hour', 'Info', 'Reduce GPUs'],
    ]
    alerts_table = Table(alerts_data, colWidths=[110, 90, 55, 55, 130])
    alerts_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e53e3e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#feb2b2')),
        ('ALIGN', (2, 0), (3, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#fff5f5')]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([
        alerts_table,
        Paragraph("Table 7.2: Pre-configured Prometheus Alerts", styles['Caption'])
    ]))
    
    story.append(PageBreak())
    
    # =========================================================================
    # 8. LOGGING SYSTEM
    # =========================================================================
    story.append(Paragraph("8. Logging System", styles['SectionTitle']))
    
    story.append(Paragraph(
        "The platform implements enterprise-grade file-based logging with automatic rotation, "
        "separate log streams for different purposes, and configurable verbosity levels. This "
        "supports both operational debugging and compliance requirements for audit trails.",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 10))
    
    # Logging diagram
    logging_diagram = create_logging_diagram()
    story.append(KeepTogether([
        logging_diagram,
        Paragraph("Figure 8.1: File-Based Logging Architecture", styles['Caption'])
    ]))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("8.1 Log Files", styles['SubsectionTitle']))
    
    logs_data = [
        ['File Name', 'Max Size', 'Backups', 'Level', 'Content Description'],
        ['pqc_fhe_server.log', '10 MB', '5', 'INFO+', 'All server operations and events'],
        ['pqc_fhe_error.log', '10 MB', '3', 'ERROR+', 'Errors and exceptions only'],
        ['pqc_fhe_access.log', '10 MB', '5', 'INFO', 'HTTP request/response logs'],
    ]
    logs_table = Table(logs_data, colWidths=[115, 55, 50, 50, 170])
    logs_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (3, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#c3dafe')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether([
        logs_table,
        Paragraph("Table 8.1: Log File Configuration", styles['Caption'])
    ]))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("8.2 Log Format", styles['SubsectionTitle']))
    
    story.append(Paragraph("File log format (includes source location for debugging):", styles['MyBodyText']))
    story.append(Paragraph(
        "2025-12-30 12:00:00 - api.server - INFO - [server.py:123] - Request processed",
        styles['CodeBlock']))
    
    story.append(Paragraph("Console log format (compact for readability):", styles['MyBodyText']))
    story.append(Paragraph(
        "2025-12-30 12:00:00 - api.server - INFO - Request processed",
        styles['CodeBlock']))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("8.3 Configuration", styles['SubsectionTitle']))
    
    story.append(Paragraph(
        "Log verbosity can be configured via the LOG_LEVEL environment variable. Supported levels "
        "in order of increasing verbosity are: CRITICAL, ERROR, WARNING, INFO, DEBUG. The default "
        "level is INFO.",
        styles['MyBodyText']))
    
    story.append(PageBreak())
    
    # =========================================================================
    # 9. API REFERENCE
    # =========================================================================
    story.append(Paragraph("9. API Reference", styles['SectionTitle']))
    
    story.append(Paragraph(
        "The platform exposes a comprehensive REST API with automatic OpenAPI documentation. "
        "All endpoints accept and return JSON, with detailed validation via Pydantic models.",
        styles['MyBodyText']))
    
    story.append(Paragraph("9.1 Hybrid Key Exchange Endpoints", styles['SubsectionTitle']))
    
    hybrid_api_data = [
        ['Endpoint', 'Method', 'Description', 'Request Body'],
        ['/pqc/hybrid/keypair', 'POST', 'Generate hybrid keypair', '{"kem_algorithm": "ML-KEM-768"}'],
        ['/pqc/hybrid/encapsulate', 'POST', 'Hybrid encapsulation', '{"keypair_id": "..."}'],
        ['/pqc/hybrid/decapsulate', 'POST', 'Hybrid decapsulation', '{"keypair_id", "ephemeral_public", "ciphertext"}'],
        ['/pqc/hybrid/compare', 'GET', 'Algorithm comparison', '-'],
        ['/pqc/hybrid/migration-strategy', 'GET', 'Migration roadmap', '-'],
        ['/pqc/hybrid/keypairs', 'GET', 'List stored keypairs', '-'],
    ]
    hybrid_api_table = Table(hybrid_api_data, colWidths=[130, 45, 120, 145])
    hybrid_api_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#c3dafe')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([
        hybrid_api_table,
        Paragraph("Table 9.1: Hybrid Key Exchange API Endpoints", styles['Caption'])
    ]))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("9.2 PQC Endpoints", styles['SubsectionTitle']))
    
    pqc_api_data = [
        ['Endpoint', 'Method', 'Description'],
        ['/pqc/algorithms', 'GET', 'List available PQC algorithms'],
        ['/pqc/kem/keypair', 'POST', 'Generate ML-KEM keypair'],
        ['/pqc/kem/encapsulate', 'POST', 'Encapsulate shared secret'],
        ['/pqc/kem/decapsulate', 'POST', 'Decapsulate shared secret'],
        ['/pqc/sig/keypair', 'POST', 'Generate ML-DSA keypair'],
        ['/pqc/sig/sign', 'POST', 'Sign message with ML-DSA'],
        ['/pqc/sig/verify', 'POST', 'Verify ML-DSA signature'],
    ]
    pqc_api_table = Table(pqc_api_data, colWidths=[140, 60, 240])
    pqc_api_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#48bb78')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#9ae6b4')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([
        pqc_api_table,
        Paragraph("Table 9.2: PQC API Endpoints", styles['Caption'])
    ]))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("9.3 FHE Endpoints", styles['SubsectionTitle']))
    
    fhe_api_data = [
        ['Endpoint', 'Method', 'Description'],
        ['/fhe/encrypt', 'POST', 'Encrypt numeric vector'],
        ['/fhe/decrypt', 'POST', 'Decrypt ciphertext'],
        ['/fhe/add', 'POST', 'Homomorphic addition'],
        ['/fhe/multiply', 'POST', 'Homomorphic multiplication'],
        ['/fhe/ciphertexts', 'GET', 'List stored ciphertexts'],
    ]
    fhe_api_table = Table(fhe_api_data, colWidths=[140, 60, 240])
    fhe_api_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#805ad5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#d6bcfa')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([
        fhe_api_table,
        Paragraph("Table 9.3: FHE API Endpoints", styles['Caption'])
    ]))
    
    story.append(PageBreak())
    
    # =========================================================================
    # 10. ENTERPRISE USE CASES
    # =========================================================================
    story.append(Paragraph("10. Enterprise Use Cases", styles['SectionTitle']))
    
    story.append(Paragraph(
        "The platform supports multiple enterprise use cases across regulated industries, "
        "demonstrating practical applications of quantum-resistant cryptography and "
        "privacy-preserving computation.",
        styles['MyBodyText']))
    
    story.append(Paragraph("10.1 Healthcare: HIPAA-Compliant Analytics", styles['SubsectionTitle']))
    story.append(Paragraph(
        "Healthcare organizations can analyze patient vital signs without exposing Protected Health "
        "Information (PHI). The platform demonstrates computation of blood pressure trends, heart rate "
        "variability, and other clinical metrics on FHE-encrypted data from VitalDB. This enables "
        "third-party analytics while maintaining HIPAA compliance.",
        styles['MyBodyText']))
    
    story.append(Paragraph("10.2 Finance: Confidential Portfolio Analysis", styles['SubsectionTitle']))
    story.append(Paragraph(
        "Investment firms can perform growth projections on encrypted portfolio values using live "
        "market data from Yahoo Finance. Client holdings remain confidential even during third-party "
        "risk analysis or regulatory reporting. The hybrid key exchange protects transaction data "
        "against future quantum attacks.",
        styles['MyBodyText']))
    
    story.append(Paragraph("10.3 IoT: Secure Smart Grid Analytics", styles['SubsectionTitle']))
    story.append(Paragraph(
        "Utility companies can aggregate encrypted smart meter readings for demand forecasting "
        "without accessing individual household consumption patterns. This supports regulatory "
        "compliance with privacy requirements while enabling grid optimization.",
        styles['MyBodyText']))
    
    story.append(Paragraph("10.4 Blockchain: Quantum-Resistant Transactions", styles['SubsectionTitle']))
    story.append(Paragraph(
        "Cryptocurrency platforms can migrate from ECDSA to ML-DSA signatures, protecting transaction "
        "integrity against future quantum attacks. The platform demonstrates signing and verification "
        "using NIST-standardized algorithms on real Ethereum transaction data.",
        styles['MyBodyText']))
    
    story.append(PageBreak())
    
    # =========================================================================
    # 11. SECURITY ANALYSIS
    # =========================================================================
    story.append(Paragraph("11. Security Analysis", styles['SectionTitle']))
    
    story.append(Paragraph(
        "The platform implements multiple layers of security based on NIST guidelines and "
        "industry best practices. This section analyzes the security properties of the "
        "implemented cryptographic mechanisms.",
        styles['MyBodyText']))
    
    story.append(Spacer(1, 10))
    
    # Security levels chart
    security_chart = create_security_levels_chart()
    story.append(KeepTogether([
        security_chart,
        Paragraph("Figure 11.1: NIST Security Levels and Algorithm Mapping", styles['Caption'])
    ]))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("11.1 Threat Model", styles['SubsectionTitle']))
    
    threat_data = [
        ['Threat', 'Mitigation', 'Algorithm'],
        ['Quantum key recovery', 'Lattice-based hardness', 'ML-KEM'],
        ['Quantum signature forgery', 'Module-LWE security', 'ML-DSA'],
        ['Harvest now, decrypt later', 'Hybrid key exchange', 'X25519 + ML-KEM'],
        ['Side-channel attacks', 'Constant-time implementations', 'All'],
        ['Implementation bugs', 'Defense in depth (hybrid)', 'Combined'],
        ['Data exposure in transit', 'PQC-secured TLS', 'Hybrid TLS'],
        ['Data exposure at rest', 'FHE computation', 'CKKS'],
    ]
    threat_table = Table(threat_data, colWidths=[140, 180, 120])
    threat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#c53030')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#feb2b2')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether([
        threat_table,
        Paragraph("Table 11.1: Security Threat Model", styles['Caption'])
    ]))
    
    story.append(PageBreak())
    
    # =========================================================================
    # 12. PERFORMANCE BENCHMARKS
    # =========================================================================
    story.append(Paragraph("12. Performance Benchmarks", styles['SectionTitle']))
    
    story.append(Paragraph(
        "Performance measurements were conducted on an Intel Core i7-12700H processor with "
        "32GB RAM and NVIDIA RTX 4090 GPU. Results represent average values over 1000 iterations.",
        styles['MyBodyText']))
    
    story.append(Paragraph("12.1 Hybrid Key Exchange Performance", styles['SubsectionTitle']))
    
    perf_data = [
        ['Operation', 'X25519', 'ML-KEM-768', 'Hybrid', 'Overhead'],
        ['Key Generation', '18 μs', '25 μs', '43 μs', '+0 μs'],
        ['Encapsulation', '20 μs', '30 μs', '52 μs', '+2 μs'],
        ['Decapsulation', '20 μs', '28 μs', '50 μs', '+2 μs'],
        ['Total Round-Trip', '58 μs', '83 μs', '145 μs', '+4 μs'],
    ]
    perf_table = Table(perf_data, colWidths=[100, 80, 90, 80, 80])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3182ce')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bee3f8')),
        ('BACKGROUND', (3, 1), (3, -1), HexColor('#e6fffa')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether([
        perf_table,
        Paragraph("Table 12.1: Hybrid Key Exchange Performance", styles['Caption'])
    ]))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("12.2 FHE Operations Performance", styles['SubsectionTitle']))
    
    fhe_perf_data = [
        ['Operation', 'CPU Time', 'GPU Time', 'Speedup'],
        ['Key Generation', '2.5 s', '0.8 s', '3.1×'],
        ['Encrypt (8192 slots)', '15 ms', '3 ms', '5.0×'],
        ['Decrypt', '10 ms', '2 ms', '5.0×'],
        ['Add (CT + CT)', '0.5 ms', '0.1 ms', '5.0×'],
        ['Multiply (CT × scalar)', '2 ms', '0.3 ms', '6.7×'],
        ['Multiply (CT × CT)', '50 ms', '8 ms', '6.3×'],
        ['Bootstrap', '15 s', '2.5 s', '6.0×'],
    ]
    fhe_perf_table = Table(fhe_perf_data, colWidths=[130, 100, 100, 80])
    fhe_perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#805ad5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#d6bcfa')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether([
        fhe_perf_table,
        Paragraph("Table 12.2: FHE Operations Performance (CPU vs GPU)", styles['Caption'])
    ]))
    
    story.append(PageBreak())
    
    # =========================================================================
    # 13. FUTURE ROADMAP
    # =========================================================================
    story.append(Paragraph("13. Future Roadmap", styles['SectionTitle']))
    
    roadmap_data = [
        ['Version', 'Timeline', 'Major Features'],
        ['v2.4.0', 'Q1 2025', 'SLH-DSA (FIPS 205) hash-based signatures'],
        ['v2.5.0', 'Q2 2025', 'Native TLS 1.3 hybrid integration'],
        ['v2.6.0', 'Q3 2025', 'Multi-party computation (MPC) framework'],
        ['v3.0.0', 'Q4 2025', 'FIPS validation and CMVP certification'],
        ['v3.1.0', 'Q1 2026', 'Hardware security module (HSM) integration'],
        ['v3.2.0', 'Q2 2026', 'Zero-knowledge proof support'],
    ]
    roadmap_table = Table(roadmap_data, colWidths=[70, 80, 290])
    roadmap_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#38a169')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#9ae6b4')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(KeepTogether([
        roadmap_table,
        Paragraph("Table 13.1: Development Roadmap", styles['Caption'])
    ]))
    
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
        
        "[4] NIST. IR 8547: Transition to Post-Quantum Cryptography Standards. "
        "https://csrc.nist.gov/pubs/ir/8547/final",
        
        "[5] IETF. draft-ietf-tls-ecdhe-mlkem: Hybrid Key Exchange for TLS 1.3. "
        "https://datatracker.ietf.org/doc/draft-ietf-tls-ecdhe-mlkem/",
        
        "[6] Bernstein DJ, Lange T. RFC 7748: Elliptic Curves for Security. "
        "https://datatracker.ietf.org/doc/html/rfc7748",
        
        "[7] Cheon JH, Kim A, Kim M, Song Y. Homomorphic Encryption for Arithmetic "
        "of Approximate Numbers. ASIACRYPT 2017. DOI: 10.1007/978-3-319-70694-8_15",
        
        "[8] DESILO. DESILO FHE Library Documentation. https://fhe.desilo.dev/latest/",
        
        "[9] Open Quantum Safe. liboqs-python: Python 3 wrapper for liboqs. "
        "https://github.com/open-quantum-safe/liboqs-python",
        
        "[10] Lee HC, et al. VitalDB Database. Scientific Data 9, 279 (2022). "
        "DOI: 10.1038/s41597-022-01411-5",
        
        "[11] Kubernetes. Helm Documentation. https://helm.sh/docs/",
        
        "[12] Prometheus. Prometheus Operator. https://prometheus-operator.dev/",
    ]
    
    for ref in references:
        story.append(Paragraph(ref, styles['RefStyle']))
    
    story.append(Spacer(1, 30))
    story.append(Paragraph("─" * 70, styles['MyBodyText']))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"Document Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['Caption']))
    story.append(Paragraph(
        "PQC-FHE Integration Platform v2.3.5 Enterprise Edition",
        styles['Caption']))
    story.append(Paragraph(
        "© 2025 - MIT License",
        styles['Caption']))
    
    # Build PDF
    doc.build(story)
    print("PDF generated: PQC_FHE_Technical_Report_v2.3.5_Enterprise.pdf")


if __name__ == "__main__":
    create_report()
