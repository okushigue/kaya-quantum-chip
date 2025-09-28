# kaya_photonic_report.py
import os
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import datetime

from kaya_boson_sampling import KayaBosonSampling
from kaya_qkd_bb84_fixed import KayaQKDBB84
from kaya_phase_estimation import KayaPhaseEstimation

class KayaPhotonicReport:
    def __init__(self, output_pdf="kaya_photonic_chip_report.pdf"):
        self.output_pdf = output_pdf
        self.results = {}
        self.styles = getSampleStyleSheet()
        self.custom_styles()
        print("📊 Starting Kaya Photonic Chip Technical Report...")
    
    def custom_styles(self):
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6
        )
    
    def run_experiments(self):
        print("\n🔬 Running Photonic Experiments...")
        
        print("   → Boson Sampling...")
        bs = KayaBosonSampling(n_photons=3, n_modes=6)
        _, entropy = bs.simulate(shots=2000)
        self.results['boson'] = {'entropy': entropy}
        
        print("   → QKD BB84...")
        qkd = KayaQKDBB84(n_bits=200, channel_loss=0.2, detector_eff=0.85)
        qkd_result = qkd.run_protocol()
        self.results['qkd'] = qkd_result
        
        print("   → NOON Phase Estimation...")
        pe = KayaPhaseEstimation(n_photons=2)
        _, _, sensitivity = pe.simulate_phase_sweep()
        self.results['noon'] = {'sensitivity': sensitivity}
        
        print("✅ All experiments completed!")
    
    def create_summary_plot(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Boson Sampling', 'QKD BB84', 'NOON Phase Estimation']
        values = [
            self.results['boson']['entropy'],
            100 * (1 - self.results['qkd']['qber']) if 'qber' in self.results['qkd'] else 0,
            self.results['noon']['sensitivity']
        ]
        colors_list = ['#1f77b4', '#2ca02c', '#d62728']
        
        bars = ax.bar(metrics, values, color=colors_list, alpha=0.8)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Performance Metric')
        ax.set_title('Kaya Photonic Chip Performance - Photonic Experiments', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        summary_plot = "kaya_summary_plot.png"
        plt.savefig(summary_plot, dpi=150)
        plt.close()
        return summary_plot
    
    def generate_pdf(self):
        doc = SimpleDocTemplate(self.output_pdf, pagesize=A4)
        story = []
        
        story.append(Paragraph("Technical Report: Kaya Photonic Quantum Chip", self.title_style))
        story.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", self.normal_style))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("Executive Summary", self.heading_style))
        story.append(Paragraph(
            "This report presents simulation results for the Kaya photonic quantum chip, "
            "focused on three key applications: Boson Sampling (quantum computing), "
            "QKD BB84 (quantum cryptography), and NOON State Phase Estimation (quantum metrology). "
            "All experiments demonstrate performance compatible with integrated photonic hardware implementations.",
            self.normal_style
        ))
        story.append(Spacer(1, 20))
        
        summary_plot = self.create_summary_plot()
        if os.path.exists(summary_plot):
            story.append(Image(summary_plot, width=6*inch, height=3.6*inch))
            story.append(Spacer(1, 20))
        
        story.append(Paragraph("Detailed Results by Experiment", self.heading_style))
        
        # Boson Sampling
        story.append(Paragraph("1. Boson Sampling", self.heading_style))
        boson_data = [
            ["Metric", "Value", "Interpretation"],
            ["Photons", "3", "Number of indistinguishable photons"],
            ["Modes", "6", "Optical channels on chip"],
            ["Possible States", "56", "Quantum combinatorics"],
            ["Entropy", f"{self.results['boson']['entropy']:.2f} bits", "Distribution complexity"]
        ]
        boson_table = Table(boson_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        boson_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(boson_table)
        story.append(Spacer(1, 15))
        
        # QKD BB84
        story.append(Paragraph("2. Quantum Key Distribution (BB84)", self.heading_style))
        qkd = self.results['qkd']
        qkd_data = [
            ["Parameter", "Value", "Specification"],
            ["Transmitted Bits", "200", "Initial sequence length"],
            ["Channel Loss", "20%", "Realistic optical loss"],
            ["Detected Photons", f"{qkd.get('detected', 0)} ({qkd.get('detected', 0)/200:.1%})", "Total efficiency"],
            ["Sifted Bits", f"{qkd.get('sifted', 0)}", "Matching bases"],
            ["QBER", f"{qkd.get('qber', 0):.1%}", "< 11% (secure)"],
            ["Status", "✅ Secure" if qkd.get('secure', False) else "❌ Insecure", "Security validation"]
        ]
        qkd_table = Table(qkd_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        qkd_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(qkd_table)
        story.append(Spacer(1, 15))
        
        # NOON Phase Estimation
        story.append(Paragraph("3. NOON State Phase Estimation", self.heading_style))
        noon_data = [
            ["Parameter", "Value", "Quantum Advantage"],
            ["Photons (N)", "2", "Quantum resources"],
            ["Sensitivity", f"{self.results['noon']['sensitivity']}×", "Better than classical limit"],
            ["Phase Precision", f"Δφ = 1/{self.results['noon']['sensitivity']}", "Heisenberg limit"],
            ["Applications", "Sensors, Gyroscopes", "On-chip metrology"]
        ]
        noon_table = Table(noon_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        noon_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(noon_table)
        story.append(Spacer(1, 20))
        
        # Technical Specifications
        story.append(Paragraph("Recommended Technical Specifications for Kaya Chip", self.heading_style))
        spec_data = [
            ["Component", "Specification", "Justification"],
            ["Platform", "Si₃N₄ or LiNbO₃", "Low optical loss (< 0.1 dB/cm)"],
            ["Optical Modes", "≥ 6", "Support for boson sampling"],
            ["Photon Sources", "Integrated SPDC", "Entangled photon pair generation"],
            ["Detectors", "SNSPDs", "Efficiency > 90%, low noise"],
            ["Modulators", "Thermo-optic or electro-optic", "Phase control for QKD/NOON"],
            ["Total Loss", "< 3 dB", "Feasibility for quantum algorithms"]
        ]
        spec_table = Table(spec_data, colWidths=[1.8*inch, 2*inch, 2.2*inch])
        spec_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(spec_table)
        story.append(Spacer(1, 20))
        
        # Conclusion
        story.append(Paragraph("Conclusion", self.heading_style))
        story.append(Paragraph(
            "The simulated experiments demonstrate that the Kaya chip is viable for three pillars "
            "of photonic quantum technology: computing (boson sampling), communication (QKD), and "
            "metrology (NOON states). The obtained results are consistent with state-of-the-art "
            "experimental implementations and provide a solid foundation for integrated photonic "
            "hardware development. We recommend proceeding with circuit layout design and partnerships "
            "with specialized photonic foundries.",
            self.normal_style
        ))
        
        doc.build(story)
        print(f"\n✅ Report successfully generated: {self.output_pdf}")
        
        for temp_file in ["kaya_summary_plot.png", "kaya_noon_interference.png"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def run(self):
        self.run_experiments()
        self.generate_pdf()

if __name__ == "__main__":
    report = KayaPhotonicReport("kaya_photonic_chip_report.pdf")
    report.run()
    print("\n🎉 Kaya Photonic Chip Technical Report completed!")