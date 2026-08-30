import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

RISK_COLORS = {
    "low": colors.HexColor("#2C7A56"),
    "moderate": colors.HexColor("#C97E2B"),
    "high": colors.HexColor("#B3392C"),
}
TEAL = colors.HexColor("#0E7C86")
INK = colors.HexColor("#10202E")
INK_SOFT = colors.HexColor("#4B5D6B")


def generate_report_pdf(patient_name, patient_email, records, viewer_label="Patient self-report"):
    """
    records: list of dicts with keys
      created_at, model_used, prediction, risk_tier, probability_malignant, inputs (dict)
    Returns a BytesIO of the finished PDF.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CQTitle", parent=styles["Title"], textColor=INK, fontSize=20, spaceAfter=2,
    )
    sub_style = ParagraphStyle(
        "CQSub", parent=styles["Normal"], textColor=TEAL, fontSize=11, spaceAfter=14,
    )
    label_style = ParagraphStyle(
        "CQLabel", parent=styles["Normal"], textColor=INK_SOFT, fontSize=8,
        spaceAfter=0, uppercase=True,
    )
    value_style = ParagraphStyle(
        "CQValue", parent=styles["Normal"], textColor=INK, fontSize=11, spaceAfter=10,
    )
    disclaimer_style = ParagraphStyle(
        "CQDisclaimer", parent=styles["Normal"], textColor=INK_SOFT, fontSize=8,
        spaceAfter=0,
    )

    story = []
    story.append(Paragraph("CureQuest", title_style))
    story.append(Paragraph("Breast Cancer Risk Assessment Report", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DCE3E7")))
    story.append(Spacer(1, 14))

    # patient info block
    info_table = Table([
        [Paragraph("PATIENT", label_style), Paragraph("GENERATED", label_style), Paragraph("VIEWED BY", label_style)],
        [Paragraph(patient_name, value_style),
         Paragraph(datetime.utcnow().strftime("%d %b %Y, %H:%M UTC"), value_style),
         Paragraph(viewer_label, value_style)],
    ], colWidths=[2.6 * inch, 2.2 * inch, 2.2 * inch])
    info_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(info_table)
    story.append(Paragraph(patient_email, disclaimer_style))
    story.append(Spacer(1, 18))

    if not records:
        story.append(Paragraph("No assessments have been recorded yet.", value_style))
    else:
        latest = records[0]
        story.append(Paragraph("MOST RECENT ASSESSMENT", label_style))
        story.append(Spacer(1, 6))
        summary_data = [
            ["Result", "Risk Tier", "Probability", "Model", "Date"],
            [
                latest["prediction"],
                latest["risk_tier"].capitalize(),
                f"{latest['probability_malignant'] * 100:.1f}%",
                latest["model_used"],
                latest["created_at"][:16].replace("T", " "),
            ],
        ]
        summary_table = Table(summary_data, colWidths=[1.3 * inch, 1.2 * inch, 1.1 * inch, 1.6 * inch, 1.8 * inch])
        risk_color = RISK_COLORS.get(latest["risk_tier"], TEAL)
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F7F8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), INK_SOFT),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("FONTSIZE", (0, 1), (-1, 1), 10),
            ("TEXTCOLOR", (1, 1), (1, 1), risk_color),
            ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE3E7")),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 22))

        story.append(Paragraph("FULL ASSESSMENT HISTORY", label_style))
        story.append(Spacer(1, 6))

        history_header = ["Date", "Model", "Age", "Tumor Size (cm)", "Result", "Risk", "Probability"]
        history_rows = [history_header]
        for r in records:
            inputs = r.get("inputs", {})
            history_rows.append([
                r["created_at"][:16].replace("T", " "),
                r["model_used"],
                str(inputs.get("Age", "—")),
                str(inputs.get("Tumor_Size_cm", "—")),
                r["prediction"],
                r["risk_tier"].capitalize(),
                f"{r['probability_malignant'] * 100:.1f}%",
            ])

        history_table = Table(
            history_rows,
            colWidths=[1.1 * inch, 1.0 * inch, 0.5 * inch, 0.9 * inch, 0.9 * inch, 0.8 * inch, 0.9 * inch],
            repeatRows=1,
        )
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE3E7")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7F8")]),
        ]
        for i, r in enumerate(records, start=1):
            color = RISK_COLORS.get(r["risk_tier"], INK)
            table_style.append(("TEXTCOLOR", (5, i), (5, i), color))
            table_style.append(("FONTNAME", (5, i), (5, i), "Helvetica-Bold"))
        history_table.setStyle(TableStyle(table_style))
        story.append(history_table)

    story.append(Spacer(1, 26))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DCE3E7")))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "CureQuest is a decision-support tool only and is not a substitute for professional "
        "clinical diagnosis. All predictions should be reviewed by a qualified physician.",
        disclaimer_style,
    ))

    doc.build(story)
    buf.seek(0)
    return buf
