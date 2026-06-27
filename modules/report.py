"""
Professional PDF Report Generator -- Fund Analysis Engine
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, PageBreak, NextPageTemplate,
    Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from datetime import date
import pandas as pd
import numpy as np
from io import BytesIO
from typing import List

# ── Page geometry ──────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN_H   = 22 * mm
MARGIN_TOP = 28 * mm
MARGIN_BOT = 22 * mm
USABLE_W   = PAGE_W - 2 * MARGIN_H   # ~470 pt

# ── Brand palette ──────────────────────────────────────────────────────────────
C_TEAL    = colors.HexColor('#154D57')
C_TEAL2   = colors.HexColor('#1A6B77')
C_TEAL_LT = colors.HexColor('#9FCDD4')
C_AQUA    = colors.HexColor('#EAF2F3')
C_WARM    = colors.HexColor('#F5F0EB')
C_BORDER  = colors.HexColor('#E8DDD3')
C_WHITE   = colors.white
C_INK     = colors.HexColor('#0A0A0A')
C_GREY    = colors.HexColor('#7A6F65')
C_GREEN   = colors.HexColor('#D4EDDA')
C_GREEN2  = colors.HexColor('#1A7A4A')
C_RED     = colors.HexColor('#FAE0E0')
C_RED2    = colors.HexColor('#C0392B')
C_AMBER   = colors.HexColor('#FEF3C7')

# ── Metric direction: True = higher raw value is better ───────────────────────
# Max Drawdown (%) is stored as a negative number (e.g. -15.2),
# so a higher (less negative) value = less severe drawdown = better.
_HIGHER_IS_BETTER = frozenset([
    'Ann. Return (%)', 'Alpha (ann. %)', 'Sharpe Ratio', 'Sortino Ratio',
    'Treynor Ratio', 'Information Ratio', 'R\u00b2', 'Upside Capture (%)',
    'Calmar Ratio', 'ESG Globe Rating', 'Max Drawdown (%)',
])
_LOWER_IS_BETTER = frozenset([
    'Ann. Volatility (%)', 'Beta', 'Tracking Error (%)',
    'Max DD Duration (mths)', 'Downside Capture (%)', 'OCF', 'Carbon Risk Score',
])

_REPORT_DATE = ''


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(val, decimals=4):
    """Format a numeric value; return '-' for NaN/non-numeric."""
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return '-'
        return f'{f:.{decimals}f}'
    except (ValueError, TypeError):
        return str(val) if val is not None else '-'


def _ordinal(n):
    n = int(n)
    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(
        n % 10 if n % 100 not in (11, 12, 13) else 0, 'th')
    return f'{n}{suffix}'


def _metric_colw(n_fund_cols, label_w=130.0):
    col = (USABLE_W - label_w) / n_fund_cols
    return [label_w] + [col] * n_fund_cols


# ── Page callbacks ─────────────────────────────────────────────────────────────

def _draw_cover(c, doc):
    # Teal header band
    band_h = 68 * mm
    c.setFillColor(C_TEAL)
    c.rect(0, PAGE_H - band_h, PAGE_W, band_h, fill=1, stroke=0)
    # Subtle second-teal accent strip at very top
    c.setFillColor(C_TEAL2)
    c.rect(0, PAGE_H - 3, PAGE_W, 3, fill=1, stroke=0)
    # Text inside band
    c.setFillColor(C_WHITE)
    c.setFont('Helvetica-Bold', 9)
    c.drawString(MARGIN_H, PAGE_H - 17 * mm, 'FUND ANALYSIS ENGINE')
    c.setStrokeColor(colors.HexColor('#1E6B78'))
    c.setLineWidth(0.5)
    c.line(MARGIN_H, PAGE_H - 21 * mm, PAGE_W - MARGIN_H, PAGE_H - 21 * mm)
    c.setFillColor(C_TEAL_LT)
    c.setFont('Helvetica', 8)
    c.drawString(MARGIN_H, PAGE_H - 27 * mm, 'QUANTITATIVE INVESTMENT ANALYSIS')
    # Left accent strip in content area
    c.setFillColor(C_AQUA)
    c.rect(MARGIN_H - 6, 18 * mm, 3, PAGE_H - band_h - 18 * mm - 4, fill=1, stroke=0)
    # Warm footer strip
    c.setFillColor(C_WARM)
    c.rect(0, 0, PAGE_W, 18 * mm, fill=1, stroke=0)
    c.setFillColor(C_BORDER)
    c.rect(0, 18 * mm, PAGE_W, 0.5, fill=1, stroke=0)
    c.setFillColor(C_GREY)
    c.setFont('Helvetica', 7.5)
    c.drawString(MARGIN_H, 7.5 * mm, 'CONFIDENTIAL \u00b7 FOR INTERNAL USE ONLY')
    c.drawRightString(PAGE_W - MARGIN_H, 7.5 * mm, f'Generated {_REPORT_DATE}')


def _draw_inner(c, doc):
    c.setFillColor(C_TEAL)
    c.rect(0, PAGE_H - 3, PAGE_W, 3, fill=1, stroke=0)
    y_hdr = PAGE_H - 13 * mm
    c.setFillColor(C_GREY)
    c.setFont('Helvetica', 7.5)
    c.drawString(MARGIN_H, y_hdr,
                 'FUND ANALYSIS ENGINE  \u00b7  QUANTITATIVE INVESTMENT ANALYSIS')
    c.drawRightString(PAGE_W - MARGIN_H, y_hdr, _REPORT_DATE)
    c.setStrokeColor(C_BORDER)
    c.setLineWidth(0.5)
    c.line(MARGIN_H, y_hdr - 3, PAGE_W - MARGIN_H, y_hdr - 3)
    y_ftr = MARGIN_BOT - 5 * mm
    c.line(MARGIN_H, y_ftr, PAGE_W - MARGIN_H, y_ftr)
    c.setFont('Helvetica', 7.5)
    c.setFillColor(C_GREY)
    c.drawString(MARGIN_H, y_ftr - 4.5 * mm, 'CONFIDENTIAL')
    c.drawCentredString(PAGE_W / 2, y_ftr - 4.5 * mm, str(doc.page))
    c.drawRightString(PAGE_W - MARGIN_H, y_ftr - 4.5 * mm, 'Fund Analysis Engine')


# ── Typography ─────────────────────────────────────────────────────────────────

def _styles():
    return {
        'cover_title': ParagraphStyle('cover_title',
            fontName='Helvetica-Bold', fontSize=30, textColor=C_INK,
            leading=36, spaceAfter=8, alignment=TA_LEFT),
        'cover_sub': ParagraphStyle('cover_sub',
            fontName='Helvetica', fontSize=13, textColor=C_TEAL,
            leading=18, spaceAfter=14, alignment=TA_LEFT),
        'cover_meta': ParagraphStyle('cover_meta',
            fontName='Helvetica', fontSize=10, textColor=C_GREY,
            leading=15, spaceAfter=4, alignment=TA_LEFT),
        'cover_body': ParagraphStyle('cover_body',
            fontName='Helvetica', fontSize=9.5, textColor=C_INK,
            leading=15, spaceAfter=6, alignment=TA_LEFT),
        'eyebrow': ParagraphStyle('eyebrow',
            fontName='Helvetica-Bold', fontSize=7.5, textColor=C_TEAL,
            leading=10, spaceAfter=2, spaceBefore=10, alignment=TA_LEFT),
        'section_title': ParagraphStyle('section_title',
            fontName='Helvetica-Bold', fontSize=17, textColor=C_INK,
            leading=21, spaceAfter=6, alignment=TA_LEFT),
        'subsection': ParagraphStyle('subsection',
            fontName='Helvetica-Bold', fontSize=10.5, textColor=C_TEAL,
            leading=14, spaceAfter=4, spaceBefore=10, alignment=TA_LEFT),
        'body': ParagraphStyle('body',
            fontName='Helvetica', fontSize=9, textColor=C_INK,
            leading=14, spaceAfter=5, alignment=TA_LEFT),
        'note': ParagraphStyle('note',
            fontName='Helvetica-Oblique', fontSize=8, textColor=C_GREY,
            leading=11, spaceAfter=4, alignment=TA_LEFT),
        'disclaimer': ParagraphStyle('disclaimer',
            fontName='Helvetica', fontSize=8, textColor=C_GREY,
            leading=12, spaceAfter=4, alignment=TA_LEFT),
        'toc_entry': ParagraphStyle('toc_entry',
            fontName='Helvetica', fontSize=10, textColor=C_INK,
            leading=14, spaceAfter=0, alignment=TA_LEFT),
        'toc_num': ParagraphStyle('toc_num',
            fontName='Helvetica-Bold', fontSize=10, textColor=C_TEAL,
            leading=14, spaceAfter=0, alignment=TA_CENTER),
        'legend': ParagraphStyle('legend',
            fontName='Helvetica', fontSize=7.5, textColor=C_GREY,
            leading=10, spaceAfter=3, alignment=TA_LEFT),
    }


def _section_hdr(eyebrow, title, S):
    return [
        Spacer(1, 4 * mm),
        Paragraph(eyebrow.upper(), S['eyebrow']),
        Paragraph(title, S['section_title']),
        HRFlowable(width='100%', thickness=0.75, color=C_BORDER, spaceAfter=5),
    ]


# ── Table builders ─────────────────────────────────────────────────────────────

def _base_cmds(n_data_rows, first_left=True):
    cmds = [
        ('BACKGROUND',    (0, 0), (-1, 0),  C_TEAL),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  C_WHITE),
        ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0),  8),
        ('ALIGN',         (0, 0), (-1, 0),  'CENTER'),
        ('TOPPADDING',    (0, 0), (-1, 0),  7),
        ('BOTTOMPADDING', (0, 0), (-1, 0),  7),
        ('LEFTPADDING',   (0, 0), (-1, 0),  7),
        ('RIGHTPADDING',  (0, 0), (-1, 0),  7),
        ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 1), (-1, -1), 8),
        ('TEXTCOLOR',     (0, 1), (-1, -1), C_INK),
        ('TOPPADDING',    (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ('LEFTPADDING',   (0, 1), (-1, -1), 7),
        ('RIGHTPADDING',  (0, 1), (-1, -1), 7),
        ('ALIGN',         (1, 1), (-1, -1), 'RIGHT'),
        ('ALIGN',         (0, 1), (0, -1),  'LEFT' if first_left else 'CENTER'),
        ('FONTNAME',      (0, 1), (0, -1),  'Helvetica-Bold'),
        ('LINEBELOW',     (0, 0), (-1, 0),  0.75, C_TEAL),
        ('LINEBELOW',     (0, 1), (-1, -2), 0.25, C_BORDER),
        ('BOX',           (0, 0), (-1, -1), 0.5,  C_BORDER),
    ]
    for i in range(1, n_data_rows + 1):
        cmds.append(('BACKGROUND', (0, i), (-1, i),
                     C_WHITE if i % 2 == 1 else C_WARM))
    return cmds


def _make_table(data, col_widths):
    t = Table(data, colWidths=col_widths, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle(_base_cmds(len(data) - 1)))
    return t


def _ranking_table(data, col_widths, n_funds):
    """Ranking table with rank-1 green and last-rank red row highlights."""
    cmds = _base_cmds(len(data) - 1)
    for i, row in enumerate(data[1:], start=1):
        try:
            rank = int(row[-1])
            if rank == 1:
                cmds += [('BACKGROUND', (0, i), (-1, i), C_GREEN),
                         ('FONTNAME',   (0, i), (-1, i), 'Helvetica-Bold')]
            elif rank == n_funds:
                cmds.append(('BACKGROUND', (0, i), (-1, i), C_RED))
        except (ValueError, TypeError):
            pass
    t = Table(data, colWidths=col_widths, hAlign='LEFT')
    t.setStyle(TableStyle(cmds))
    return t


def _color_metrics_table(data, col_widths):
    """
    Metrics table with per-row best (green) / worst (red) cell highlighting.
    data[0] is the header row: ["Metric", fund1, fund2, ...]
    data[r] for r >= 1: [metric_name, val1_str, val2_str, ...]
    """
    n_data_rows = len(data) - 1
    cmds = _base_cmds(n_data_rows)

    for r_idx, row in enumerate(data[1:], start=1):
        metric = row[0]

        # Parse string values back to floats
        vals = []
        for cell in row[1:]:
            try:
                f = float(cell)
                vals.append(None if (np.isnan(f) or np.isinf(f)) else f)
            except (ValueError, TypeError):
                vals.append(None)

        valid = [v for v in vals if v is not None]
        if len(valid) < 2:
            continue

        if metric in _HIGHER_IS_BETTER:
            best_val  = max(valid)
            worst_val = min(valid)
        elif metric in _LOWER_IS_BETTER:
            best_val  = min(valid)
            worst_val = max(valid)
        else:
            continue

        # Avoid marking identical best/worst (all funds tied)
        if best_val == worst_val:
            continue

        for c_idx, v in enumerate(vals, start=1):
            if v is None:
                continue
            if v == best_val:
                cmds += [
                    ('BACKGROUND', (c_idx, r_idx), (c_idx, r_idx), C_GREEN),
                    ('FONTNAME',   (c_idx, r_idx), (c_idx, r_idx), 'Helvetica-Bold'),
                ]
            elif v == worst_val:
                cmds.append(
                    ('BACKGROUND', (c_idx, r_idx), (c_idx, r_idx), C_RED))

    t = Table(data, colWidths=col_widths, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle(cmds))
    return t


def _color_legend(S):
    """Small inline legend explaining the green/red cell coloring."""
    legend_data = [
        ['', 'Best value', '', 'Worst value', ''],
    ]
    legend_colw = [6, 52, 10, 56, USABLE_W - 124]
    t = Table(legend_data, colWidths=legend_colw, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (0, 0), C_GREEN),
        ('BACKGROUND',    (2, 0), (2, 0), C_RED),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica'),
        ('FONTSIZE',      (0, 0), (-1, 0), 7.5),
        ('TEXTCOLOR',     (0, 0), (-1, 0), C_GREY),
        ('VALIGN',        (0, 0), (-1, 0), 'MIDDLE'),
        ('TOPPADDING',    (0, 0), (-1, 0), 3),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 3),
        ('BOX',           (0, 0), (0, 0), 0.5, C_BORDER),
        ('BOX',           (2, 0), (2, 0), 0.5, C_BORDER),
    ]))
    return [t, Spacer(1, 3 * mm)]


# ── Metric DataFrame → table data (metrics as rows, funds as cols) ─────────────

def _metrics_as_rows(df: pd.DataFrame, label_col='Metric'):
    """
    df: index = fund names, columns = metric names.
    Returns table data list: row 0 = header, row r = [metric, val1, val2, ...]
    """
    fund_names   = list(df.index)
    metric_names = list(df.columns)
    header = [label_col] + fund_names
    rows   = [header]
    for metric in metric_names:
        row = [metric]
        for fund in fund_names:
            val = df.loc[fund, metric]
            row.append(_fmt(val))
        rows.append(row)
    return rows


# ── Table of Contents ──────────────────────────────────────────────────────────

def _toc(S):
    """Returns story elements for the Table of Contents page."""
    sections = [
        ('01', 'Executive Summary',       'Key findings and fund overview'),
        ('02', 'Performance Metrics',     'Core and downside risk metrics for all funds'),
        ('03', 'Fund Rankings',           'TOPSIS and Yuan & Yuan results and comparison'),
        ('04', 'Sensitivity Analysis',    'Rankings across five alternative weight schemes'),
        ('05', 'Methodology',             'Algorithm descriptions and metric glossary'),
        ('06', 'Disclaimer & Sources',    'Data provenance and regulatory notices'),
    ]

    toc_data = [['Section', 'Title', 'Description']]
    for num, title, desc in sections:
        toc_data.append([num, title, desc])

    toc_colw = [32, USABLE_W * 0.30, USABLE_W - 32 - USABLE_W * 0.30]
    cmds = [
        ('BACKGROUND',    (0, 0), (-1, 0), C_TEAL),
        ('TEXTCOLOR',     (0, 0), (-1, 0), C_WHITE),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 8),
        ('ALIGN',         (0, 0), (-1, 0), 'CENTER'),
        ('TOPPADDING',    (0, 0), (-1, 0), 7),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 7),
        ('FONTSIZE',      (0, 1), (-1, -1), 9),
        ('FONTNAME',      (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',      (1, 1), (1, -1), 'Helvetica-Bold'),
        ('FONTNAME',      (2, 1), (2, -1), 'Helvetica'),
        ('TEXTCOLOR',     (0, 1), (0, -1), C_TEAL),
        ('TEXTCOLOR',     (1, 1), (1, -1), C_INK),
        ('TEXTCOLOR',     (2, 1), (2, -1), C_GREY),
        ('ALIGN',         (0, 1), (0, -1), 'CENTER'),
        ('ALIGN',         (1, 1), (1, -1), 'LEFT'),
        ('ALIGN',         (2, 1), (2, -1), 'LEFT'),
        ('TOPPADDING',    (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 9),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
        ('LINEBELOW',     (0, 0), (-1, 0), 0.75, C_TEAL),
        ('LINEBELOW',     (0, 1), (-1, -2), 0.25, C_BORDER),
        ('BOX',           (0, 0), (-1, -1), 0.5, C_BORDER),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]
    for i in range(1, len(sections) + 1):
        cmds.append(('BACKGROUND', (0, i), (-1, i),
                     C_WHITE if i % 2 == 1 else C_WARM))

    t = Table(toc_data, colWidths=toc_colw, hAlign='LEFT')
    t.setStyle(TableStyle(cmds))

    return [
        *_section_hdr('Contents', 'Table of Contents', S),
        Spacer(1, 3 * mm),
        t,
        PageBreak(),
    ]


# ── KPI / Key Findings strip ───────────────────────────────────────────────────

def _kpi_strip(fund_names, topsis_df, yuan_df, sample_period, S):
    """4-column KPI strip: funds analysed, sample period, top TOPSIS, top Yuan."""
    all_funds = set(fund_names)
    try:
        t_funds     = topsis_df.loc[topsis_df.index.isin(all_funds)]
        best_topsis = t_funds['Rank'].idxmin() if not t_funds.empty else '-'
    except Exception:
        best_topsis = '-'
    try:
        y_funds   = yuan_df.loc[yuan_df.index.isin(all_funds)]
        best_yuan = y_funds['Rank'].idxmin() if not y_funds.empty else '-'
    except Exception:
        best_yuan = '-'

    col_w = USABLE_W / 4
    kpi = [
        ['Funds Analysed', 'Sample Period', 'Top Ranked (TOPSIS)', 'Top Ranked (Yuan)'],
        [str(len(fund_names)), sample_period, str(best_topsis), str(best_yuan)],
    ]
    t = Table(kpi, colWidths=[col_w] * 4, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), C_AQUA),
        ('TEXTCOLOR',     (0, 0), (-1, 0), C_TEAL),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 7.5),
        ('ALIGN',         (0, 0), (-1, 0), 'CENTER'),
        ('TOPPADDING',    (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND',    (0, 1), (-1, 1), C_WHITE),
        ('TEXTCOLOR',     (0, 1), (-1, 1), C_INK),
        ('FONTNAME',      (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 1), (-1, 1), 11),
        ('ALIGN',         (0, 1), (-1, 1), 'CENTER'),
        ('TOPPADDING',    (0, 1), (-1, 1), 9),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 9),
        ('BOX',           (0, 0), (-1, -1), 0.75, C_TEAL),
        ('LINEBELOW',     (0, 0), (-1, 0), 0.5, C_TEAL),
        ('INNERGRID',     (0, 0), (-1, -1), 0.25, C_BORDER),
    ]))
    return [t, Spacer(1, 5 * mm)]


def _key_findings(fund_names, topsis_df, yuan_df, core_df, downside_df, S):
    """
    Six-column key findings strip: top TOPSIS, top Yuan, best Sharpe,
    best alpha, best drawdown, best downside capture.
    core_df / downside_df: index = fund names, columns = metric names.
    """
    all_funds = set(fund_names)

    def _safe_idxmax(series):
        try:
            return series.astype(float).idxmax()
        except Exception:
            return fund_names[0]

    def _safe_idxmin(series):
        try:
            return series.astype(float).idxmin()
        except Exception:
            return fund_names[0]

    def _safe_val(df, metric, fund, decimals=3):
        try:
            return _fmt(float(df.loc[fund, metric]), decimals)
        except Exception:
            return '-'

    try:
        t_funds     = topsis_df.loc[topsis_df.index.isin(all_funds)]
        top_topsis  = t_funds['Rank'].idxmin() if not t_funds.empty else fund_names[0]
        top_topsis_s = _fmt(float(t_funds.loc[top_topsis, 'Score']), 3) if top_topsis in t_funds.index else '-'
    except Exception:
        top_topsis, top_topsis_s = fund_names[0], '-'

    try:
        y_funds   = yuan_df.loc[yuan_df.index.isin(all_funds)]
        top_yuan  = y_funds['Rank'].idxmin() if not y_funds.empty else fund_names[0]
        top_yuan_s = _fmt(float(y_funds.loc[top_yuan, 'Score']), 4) if top_yuan in y_funds.index else '-'
    except Exception:
        top_yuan, top_yuan_s = fund_names[0], '-'

    best_sharpe = _safe_idxmax(core_df['Sharpe Ratio'])
    best_alpha  = _safe_idxmax(core_df['Alpha (ann. %)'])
    # Max Drawdown is negative; highest value = least severe = best
    best_dd     = _safe_idxmax(downside_df['Max Drawdown (%)'])
    # Downside Capture: lower is better
    best_dc     = _safe_idxmin(downside_df['Downside Capture (%)'])

    items = [
        ('TOP TOPSIS',         top_topsis,  f'Score: {top_topsis_s}'),
        ('TOP YUAN & YUAN',    top_yuan,    f'Score: {top_yuan_s}'),
        ('BEST SHARPE',        best_sharpe, f'Sharpe: {_safe_val(core_df, "Sharpe Ratio", best_sharpe)}'),
        ('BEST ALPHA',         best_alpha,  f'Alpha: {_safe_val(core_df, "Alpha (ann. %)", best_alpha, 2)}%'),
        ('BEST DRAWDOWN',      best_dd,     f'Max DD: {_safe_val(downside_df, "Max Drawdown (%)", best_dd, 1)}%'),
        ('LOWEST DN CAPTURE',  best_dc,     f'Dn Cap: {_safe_val(downside_df, "Downside Capture (%)", best_dc, 1)}%'),
    ]

    n      = len(items)
    col_w  = USABLE_W / n
    header = [it[0] for it in items]
    values = [it[1] for it in items]
    subs   = [it[2] for it in items]

    t = Table([header, values, subs], colWidths=[col_w] * n, hAlign='LEFT')
    t.setStyle(TableStyle([
        # Header row
        ('BACKGROUND',    (0, 0), (-1, 0), C_TEAL),
        ('TEXTCOLOR',     (0, 0), (-1, 0), C_TEAL_LT),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 6.5),
        ('ALIGN',         (0, 0), (-1, 0), 'CENTER'),
        ('TOPPADDING',    (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        # Value row
        ('BACKGROUND',    (0, 1), (-1, 1), C_WHITE),
        ('TEXTCOLOR',     (0, 1), (-1, 1), C_INK),
        ('FONTNAME',      (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 1), (-1, 1), 9.5),
        ('ALIGN',         (0, 1), (-1, 1), 'CENTER'),
        ('TOPPADDING',    (0, 1), (-1, 1), 7),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 3),
        # Sub row
        ('BACKGROUND',    (0, 2), (-1, 2), C_WHITE),
        ('TEXTCOLOR',     (0, 2), (-1, 2), C_TEAL),
        ('FONTNAME',      (0, 2), (-1, 2), 'Helvetica'),
        ('FONTSIZE',      (0, 2), (-1, 2), 7.5),
        ('ALIGN',         (0, 2), (-1, 2), 'CENTER'),
        ('TOPPADDING',    (0, 2), (-1, 2), 1),
        ('BOTTOMPADDING', (0, 2), (-1, 2), 7),
        # Borders
        ('BOX',           (0, 0), (-1, -1), 0.75, C_TEAL),
        ('INNERGRID',     (0, 0), (-1, -1), 0.25, C_BORDER),
        ('LINEBELOW',     (0, 0), (-1, 0), 0.5, C_TEAL),
    ]))
    return [t, Spacer(1, 5 * mm)]


# ── Main function ──────────────────────────────────────────────────────────────

def generate_pdf(
    fund_names: List[str],
    sample_period: str,
    core_metrics_df: pd.DataFrame,
    downside_df: pd.DataFrame,
    topsis_ranking: pd.DataFrame,
    yuan_ranking: pd.DataFrame,
    combined_df: pd.DataFrame,
    sensitivity_table: pd.DataFrame,
) -> bytes:

    global _REPORT_DATE
    _REPORT_DATE = date.today().strftime('%d %B %Y')

    S = _styles()
    buffer = BytesIO()
    n_funds = len(fund_names)

    # ── Frames & document ──────────────────────────────────────────────────
    doc = BaseDocTemplate(buffer, pagesize=A4,
        leftMargin=MARGIN_H, rightMargin=MARGIN_H,
        topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOT)

    cover_frame = Frame(MARGIN_H, 20 * mm, USABLE_W, PAGE_H - 78 * mm,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id='cover')
    inner_frame = Frame(MARGIN_H, MARGIN_BOT, USABLE_W,
        PAGE_H - MARGIN_TOP - MARGIN_BOT,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id='inner')

    doc.addPageTemplates([
        PageTemplate(id='Cover', frames=[cover_frame], onPage=_draw_cover),
        PageTemplate(id='Inner', frames=[inner_frame], onPage=_draw_inner),
    ])

    story = []

    # ═══════════════════════════════════════════════════════════════════════
    # COVER
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph('Investment Fund<br/>Comparative Analysis',
                            S['cover_title']))
    story.append(Paragraph('Quantitative Multi-Criteria Ranking Report',
                            S['cover_sub']))
    story.append(HRFlowable(width='100%', thickness=0.5, color=C_BORDER,
                             spaceAfter=6))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(f'<b>Date Prepared:</b>  {_REPORT_DATE}',
                            S['cover_meta']))
    story.append(Paragraph(f'<b>Sample Period:</b>  {sample_period}',
                            S['cover_meta']))
    story.append(Paragraph(f'<b>Funds Analysed:</b>  {", ".join(fund_names)}',
                            S['cover_meta']))
    story.append(Paragraph(
        '<b>Methodologies:</b>  TOPSIS (Hwang &amp; Yoon, 1981)  '
        '\u00b7  Yuan &amp; Yuan Eigenvector (2023)',
        S['cover_meta']))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(
        'This report presents a systematic, quantitative comparison of '
        'emerging-market equity funds using two independent multi-criteria '
        'decision analysis (MCDA) frameworks. All rankings are derived from '
        'historical return data and should be interpreted alongside the full '
        'methodology and disclaimer sections contained herein.',
        S['cover_body']))

    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ═══════════════════════════════════════════════════════════════════════
    story += _toc(S)

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 01 -- EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    story += _section_hdr('Section 01', 'Executive Summary', S)

    # 6-metric key findings strip
    story += _key_findings(fund_names, topsis_ranking, yuan_ranking,
                           core_metrics_df, downside_df, S)

    story.append(Paragraph(
        f'This report evaluates <b>{n_funds} funds</b> across a comprehensive '
        'set of performance, risk-adjusted return, and downside-protection '
        'metrics over the period <b>' + sample_period + '</b>. Two independent '
        'multi-criteria ranking methodologies -- TOPSIS and the Yuan &amp; Yuan '
        'eigenvector method -- are applied to deliver robust, cross-validated '
        'rankings. A five-scheme sensitivity analysis tests the stability of '
        'results under alternative weight allocations.',
        S['body']))

    story.append(Paragraph('Funds in Scope', S['subsection']))
    fund_data = [['#', 'Fund Name']]
    for i, fn in enumerate(fund_names, 1):
        fund_data.append([str(i), fn])
    fund_t = Table(fund_data, colWidths=[30, USABLE_W - 30], hAlign='LEFT')
    fund_t.setStyle(TableStyle(_base_cmds(len(fund_names))))
    story.append(KeepTogether([fund_t]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 02 -- PERFORMANCE METRICS
    # ═══════════════════════════════════════════════════════════════════════
    story += _section_hdr('Section 02', 'Performance Metrics', S)

    story.append(Paragraph('Core Risk-Adjusted Metrics', S['subsection']))
    story.append(Paragraph(
        'Annualised return, volatility, Sharpe ratio, information ratio, '
        'and regression-based metrics computed over the full sample period. '
        'Risk-free rate: 5% p.a. (annualised). Benchmark: MSCI EM (EEM proxy).',
        S['body']))
    story += _color_legend(S)

    core_row_data = _metrics_as_rows(core_metrics_df, label_col='Metric')
    n_fund_cols   = len(core_metrics_df.index)
    core_colw     = _metric_colw(n_fund_cols, label_w=max(120.0, USABLE_W * 0.30))
    story.append(KeepTogether([
        _color_metrics_table(core_row_data, core_colw),
        Spacer(1, 4 * mm),
    ]))

    story.append(Paragraph('Downside Risk Metrics', S['subsection']))
    story.append(Paragraph(
        'Tail-risk characteristics including maximum drawdown, drawdown '
        'duration, Sortino ratio, Calmar ratio, and up/down capture ratios '
        'relative to the benchmark. ESG and cost metrics are included.',
        S['body']))
    story += _color_legend(S)

    down_row_data = _metrics_as_rows(downside_df, label_col='Metric')
    n_down_cols   = len(downside_df.index)
    down_colw     = _metric_colw(n_down_cols, label_w=max(120.0, USABLE_W * 0.30))
    story.append(KeepTogether([
        _color_metrics_table(down_row_data, down_colw),
        Spacer(1, 4 * mm),
    ]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 03 -- RANKINGS
    # ═══════════════════════════════════════════════════════════════════════
    story += _section_hdr('Section 03', 'Fund Rankings', S)

    # Filter topsis_ranking to fund names only
    all_fund_names_set = set(fund_names) | {'Benchmark'}
    topsis_funds = topsis_ranking.loc[
        topsis_ranking.index.isin(all_fund_names_set)
    ].copy().sort_values('Rank')

    # TOPSIS
    story.append(Paragraph('TOPSIS Rankings', S['subsection']))
    story.append(Paragraph(
        'Closeness coefficients range 0 to 1; a higher value indicates '
        'superior proximity to the ideal solution across all weighted criteria. '
        '<font color="#1A7A4A"><b>Rank 1 highlighted in green</b></font>; '
        '<font color="#C0392B">last rank in red</font>.',
        S['body']))

    t_data = [['Fund', 'Closeness Coefficient', 'Rank']]
    t_colw = [USABLE_W * 0.55, USABLE_W * 0.30, USABLE_W * 0.15]

    if topsis_funds.empty:
        t_data.append(['-', '-', '-'])
        story.append(KeepTogether([
            _ranking_table(t_data, t_colw, 0),
            Spacer(1, 5 * mm),
        ]))
    else:
        topsis_funds = topsis_funds.copy()
        topsis_funds['_display_rank'] = range(1, len(topsis_funds) + 1)
        n_t = len(topsis_funds)
        for idx, row in topsis_funds.iterrows():
            score = row.get('Score', row.iloc[0])
            rank  = int(row['_display_rank'])
            t_data.append([str(idx), _fmt(score, 4), str(rank)])
        story.append(KeepTogether([
            _ranking_table(t_data, t_colw, n_t),
            Spacer(1, 5 * mm),
        ]))

    # Yuan & Yuan
    story.append(Paragraph('Yuan &amp; Yuan Rankings', S['subsection']))
    story.append(Paragraph(
        'Eigenvector-derived scores from pairwise competitive comparisons '
        'across all criteria. Scores form a ratio scale invariant to linear '
        'transformations of the underlying data.',
        S['body']))

    yuan_funds = yuan_ranking.loc[
        yuan_ranking.index.isin(all_fund_names_set)
    ].copy().sort_values('Rank')

    y_data = [['Fund', 'Eigenvector Score', 'Rank']]
    y_colw = [USABLE_W * 0.55, USABLE_W * 0.30, USABLE_W * 0.15]
    n_y    = len(yuan_funds)
    for idx, row in yuan_funds.iterrows():
        score = row.get('Score', row.iloc[0])
        rank  = int(row.get('Rank', row.iloc[-1]))
        y_data.append([str(idx), _fmt(score, 6), str(rank)])
    story.append(KeepTogether([
        _ranking_table(y_data, y_colw, n_y),
        Spacer(1, 5 * mm),
    ]))

    # Combined comparison
    story.append(Paragraph('Combined Ranking Comparison', S['subsection']))
    story.append(Paragraph(
        'Side-by-side view of both methodologies. '
        '"Agree" = both methods assigned the same rank; '
        '"Differ" = outcome is methodology-sensitive.',
        S['body']))

    c_data = [['Fund', 'TOPSIS Score', 'TOPSIS Rank',
               'Yuan Score', 'Yuan Rank', 'Consensus']]
    c_colw = [
        USABLE_W * 0.30,
        USABLE_W * 0.15,
        USABLE_W * 0.12,
        USABLE_W * 0.18,
        USABLE_W * 0.12,
        USABLE_W * 0.13,
    ]
    for idx, row in combined_df.iterrows():
        try:
            ts = float(row.get('TOPSIS Score', row.iloc[0]))
            tr = int(row.get('TOPSIS Rank',  row.iloc[1]))
        except (ValueError, TypeError):
            ts, tr = float('nan'), 0
        try:
            ys = float(row.get('Yuan Score', row.iloc[2]))
            yr = int(row.get('Yuan Rank',   row.iloc[3]))
        except (ValueError, TypeError):
            ys, yr = float('nan'), 0

        t_match    = topsis_funds.loc[topsis_funds.index == idx]
        tr_display = int(t_match['_display_rank'].iloc[0]) \
                     if not t_match.empty else tr
        agree = 'Agree' if tr_display == yr else 'Differ'
        c_data.append([str(idx), _fmt(ts, 4), str(tr_display),
                        _fmt(ys, 6), str(yr), agree])

    # Colour "Agree" rows teal-tinted
    agree_cmds = _base_cmds(len(c_data) - 1)
    for i, row in enumerate(c_data[1:], start=1):
        if row[-1] == 'Agree':
            agree_cmds.append(('TEXTCOLOR', (5, i), (5, i), C_GREEN2))
            agree_cmds.append(('FONTNAME',  (5, i), (5, i), 'Helvetica-Bold'))
        else:
            agree_cmds.append(('TEXTCOLOR', (5, i), (5, i), C_RED2))
    c_t = Table(c_data, colWidths=c_colw, hAlign='LEFT')
    c_t.setStyle(TableStyle(agree_cmds))

    story.append(KeepTogether([
        c_t,
        Spacer(1, 2 * mm),
    ]))
    story.append(Paragraph(
        'Agree = both methods assigned the same rank.  '
        'Differ = rankings diverge between methodologies.',
        S['note']))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 04 -- SENSITIVITY
    # ═══════════════════════════════════════════════════════════════════════
    story += _section_hdr('Section 04', 'Sensitivity Analysis', S)
    story.append(Paragraph(
        'Rankings are recomputed under five alternative metric-weight schemes '
        'to assess robustness. Consistent ranks across all schemes signal a '
        'reliable result; variation indicates sensitivity to the weighting '
        'assumptions. Each cell shows the TOPSIS score and rank in parentheses.',
        S['body']))

    sa_fund_cols  = list(sensitivity_table.columns)
    sa_colw_label = USABLE_W * 0.30
    sa_col_w      = (USABLE_W - sa_colw_label) / max(len(sa_fund_cols), 1)
    sa_colw       = [sa_colw_label] + [sa_col_w] * len(sa_fund_cols)

    sa_data = [['Weight Scheme'] + sa_fund_cols]
    for scheme in sensitivity_table.index:
        row = [str(scheme)]
        for fund in sa_fund_cols:
            v = sensitivity_table.loc[scheme, fund]
            row.append(str(v) if pd.notna(v) else '-')
        sa_data.append(row)

    story.append(KeepTogether([
        _make_table(sa_data, sa_colw),
        Spacer(1, 4 * mm),
    ]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 05 -- METHODOLOGY
    # ═══════════════════════════════════════════════════════════════════════
    story += _section_hdr('Section 05', 'Methodology', S)

    story.append(Paragraph('TOPSIS', S['subsection']))
    story.append(Paragraph(
        'The Technique for Order Preference by Similarity to Ideal Solution '
        '(Hwang &amp; Yoon, 1981) ranks alternatives in five steps: '
        '(1) Construct a normalised decision matrix using vector normalisation. '
        '(2) Apply criterion weights to produce the weighted normalised matrix. '
        '(3) Identify the positive ideal solution (PIS) and negative ideal '
        'solution (NIS) for each criterion. '
        '(4) Compute each alternative\'s Euclidean distance from the PIS '
        '(D\u207a) and NIS (D\u207b). '
        '(5) Derive the closeness coefficient C* = D\u207b / (D\u207a + D\u207b); '
        'a higher C* reflects superior overall performance.',
        S['body']))

    story.append(Paragraph('Yuan &amp; Yuan Eigenvector Method', S['subsection']))
    story.append(Paragraph(
        'Yuan &amp; Yuan (2023) construct an n \u00d7 n pairwise competition '
        'matrix C where element c(i, j) reflects the competitive performance '
        'of fund i against fund j across all selected metrics simultaneously. '
        'The principal eigenvector of C is extracted via power iteration; '
        'the resulting score vector provides a ratio-scale ranking invariant '
        'to positive linear transformations of the input data. Convergence '
        'is typically achieved in fewer than 20 iterations.',
        S['body']))

    story.append(Paragraph('Weighting Structure', S['subsection']))
    story.append(Paragraph(
        'Criterion weights are allocated across five pillars: Returns, '
        'Risk-Adjusted Performance, Risk &amp; Drawdown, Costs, and ESG. '
        'Within each pillar, weight is distributed equally across its component '
        'metrics. The default allocation is 30 / 30 / 20 / 10 / 10 '
        '(Returns / Risk-Adj / Risk-DD / Costs / ESG).',
        S['body']))

    story.append(Paragraph('Performance Metrics Glossary', S['subsection']))
    mdef = [
        ['Metric', 'Definition'],
        ['Ann. Return (%)',       'Geometric mean of monthly log returns annualised: (exp(\u0305r \u00d7 12) \u2212 1) \u00d7 100'],
        ['Ann. Volatility (%)',   'Monthly return standard deviation scaled by \u221a12, expressed as %'],
        ['Sharpe Ratio',          '(Annualised excess return) / (Annualised volatility); Rf = 5% p.a.'],
        ['Alpha (ann. %)',        "Annualised Jensen's alpha from OLS regression on benchmark excess returns"],
        ['Beta',                  'Market beta from OLS regression on the benchmark; 1.0 = full benchmark exposure'],
        ['R\u00b2',               'Coefficient of determination from the OLS regression; higher = more benchmark-driven'],
        ['Sortino Ratio',         'Annualised excess return / downside semi-deviation (negative returns only)'],
        ['Treynor Ratio',         'Annualised excess return / beta; reward per unit of market risk'],
        ['Information Ratio',     'Annualised active return / tracking error; consistency of outperformance'],
        ['Tracking Error (%)',    'Standard deviation of active returns (fund minus benchmark), annualised'],
        ['Max Drawdown (%)',      'Largest peak-to-trough decline in the cumulative log-return NAV; stored as negative'],
        ['Max DD Duration (mths)','Longest consecutive period (months) below the prior NAV peak'],
        ['Calmar Ratio',          'Annualised return / |Max Drawdown| (higher = better risk-adjusted outcome)'],
        ['Upside Capture (%)',    'Fund / benchmark return in benchmark up-months; >100% = outperforms in rallies'],
        ['Downside Capture (%)', 'Fund / benchmark return in benchmark down-months; <100% = limits losses better'],
        ['OCF',                   'Ongoing Charges Figure; total annual cost as % of NAV'],
        ['ESG Globe Rating',      'Morningstar sustainability rating 1\u20135 globes (5 = most sustainable)'],
        ['Carbon Risk Score',     'Sustainalytics carbon risk score (lower = less exposed to transition risk)'],
    ]
    md_colw = [USABLE_W * 0.30, USABLE_W * 0.70]
    md_cmds = _base_cmds(len(mdef) - 1)
    md_cmds += [('ALIGN', (1, 1), (1, -1), 'LEFT')]
    md_t = Table(mdef, colWidths=md_colw, hAlign='LEFT')
    md_t.setStyle(TableStyle(md_cmds))
    story.append(KeepTogether([md_t, Spacer(1, 4 * mm)]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 06 -- DISCLAIMER
    # ═══════════════════════════════════════════════════════════════════════
    story += _section_hdr('Section 06', 'Disclaimer &amp; Data Sources', S)
    disclaimers = [
        '<b>Data Source.</b>  Historical pricing data are sourced via Yahoo '
        'Finance. Where UK-domiciled funds are unavailable on Yahoo Finance, '
        'exchange-traded fund (ETF) proxies with equivalent investment mandates '
        'are used as a practical approximation.',
        '<b>For Informational Purposes Only.</b>  All analysis, rankings, and '
        'content in this report are produced solely for educational and research '
        'purposes. Nothing herein constitutes investment advice, a recommendation, '
        'or an offer or solicitation to buy or sell any security or financial '
        'instrument.',
        '<b>Past Performance.</b>  Past performance is not a reliable indicator '
        'of future results. The value of investments and any income from them can '
        'fall as well as rise. Investors may not recover the amount originally '
        'invested.',
        '<b>No Warranty.</b>  While reasonable care has been taken to ensure '
        'accuracy, no warranty is given as to the completeness or accuracy of the '
        'information contained herein. The Fund Analysis Engine and its authors '
        'accept no liability for any loss arising from reliance on this report.',
        '<b>Benchmark.</b>  Where referenced, the benchmark is the MSCI Emerging '
        'Markets Index total return (USD), proxied via the iShares MSCI Emerging '
        'Markets ETF (EEM).',
        '<b>Academic References.</b>  Hwang, C.-L. and Yoon, K. (1981). '
        '<i>Multiple Attribute Decision Making: Methods and Applications</i>. '
        'Springer. \u00b7  Yuan, Y. and Yuan, J. (2023). A new approach to '
        'fund performance evaluation. <i>Journal of Portfolio Management</i>.',
    ]
    for p in disclaimers:
        story.append(Paragraph(p, S['disclaimer']))
        story.append(Spacer(1, 2.5 * mm))

    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=C_BORDER,
                             spaceAfter=4))
    story.append(Paragraph(
        f'Fund Analysis Engine  \u00b7  Report generated {_REPORT_DATE}'
        '  \u00b7  CONFIDENTIAL',
        S['note']))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
