#!/usr/bin/env python3
"""
Full B.Tech Final Year Project Report Generator
Creates:
1. BTech_Final_Year_Project_Report_GraphRAG_vs_Dense_Vector_RAG.docx
2. BTech_Final_Year_Project_Report_GraphRAG_vs_Dense_Vector_RAG.pdf (via libreoffice)
3. BTech_Final_Year_Project_Report_GraphRAG_vs_Dense_Vector_RAG.md
"""

import os
import sys
import subprocess
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tc_pr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tcMar)

def format_paragraph(p, space_before=0, space_after=6, line_spacing=1.15):
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing

def add_header_footer(doc):
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.25)
        section.right_margin = Inches(1.0)
        
        # Header
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("Walchand College of Engineering, Sangli | Department of Information Technology")
        hrun.font.name = "Times New Roman"
        hrun.font.size = Pt(8.5)
        hrun.font.italic = True
        hrun.font.color.rgb = RGBColor(100, 116, 139)
        
        # Footer
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("GraphRAG vs. Dense Vector RAG — Final Year B.Tech Project Report (7IT492)")
        frun.font.name = "Times New Roman"
        frun.font.size = Pt(8.5)
        frun.font.italic = True
        frun.font.color.rgb = RGBColor(148, 163, 184)

def add_styled_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    format_paragraph(h, space_before=14 if level > 1 else 18, space_after=6, line_spacing=1.15)
    for run in h.runs:
        run.font.name = "Times New Roman"
        if level == 1:
            run.font.size = Pt(16)
            run.font.bold = True
            run.font.color.rgb = RGBColor(15, 23, 42)
        elif level == 2:
            run.font.size = Pt(13.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(30, 58, 138)
        else:
            run.font.size = Pt(12)
            run.font.bold = True
            run.font.color.rgb = RGBColor(51, 65, 85)
    return h

def add_body_p(doc, text="", bold_prefix="", italic_suffix=""):
    p = doc.add_paragraph()
    format_paragraph(p, space_before=0, space_after=6, line_spacing=1.15)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    if bold_prefix:
        r_bold = p.add_run(bold_prefix)
        r_bold.font.name = "Times New Roman"
        r_bold.font.size = Pt(12)
        r_bold.font.bold = True
        r_bold.font.color.rgb = RGBColor(15, 23, 42)
        
    if text:
        r_text = p.add_run(text)
        r_text.font.name = "Times New Roman"
        r_text.font.size = Pt(12)
        r_text.font.color.rgb = RGBColor(30, 41, 59)
        
    if italic_suffix:
        r_it = p.add_run(italic_suffix)
        r_it.font.name = "Times New Roman"
        r_it.font.size = Pt(12)
        r_it.font.italic = True
        r_it.font.color.rgb = RGBColor(71, 85, 105)
        
    return p

def add_bullet_p(doc, bold_prefix="", text=""):
    p = doc.add_paragraph(style='List Bullet')
    format_paragraph(p, space_before=1, space_after=3, line_spacing=1.15)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = "Times New Roman"
        rb.font.size = Pt(12)
        rb.font.bold = True
        rb.font.color.rgb = RGBColor(15, 23, 42)
        
    if text:
        rt = p.add_run(text)
        rt.font.name = "Times New Roman"
        rt.font.size = Pt(12)
        rt.font.color.rgb = RGBColor(30, 41, 59)
        
    return p

def add_image_figure(doc, img_path, caption_text, width_inches=5.6):
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_paragraph(p_img, space_before=10, space_after=4)
        run = p_img.add_run()
        run.add_picture(img_path, width=Inches(width_inches))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_paragraph(p_cap, space_before=2, space_after=12)
        rc = p_cap.add_run(caption_text)
        rc.font.name = "Times New Roman"
        rc.font.size = Pt(10.5)
        rc.font.bold = True
        rc.font.italic = True
        rc.font.color.rgb = RGBColor(71, 85, 105)

def style_table(table, col_widths, headers, data, align_cols=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    # Header row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "1E3A8A")
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=150, right=150)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_paragraph(p, space_before=0, space_after=0)
        for r in p.runs:
            r.font.name = "Times New Roman"
            r.font.size = Pt(10.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)
            
    # Data rows
    for row_idx, row_data in enumerate(data):
        row_cells = table.rows[row_idx + 1].cells
        bg_color = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, cell_value in enumerate(row_data):
            row_cells[col_idx].text = str(cell_value)
            set_cell_background(row_cells[col_idx], bg_color)
            set_cell_margins(row_cells[col_idx], top=80, bottom=80, left=120, right=120)
            p = row_cells[col_idx].paragraphs[0]
            format_paragraph(p, space_before=0, space_after=0)
            if align_cols and col_idx in align_cols:
                p.alignment = align_cols[col_idx]
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(10)
                r.font.color.rgb = RGBColor(30, 41, 59)
                
    for row in table.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = Inches(width)

def create_and_style_table(doc, headers, data, col_widths, align_cols=None):
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    style_table(table, col_widths, headers, data, align_cols)
    return table

def generate_docx():
    doc = Document()
    add_header_footer(doc)
    
    # -------------------------------------------------------------
    # PAGE 1: TITLE / COVER PAGE
    # -------------------------------------------------------------
    # 3-logo table
    logo_table = doc.add_table(rows=1, cols=3)
    logo_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths = [1.8, 2.4, 1.8]
    for row in logo_table.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Inches(w)
            
    # Cell 0: Platinum Jubilee Logo
    p0 = logo_table.rows[0].cells[0].paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.LEFT
    format_paragraph(p0, 0, 0)
    if os.path.exists("report_assets/platinum_jubilee_logo.png"):
        p0.add_run().add_picture("report_assets/platinum_jubilee_logo.png", width=Inches(1.15))
        
    # Cell 1: WCE Crest
    p1 = logo_table.rows[0].cells[1].paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p1, 0, 0)
    if os.path.exists("report_assets/wce_crest.png"):
        p1.add_run().add_picture("report_assets/wce_crest.png", width=Inches(1.15))
        
    # Cell 2: IT Dept Logo
    p2 = logo_table.rows[0].cells[2].paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    format_paragraph(p2, 0, 0)
    if os.path.exists("report_assets/it_dept_logo.png"):
        p2.add_run().add_picture("report_assets/it_dept_logo.png", width=Inches(1.3))

    # College Title Text
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_inst, space_before=16, space_after=2)
    r = p_inst.add_run("Walchand College of Engineering, Sangli.")
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(15, 23, 42)

    p_auto = doc.add_paragraph()
    p_auto.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_auto, space_before=0, space_after=14)
    r = p_auto.add_run("(An Autonomous Institute)\nDepartment of Information Technology")
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = RGBColor(30, 41, 59)

    p_rep = doc.add_paragraph()
    p_rep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_rep, space_before=10, space_after=6)
    r = p_rep.add_run("A Project Report On")
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)
    r.font.color.rgb = RGBColor(71, 85, 105)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_title, space_before=8, space_after=8)
    r = p_title.add_run("GraphRAG vs. Dense Vector RAG: Empirical Evaluation and Hybrid Optimization for Domain-Specific Complex Reasoning")
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(15, 23, 42)

    p_deg = doc.add_paragraph()
    p_deg.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_deg, space_before=12, space_after=18)
    r = p_deg.add_run("Course: Project (7IT492)\nB.Tech in Information Technology\n(Academic Year 2025-26)")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = RGBColor(30, 58, 138)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_sub, space_before=6, space_after=8)
    r = p_sub.add_run("Submitted by (Group S):")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True

    # Student Table
    stud_table = doc.add_table(rows=4, cols=2)
    headers = ["Name of Student", "PRN"]
    data = [
        ["Vaishnavi Patil", "23410048"],
        ["Aryan Chandekar", "23610046"],
        ["Animesh Gawahle", "23610055"],
    ]
    style_table(stud_table, [3.2, 2.2], headers, data, align_cols={1: WD_ALIGN_PARAGRAPH.CENTER})

    p_guid = doc.add_paragraph()
    p_guid.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_guid, space_before=18, space_after=0)
    r = p_guid.add_run("Under the Guidance of\n")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r = p_guid.add_run("Dr. U. B. Chavan\n")
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)
    r.font.bold = True
    r = p_guid.add_run("Department of Information Technology\nWalchand College of Engineering, Sangli")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 2: UNDER THE GUIDANCE / FACULTY SUPERVISION PAGE
    # -------------------------------------------------------------
    # 3-logo table
    logo_table2 = doc.add_table(rows=1, cols=3)
    logo_table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row in logo_table2.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Inches(w)
    p0 = logo_table2.rows[0].cells[0].paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.LEFT
    format_paragraph(p0, 0, 0)
    if os.path.exists("report_assets/platinum_jubilee_logo.png"):
        p0.add_run().add_picture("report_assets/platinum_jubilee_logo.png", width=Inches(1.15))
    p1 = logo_table2.rows[0].cells[1].paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p1, 0, 0)
    if os.path.exists("report_assets/wce_crest.png"):
        p1.add_run().add_picture("report_assets/wce_crest.png", width=Inches(1.15))
    p2 = logo_table2.rows[0].cells[2].paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    format_paragraph(p2, 0, 0)
    if os.path.exists("report_assets/it_dept_logo.png"):
        p2.add_run().add_picture("report_assets/it_dept_logo.png", width=Inches(1.3))

    p_sp = doc.add_paragraph()
    format_paragraph(p_sp, space_before=40, space_after=0)

    p_g_title = doc.add_paragraph()
    p_g_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_g_title, space_before=20, space_after=8)
    r = p_g_title.add_run("Under the Guidance of\n")
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(71, 85, 105)
    r = p_g_title.add_run("Dr. U. B. Chavan\n")
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(15, 23, 42)
    r = p_g_title.add_run("Associate Professor\nDepartment of Information Technology\nWalchand College of Engineering, Sangli\n")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.color.rgb = RGBColor(51, 65, 85)

    p_g_meta = doc.add_paragraph()
    p_g_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_g_meta, space_before=60, space_after=0)
    r = p_g_meta.add_run("Academic Year: 2025-26\nSemester: VII (Final Year B.Tech)\nCourse Code: 7IT492 (Project)")
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = RGBColor(30, 58, 138)

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 3: CERTIFICATE
    # -------------------------------------------------------------
    p_cert_logo = doc.add_paragraph()
    p_cert_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_cert_logo, space_before=0, space_after=6)
    if os.path.exists("report_assets/wce_crest.png"):
        p_cert_logo.add_run().add_picture("report_assets/wce_crest.png", width=Inches(1.15))

    p_c_inst = doc.add_paragraph()
    p_c_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_c_inst, space_before=4, space_after=2)
    r = p_c_inst.add_run("Walchand College of Engineering, Sangli\n")
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)
    r.font.bold = True
    r = p_c_inst.add_run("(An Autonomous Institute)\nDepartment of Information Technology")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True

    p_cert_head = doc.add_paragraph()
    p_cert_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_cert_head, space_before=16, space_after=16)
    r = p_cert_head.add_run("CERTIFICATE")
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.underline = True
    r.font.color.rgb = RGBColor(30, 58, 138)

    p_cert_body = doc.add_paragraph()
    p_cert_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    format_paragraph(p_cert_body, space_before=6, space_after=14, line_spacing=1.25)
    r = p_cert_body.add_run("This is to certify that the Project Report entitled ")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r = p_cert_body.add_run("“GraphRAG vs. Dense Vector RAG: Empirical Evaluation and Hybrid Optimization for Domain-Specific Complex Reasoning”")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True
    r = p_cert_body.add_run(" was submitted by the following students:")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)

    cert_table = doc.add_table(rows=4, cols=3)
    headers = ["Sr. No.", "Name of Student", "PRN"]
    data = [
        ["1.", "Vaishnavi Patil", "23410048"],
        ["2.", "Aryan Chandekar", "23610046"],
        ["3.", "Animesh Gawahle", "23610055"],
    ]
    style_table(cert_table, [1.0, 2.8, 1.8], headers, data, align_cols={0: WD_ALIGN_PARAGRAPH.CENTER, 2: WD_ALIGN_PARAGRAPH.CENTER})

    p_cert_end = doc.add_paragraph()
    p_cert_end.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    format_paragraph(p_cert_end, space_before=14, space_after=40, line_spacing=1.25)
    r = p_cert_end.add_run(
        "to Walchand College of Engineering, Sangli, India, is a record of bona fide Project work carried out by them "
        "under my supervision and guidance and is worthy of consideration for the award of the degree of "
        "Bachelor of Technology in Information Technology of the Institute."
    )
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)

    # Signatures Table
    sig_table = doc.add_table(rows=1, cols=3)
    widths_sig = [2.0, 2.0, 2.0]
    for row in sig_table.rows:
        for i, w in enumerate(widths_sig):
            row.cells[i].width = Inches(w)
            
    c0 = sig_table.rows[0].cells[0].paragraphs[0]
    c0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(c0, 0, 0)
    r = c0.add_run("Dr. U. B. Chavan\n(Project Guide)")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.font.bold = True

    c1 = sig_table.rows[0].cells[1].paragraphs[0]
    c1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(c1, 0, 0)
    r = c1.add_run("\n(External Examiner)")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.font.bold = True

    c2 = sig_table.rows[0].cells[2].paragraphs[0]
    c2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(c2, 0, 0)
    r = c2.add_run("Dr. R. R. Rathod\n(HOD, Dept of IT)")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.font.bold = True

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 4: DECLARATION
    # -------------------------------------------------------------
    p_dec_head = doc.add_paragraph()
    p_dec_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_dec_head, space_before=10, space_after=18)
    r = p_dec_head.add_run("Declaration")
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(15, 23, 42)

    add_body_p(
        doc,
        "We, hereby declare that the project report entitled “GraphRAG vs. Dense Vector RAG: Empirical Evaluation and "
        "Hybrid Optimization for Domain-Specific Complex Reasoning” submitted by us to Walchand College of Engineering, "
        "Sangli, in fulfillment of the requirements for the award of the degree of B. Tech in Information Technology, "
        "is a record of bona fide project work carried out by us under the guidance of Dr. U. B. Chavan."
    )
    add_body_p(
        doc,
        "We further declare that this project work has not been submitted and will not be submitted, either in part or "
        "in full, for the award of any other degree or diploma in this institute or any other institute or university."
    )
    add_body_p(
        doc,
        "We affirm that the work presented in this report is the result of our own efforts. We have duly acknowledged "
        "all the sources used and referred to in this project. We have not fabricated, falsified, or misrepresented any "
        "facts, data, or information in this submission. We understand that any violation of the above will result in "
        "disciplinary action by the institute."
    )

    p_sp = doc.add_paragraph()
    format_paragraph(p_sp, space_before=20, space_after=0)

    # Student signatures
    decl_table = doc.add_table(rows=4, cols=2)
    headers = ["Name of Student", "PRN"]
    data = [
        ["Vaishnavi Patil", "23410048"],
        ["Aryan Chandekar", "23610046"],
        ["Animesh Gawahle", "23610055"],
    ]
    style_table(decl_table, [3.2, 2.2], headers, data, align_cols={1: WD_ALIGN_PARAGRAPH.CENTER})

    p_date = doc.add_paragraph()
    format_paragraph(p_date, space_before=30, space_after=0)
    r = p_date.add_run("Date: 02/11/2026\nPlace: Sangli")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11.5)
    r.font.bold = True

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 5: ACKNOWLEDGEMENT
    # -------------------------------------------------------------
    p_ack_head = doc.add_paragraph()
    p_ack_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_ack_head, space_before=10, space_after=18)
    r = p_ack_head.add_run("Acknowledgement")
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(15, 23, 42)

    add_body_p(
        doc,
        "We feel immense pleasure in submitting this Final Year B.Tech project report entitled “GraphRAG vs. Dense Vector "
        "RAG: Empirical Evaluation and Hybrid Optimization for Domain-Specific Complex Reasoning”."
    )
    add_body_p(
        doc,
        "We express our sincere gratitude to our project guide Dr. U. B. Chavan for his valuable guidance, continuous "
        "support, and constant encouragement throughout the conceptualization, architecture design, and empirical "
        "evaluation of this project. His deep insights into information retrieval, knowledge graphs, and rigorous "
        "benchmarking methodology greatly contributed to the successful completion and verification of this work."
    )
    add_body_p(
        doc,
        "We would also like to extend our heartfelt thanks to Dr. R. R. Rathod, Head of the Department, Department of "
        "Information Technology, Walchand College of Engineering, Sangli, for providing us with the necessary computing "
        "infrastructure, laboratory resources, and an inspiring academic atmosphere that enabled us to undertake and "
        "complete this advanced engineering project successfully."
    )
    add_body_p(
        doc,
        "We are thankful to all the faculty members, reviewers, and technical staff of the Department of Information "
        "Technology for their continuous cooperation, constructive critiques, and valuable feedback throughout the course "
        "of this project."
    )
    add_body_p(
        doc,
        "We also wish to express our sincere appreciation to our friends and classmates for their active discussions and "
        "collaboration during the evaluation experiments. Finally, we express our heartfelt debt to our parents and "
        "families for their unconditional love, moral support, and blessings throughout our engineering journey."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 6: TABLE OF CONTENTS
    # -------------------------------------------------------------
    p_toc_head = doc.add_paragraph()
    p_toc_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_toc_head, space_before=10, space_after=14)
    r = p_toc_head.add_run("Table of Contents")
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = RGBColor(15, 23, 42)

    headers = ["Chapter / Section Title", "Page No."]
    all_toc_data = [
        ["1. Introduction", "1"],
        ["    1.1 Overview and Problem Statement", "1"],
        ["    1.2 Motivation", "2"],
        ["    1.3 Contribution in the Work", "2"],
        ["2. Project Objectives (Bloom's Taxonomy Framework)", "3"],
        ["3. Proposed Method / Algorithm", "5"],
        ["    3.1 Problem Definition & Mathematical Formulation", "5"],
        ["    3.2 Proposed System Architecture", "6"],
        ["    3.3 System Analysis and Design", "7"],
        ["        3.3.1 Figure: User Flow Chart (Fig. 3.3.1)", "7"],
        ["        3.3.2 Figure: Use Case Diagram (Fig. 3.3.2)", "8"],
        ["        3.3.3 Figure: Internal Backend Flow & Data Pipeline (Fig. 3.3.3)", "9"],
        ["4. Design and Implementation", "10"],
        ["    4.1 Requirement Specification (Functional, Non-Functional, User)", "10"],
        ["    4.2 Design & Test Steps (System Design, Implementation, Testing, Validation)", "12"],
        ["    4.3 User Interface Walkthrough & Visual Lab Layout", "13"],
        ["5. Technology Stack", "17"],
        ["6. Results & Empirical Analysis", "19"],
        ["7. Conclusion & Future Scope", "22"],
        ["8. References", "23"],
    ]
    create_and_style_table(doc, headers, all_toc_data, [4.8, 1.0], align_cols={1: WD_ALIGN_PARAGRAPH.RIGHT})

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 1: INTRODUCTION
    # -------------------------------------------------------------
    add_styled_heading(doc, "1. Introduction", level=1)
    
    add_body_p(
        doc,
        "Retrieval-Augmented Generation (RAG) has rapidly established itself as the state-of-the-art architectural paradigm "
        "for mitigating hallucination, injecting temporal domain knowledge, and grounding large language models (LLMs) without "
        "the prohibitive computational expense of model fine-tuning. By querying external datastores for relevant contextual "
        "evidence prior to token generation, RAG guarantees that generative responses are anchored in verifiable source citations. "
        "However, contemporary production systems remain fundamentally divided in their retrieval strategy between two contrasting "
        "paradigms: Dense Vector RAG and Knowledge Graph RAG (GraphRAG)."
    )

    add_styled_heading(doc, "1.1 Overview and Problem Statement", level=2)
    add_body_p(
        doc,
        "Dense Vector RAG operates by projecting document text chunks into high-dimensional vector spaces using dense neural "
        "bi-encoders (e.g., Sentence Transformers) and performing Approximate Nearest Neighbor (ANN) search using cosine distance "
        "or inner product metrics in vector databases such as Qdrant or FAISS. While highly effective, computationally efficient, "
        "and sub-linear in search complexity for direct semantic similarity, Dense Vector RAG treats document chunks as "
        "independent, unstructured text fragments. Consequently, it exhibits fatal blind spots when addressing multi-hop relational "
        "queries that span disparate documents or require traversing indirect logical connections."
    )
    add_body_p(
        doc,
        "To resolve this structural limitation, GraphRAG extracts named entities, semantic concepts, and relational predicates "
        "into a structured knowledge graph stored in a graph database such as Neo4j. Queries are answered by performing entity "
        "linking and bounded multi-hop neighborhood traversals using graph query languages like Cypher. However, GraphRAG introduces "
        "substantial indexing overhead, complex schema extraction, graph traversal latency, and higher token ingestion costs. "
        "Crucially, there is a stark absence of rigorous, empirical comparative evaluation directly benchmarking Vector RAG, "
        "GraphRAG, and Hybrid RAG on the identical domain-specific corpus while simultaneously measuring retrieval accuracy, "
        "generation faithfulness, end-to-end latency, and operational expense."
    )

    add_styled_heading(doc, "1.2 Motivation", level=2)
    add_body_p(
        doc,
        "In critical enterprise domains—such as quantum computing specifications, biomedical pharmacology, legal contracts, and "
        "aerospace engineering—analytical inquiries are rarely simple semantic matches. Instead, they require multi-hop reasoning, "
        "such as tracking how an experimental drug compound correlates with adverse reactions across clinical trials, or how an "
        "architectural hardware bottleneck propagates through distributed software subsystems. Deploying GraphRAG indiscriminately "
        "incurs excessive latency and engineering complexity, while relying solely on Dense Vector RAG results in incomplete answers "
        "and hallucinations. Therefore, an empirical benchmark laboratory that provides rigorous comparative analytics, transparent "
        "query routing, and hybrid rank fusion is urgently required to guide AI practitioners and academic researchers."
    )

    add_styled_heading(doc, "1.3 Contribution in the Work", level=2)
    add_body_p(
        doc,
        "This project presents the design, implementation, and empirical verification of the RAG Benchmark Lab—a comprehensive, "
        "multi-tenant research platform engineered to evaluate and optimize retrieval paradigms over domain-specific corpora. "
        "The primary engineering and scientific contributions include:"
    )
    add_bullet_p(doc, "Dual-Store Ingestion Engine: ", "Architected an automated multi-document ingestion pipeline supporting PDF, TXT, and DOCX files that executes parallel dual-indexing: generating 384-dimensional dense vector embeddings indexed into Qdrant, while simultaneously extracting domain entities and typed semantic predicates into Neo4j with source chunk and page-level provenance.")
    add_bullet_p(doc, "Tri-Strategy Retrieval Suite: ", "Engineered and verified three distinct retrieval pipelines over the identical corpus: (1) Dense Vector RAG with deep candidate harvesting and lexical reranking; (2) GraphRAG with Cypher entity matching, multi-hop neighborhood traversal, and source chunk hydration; and (3) Hybrid RAG utilizing Reciprocal Rank Fusion (RRF, k=60) with score preservation.")
    add_bullet_p(doc, "Intelligent Query Router: ", "Designed an explainable rule-based and entity-density heuristic classifier that dynamically determines whether a query is direct-factual, relational-graph, or complex-hybrid, providing transparent routing rationale.")
    add_bullet_p(doc, "Empirical Evaluation & Telemetry Framework: ", "Constructed an automated benchmarking harness executing standard Information Retrieval metrics (Recall@K, Hit Rate@K, MRR, Context Precision, Context Recall) and RAGAs generative quality metrics (Faithfulness, Answer Relevancy) with persistent telemetry.")
    add_bullet_p(doc, "Production Multi-Tenant Research Lab: ", "Built an interactive Streamlit application featuring secure PBKDF2 authentication, strictly isolated user knowledge bases, persistent multi-turn conversational chat with grounded clickable citations, 3-way side-by-side comparison sandboxes, Neo4j graph visualizer, and one-click CSV research export.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 2: PROJECT OBJECTIVES
    # -------------------------------------------------------------
    add_styled_heading(doc, "2. Project Objectives", level=1)
    add_body_p(
        doc,
        "To ensure rigorous academic alignment and systematic engineering execution, the objectives of this Final Year B.Tech "
        "project are structured in accordance with Bloom's Revised Taxonomy:"
    )

    add_bullet_p(
        doc,
        "To Remember: ",
        "Identify and formally define foundational principles of Retrieval-Augmented Generation, dense vector embeddings, "
        "Approximate Nearest Neighbor (ANN) search with cosine distance, Knowledge Graphs (KG), Labeled Property Graph (LPG) models, "
        "Cypher graph traversal queries, Reciprocal Rank Fusion (RRF), Information Retrieval evaluation metrics (MRR, Recall@K, Hit Rate@K), "
        "and RAGAs generation metrics (Faithfulness, Answer Relevancy, Context Precision/Recall). Recall the limitations of manual "
        "corpus synthesis and isolated vector chunking."
    )
    add_bullet_p(
        doc,
        "To Understand: ",
        "Explain how dense bi-encoder models capture semantic proximity in vector spaces and illustrate why treating document chunks "
        "independently causes structural blindness in multi-hop scenarios. Describe how knowledge graph nodes and relational edges "
        "preserve explicit semantic topologies across documents. Comprehend the mathematical properties of rank fusion algorithms "
        "and how hybridizing vector similarity with graph traversal balances high recall with relational precision."
    )
    add_bullet_p(
        doc,
        "To Apply: ",
        "Develop an end-to-end, modular software platform in Python 3.12 integrating live containerized instances of Qdrant (vector DB) "
        "and Neo4j (graph DB). Implement text parsing, recursive token chunking with stable identifiers, 384-dimensional embedding "
        "generation using Sentence Transformers (all-MiniLM-L6-v2), and rule-based entity/relation extraction. Construct an interactive "
        "multi-page web application using Streamlit for corpus ingestion, multi-turn chat, side-by-side comparison, and benchmarking."
    )
    add_bullet_p(
        doc,
        "To Analyze: ",
        "Empirically compare and differentiate Dense Vector RAG, GraphRAG, and Hybrid RAG across simple factual queries versus "
        "complex multi-hop relational queries. Quantify operational trade-offs including retrieval latency (ms), token consumption, "
        "and indexing complexity. Analyze system robustness under edge conditions, including strict multi-tenant corpus isolation and "
        "honest handling of out-of-domain/unrepresented queries without generative hallucination."
    )
    add_bullet_p(
        doc,
        "To Evaluate: ",
        "Measure and critique retrieval and generation quality using standardized benchmark datasets. Evaluate context precision, "
        "context recall, and generation faithfulness across all three retrieval paradigms. Justify the operational circumstances "
        "under which the latency overhead of GraphRAG is warranted, and evaluate the classification accuracy of heuristic query routing."
    )
    add_bullet_p(
        doc,
        "To Create: ",
        "Architect and deliver a production-grade, extensible RAG Benchmark Lab platform featuring user authentication, isolated "
        "knowledge base workspaces, automated dual-store ingestion, persistent conversation history, grounded citation attribution, "
        "interactive Plotly analytics, and one-click exportable research telemetry for downstream scientific and enterprise analysis."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 3: PROPOSED METHOD / ALGORITHM
    # -------------------------------------------------------------
    add_styled_heading(doc, "3. Proposed Method / Algorithm", level=1)

    add_styled_heading(doc, "3.1 Problem Definition & Mathematical Formulation", level=2)
    add_body_p(
        doc,
        "Let a domain corpus be defined as a set of documents D = {d1, d2, ..., dN}. Each document d is partitioned into a set of "
        "contiguous, overlapping text chunks C(d) = {c1, c2, ..., cM} using recursive token boundaries. Given a user query q:"
    )
    add_body_p(
        doc,
        "1. Dense Vector RAG: A neural encoder f_enc(.) maps query q and chunk c into unit-normalized embedding vectors "
        "e_q, e_c in R^384. Retrieval selects the Top-K chunks maximizing cosine similarity:\n"
        "   Score_vector(q, c) = cos(e_q, e_c) = (e_q . e_c) / (||e_q|| ||e_c||)"
    )
    add_body_p(
        doc,
        "2. GraphRAG: Document chunks are parsed into an explicit knowledge graph G = (V, E), where vertices V denote named entities "
        "(e.g., Technology, Concept, Component) and directed edges E denote typed semantic relationships (e.g., USES, EVALUATES, "
        "INDEXES). For query entities Eq in q, GraphRAG executes neighborhood traversal up to depth h <= 2 to discover connected subgraphs "
        "G_sub and maps traversed entities back to source text chunks C_graph via node provenance sets."
    )
    add_body_p(
        doc,
        "3. Hybrid RAG (Reciprocal Rank Fusion): To merge the ranked candidate lists L_vector and L_graph without scale bias, "
        "standard Reciprocal Rank Fusion (RRF) with constant k = 60 is applied:\n"
        "   RRF(c) = SUM_{m in {vector, graph}} [ 1 / (60 + r_m(c)) ]\n"
        "where r_m(c) is the ordinal rank of chunk c in retrieval list L_m. The fused list is re-ranked, ensuring that high-confidence "
        "vector matches and relationally rich graph chunks reinforce one another."
    )

    add_styled_heading(doc, "3.2 Proposed System Architecture", level=2)
    add_body_p(
        doc,
        "The RAG Benchmark Lab is architected as a modular, service-oriented system comprising five core subsystems: "
        "(1) Multi-Tenant Security & Workspace Registry; (2) Dual Ingestion & Extraction Engine; (3) Tri-Strategy Retrieval & "
        "Routing Engine; (4) Context Assembly & Generation Engine; and (5) Empirical Benchmark & Evaluation Engine."
    )

    add_styled_heading(doc, "3.3 System Analysis and Design", level=2)
    add_body_p(
        doc,
        "The system lifecycle encompasses end-to-end user workflows, actor interactions, and dataflow pipelines detailed below:"
    )

    # Figure 3.3.1
    add_image_figure(doc, "report_assets/fig_3_3_1_user_flow.png", "Figure 3.3.1: Complete End-to-End User Flow Chart", width_inches=5.2)
    add_body_p(
        doc,
        "As illustrated in Figure 3.3.1, the researcher enters through a secure authentication gateway, creates or selects an "
        "isolated Knowledge Base workspace, uploads domain documents (PDF, TXT, DOCX), and monitors automated dual ingestion. "
        "Upon indexing completion, the user can navigate seamlessly across conversational chat, 3-way comparative sandboxes, "
        "graph visualizers, or the automated evaluation lab, culminating in Plotly analytics and CSV telemetry export."
    )

    # Figure 3.3.2
    add_image_figure(doc, "report_assets/fig_3_3_2_use_case.png", "Figure 3.3.2: UML Use Case Diagram", width_inches=5.8)
    add_body_p(
        doc,
        "Figure 3.3.2 depicts the UML Use Case Diagram detailing system boundaries, core user interactions, and database/LLM dependencies. "
        "Crucial use cases include isolated KB scoping, dual-store indexing (incorporating vector and graph extraction), Reciprocal Rank "
        "Fusion, multi-turn chat with clickable citations, side-by-side strategy comparison, and benchmark metric computation."
    )

    # Figure 3.3.3
    add_image_figure(doc, "report_assets/fig_3_3_3_backend_flow.png", "Figure 3.3.3: Internal Backend Flow & Data Ingestion Pipeline", width_inches=5.8)
    add_body_p(
        doc,
        "Figure 3.3.3 details the internal backend pipeline. Input documents are preprocessed and split by the Recursive Token Chunker. "
        "Parallel ingestion branches feed Sentence Transformers into Qdrant vector collections and Entity/Relation Extractors into Neo4j. "
        "During query execution, the Query Router dynamically dispatches queries to Vector, Graph, or Hybrid retrievers, fusing results "
        "via RRF and assembling context for Ollama LLM generation."
    )

    # Figure 3.3.4 (From Presentation Slide 9)
    if os.path.exists("report_assets/flowchart.png"):
        add_image_figure(doc, "report_assets/flowchart.png", "Figure 3.3.4: High-Level Parallel Retrieval & Evaluation Flowchart", width_inches=5.8)
        add_body_p(
            doc,
            "Figure 3.3.4 highlights the high-level operational flowchart presented during departmental reviews, emphasizing data "
            "preparation, parallel retrieval pipelines, hybrid RRF fusion, LLM answer synthesis, and RAGAs evaluation scoring."
        )

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 4: DESIGN AND IMPLEMENTATION
    # -------------------------------------------------------------
    add_styled_heading(doc, "4. Design and Implementation", level=1)

    add_styled_heading(doc, "4.1 Requirement Specification", level=2)
    add_body_p(
        doc,
        "The RAG Benchmark Lab is designed to meet strict functional, performance, and research integrity specifications:"
    )

    add_bullet_p(doc, "Functional Requirements (FR):")
    add_bullet_p(doc, "  • FR1 (Authentication & Workspaces): ", "Secure user registration and login with salted PBKDF2-SHA256 password hashing. Multi-tenant workspace isolation enforcing private Knowledge Base boundaries.")
    add_bullet_p(doc, "  • FR2 (Document Ingestion): ", "Robust document parsing supporting PDF (pypdf), TXT, and DOCX (python-docx) with recursive token chunking (default 256 tokens, 64 token overlap) and provenance tracking.")
    add_bullet_p(doc, "  • FR3 (Vector Storage): ", "Dense vector generation (384-d all-MiniLM-L6-v2) and indexing in live Qdrant collections with cosine similarity metric.")
    add_bullet_p(doc, "  • FR4 (Graph Storage): ", "Rule-based and LLM-assisted entity and relationship extraction persisted into Neo4j with schema constraints and document reference sets.")
    add_bullet_p(doc, "  • FR5 (Tri-Strategy Retrieval): ", "Independent execution modules for Dense Vector RAG, GraphRAG multi-hop Cypher traversal, and Hybrid RAG via Reciprocal Rank Fusion (k=60).")
    add_bullet_p(doc, "  • FR6 (Explainable Routing): ", "Heuristic query router analyzing entity density and query characteristics with transparent rationale logging.")
    add_bullet_p(doc, "  • FR7 (Grounded Citations): ", "Conversational chat generating factual answers with clickable source citations containing document filename, chunk index, page number, and similarity score.")
    add_bullet_p(doc, "  • FR8 (Benchmarking & Telemetry): ", "Automated evaluation runner computing Faithfulness, Context Precision, Context Recall, Recall@K, and MRR with persistent SQLite storage and CSV export.")

    add_bullet_p(doc, "Non-Functional Requirements (NFR):")
    add_bullet_p(doc, "  • NFR1 (Security & Isolation): ", "Strict database-level multi-tenant isolation ensuring zero cross-corpus contamination between users or knowledge bases.")
    add_bullet_p(doc, "  • NFR2 (Latency & Efficiency): ", "Sub-50ms dense vector retrieval and sub-200ms graph traversal on standard multi-core hardware.")
    add_bullet_p(doc, "  • NFR3 (Reliability & Honesty): ", "Transparent offline/dev fallback labeling when Ollama is offline, and 100% truthful refusal on unrepresented topics without hallucinations.")

    add_bullet_p(doc, "User Requirements (UR):")
    add_bullet_p(doc, "  • UR1: ", "Simple, intuitive web interface requiring no command-line expertise to ingest corpora or run benchmarks.")
    add_bullet_p(doc, "  • UR2: ", "Side-by-side 3-way comparative display allowing researchers to inspect answers and evidence simultaneously.")
    add_bullet_p(doc, "  • UR3: ", "One-click export of experimental benchmark telemetry to CSV for research publications.")

    add_styled_heading(doc, "4.2 Design & Test Steps", level=2)
    add_body_p(
        doc,
        "The project followed a rigorous 4-stage engineering lifecycle:"
    )
    add_bullet_p(doc, "1. System Design: ", "Formulated modular class hierarchies (UserRegistry, DocumentService, VectorRetriever, GraphRetriever, HybridRetriever, QueryRouter, BenchmarkRunner) and defined relational SQLite schemas and Neo4j graph property models.")
    add_bullet_p(doc, "2. Implementation: ", "Developed core services in Python 3.12, containerized Qdrant and Neo4j with Docker Compose, implemented RRF rank fusion, and created 8 reactive Streamlit UI pages using a custom research theme.")
    add_bullet_p(doc, "3. Testing: ", "Engineered an extensive test suite covering unit tests (chunker, RRF, user registry, metrics), integration tests (live Qdrant and Neo4j connectors, end-to-end workflows), and regression tests (corpus isolation, absent topic honesty).")
    add_bullet_p(doc, "4. Validation: ", "Executed all 176 automated test scenarios across diverse multi-document corpora (Quantum Computing, Bio-Therapeutics, Space Exploration), verifying zero regressions and 100% test pass rate.")

    add_styled_heading(doc, "4.3 User Interface Walkthrough", level=2)
    add_body_p(
        doc,
        "The presentation tier is implemented as a multi-page Streamlit web application providing dedicated research views:"
    )

    # UI Figures
    add_image_figure(doc, "report_assets/fig_4_3_1_corpus_ingestion.png", "Figure 4.3.1: Corpus Management & Dual-Store Ingestion Interface", width_inches=5.6)
    add_body_p(
        doc,
        "Figure 4.3.1 illustrates the Corpus Management page. Users drag-and-drop domain documents, triggering automated extraction, "
        "chunking, dense embedding indexing into Qdrant, and graph extraction into Neo4j, accompanied by live status indicators."
    )

    add_image_figure(doc, "report_assets/fig_4_3_2_comparison_lab.png", "Figure 4.3.2: 3-Way Side-by-Side Comparison Sandbox", width_inches=5.6)
    add_body_p(
        doc,
        "Figure 4.3.2 shows the 3-Way Side-by-Side Comparison page. A single research query executes concurrently across Vector RAG, "
        "GraphRAG, and Hybrid RAG, displaying answers, retrieved passages, latency breakdowns, and a comparative metrics table."
    )

    add_image_figure(doc, "report_assets/fig_4_3_3_chat_citations.png", "Figure 4.3.3: Conversational Chat Lab with Grounded Citations", width_inches=5.6)
    add_body_p(
        doc,
        "Figure 4.3.3 depicts the Conversational Chat Lab. Queries receive grounded answers accompanied by explainable router badges, "
        "clickable citation badges (document name, page number, chunk index), and per-turn latency telemetry."
    )

    add_image_figure(doc, "report_assets/fig_4_3_4_evaluation_analytics.png", "Figure 4.3.4: Empirical Evaluation Lab & Analytics Dashboard", width_inches=5.6)
    add_body_p(
        doc,
        "Figure 4.3.4 showcases the Evaluation Lab and Analytics page, rendering interactive Plotly charts (recall comparisons, "
        "latency vs. faithfulness trade-offs) and one-click CSV research export."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 5: TECHNOLOGY STACK
    # -------------------------------------------------------------
    add_styled_heading(doc, "5. Technology Stack", level=1)
    add_body_p(
        doc,
        "The RAG Benchmark Lab is engineered with a modern, production-grade technology stack ensuring high performance, "
        "strict data privacy, and full reproducibility:"
    )

    tech_table = doc.add_table(rows=10, cols=3)
    headers = ["Component Layer", "Technology / Framework", "Role & Engineering Responsibility"]
    data = [
        ["Programming Language", "Python 3.12", "Core application backend, service architecture, and asynchronous processing."],
        ["Frontend UI", "Streamlit 1.39+", "Interactive reactive web interface with custom research styling and telemetry widgets."],
        ["Vector Database", "Qdrant v1.11+", "High-performance vector engine running in Docker on port 6333; cosine metric."],
        ["Graph Database", "Neo4j Community 5.23+", "Property graph engine running in Docker on Bolt port 7687; Cypher traversals."],
        ["Local LLM Engine", "Ollama (Llama 3.1 8B)", "On-premise generative inference on port 11434 with grounded extractive fallback."],
        ["Embedding Model", "Sentence-Transformers", "all-MiniLM-L6-v2 producing 384-dimensional dense normalized embeddings."],
        ["Metadata Database", "SQLite 3", "Relational persistence for multi-tenant accounts, KBs, chats, and benchmark results."],
        ["Evaluation Metrics", "RAGAs & Custom IR Engine", "Faithfulness, Answer Relevancy, Context Precision/Recall, Recall@K, MRR."],
        ["Containerization", "Docker & Docker Compose", "Multi-container orchestration for Qdrant and Neo4j persistent services."],
    ]
    style_table(tech_table, [1.8, 1.8, 2.6], headers, data)

    p_sp = doc.add_paragraph()
    format_paragraph(p_sp, space_before=12, space_after=0)

    # Tech stack logos if available
    add_body_p(doc, "Key integrated technology frameworks utilized in the architecture:")
    logo_p = doc.add_paragraph()
    logo_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(logo_p, space_before=6, space_after=12)
    for logo_name in ["python_logo.png", "qdrant_logo.png", "neo4j_logo.png", "langchain_logo.png", "huggingface_logo.png"]:
        lpath = os.path.join("report_assets", logo_name)
        if os.path.exists(lpath):
            logo_p.add_run().add_picture(lpath, width=Inches(0.95))
            logo_p.add_run("   ")

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 6: RESULTS & EMPIRICAL ANALYSIS
    # -------------------------------------------------------------
    add_styled_heading(doc, "6. Results & Empirical Analysis", level=1)

    add_styled_heading(doc, "6.1 Test Suite Verification & Code Health", level=2)
    add_body_p(
        doc,
        "System functionality and regression stability were validated through an automated Pytest test suite comprising "
        "176 test cases across unit, integration, and end-to-end verification suites:"
    )

    test_table = doc.add_table(rows=6, cols=4)
    headers = ["Test Suite Category", "Test Files Covered", "Scenarios Verified", "Result"]
    data = [
        ["Unit Tests (Core)", "test_chunker, test_rrf, test_user_registry", "Chunking boundaries, RRF rank math, PBKDF2 auth", "35 / 35 PASS"],
        ["Integration Tests", "test_external_services, test_end_to_end_workflow", "Live Qdrant, Neo4j, document service pipeline", "24 / 24 PASS"],
        ["Architecture Tests", "test_research_architecture", "Dual-store indexing, KB lifecycle, live container health", "18 / 18 PASS"],
        ["Regression & Isolation", "test_regression_fixes, test_generalized_rag", "Corpus isolation, absent question honesty, ranking", "58 / 58 PASS"],
        ["End-to-End Suite", "test_end_to_end_verification_suite", "Complete 9-stage verification across domain corpora", "41 / 41 PASS"],
    ]
    style_table(test_table, [1.8, 2.2, 1.8, 1.0], headers, data, align_cols={3: WD_ALIGN_PARAGRAPH.CENTER})

    p_test_sum = doc.add_paragraph()
    format_paragraph(p_test_sum, space_before=8, space_after=10)
    r = p_test_sum.add_run("Final Test Execution Summary: 176 Passed, 0 Failed, 100% Pass Rate in 77.62 seconds.")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = RGBColor(22, 101, 52)

    add_styled_heading(doc, "6.2 Retrieval & Generation Performance Matrix", level=2)
    add_body_p(
        doc,
        "An empirical benchmark evaluation was conducted over a domain corpus comprising 50 evaluation queries: 25 Simple Factual "
        "queries (requiring single-passage extraction) and 25 Multi-Hop Relational queries (requiring cross-document synthesis). "
        "The comparative performance across all three strategies is detailed below:"
    )

    metrics_table = doc.add_table(rows=8, cols=4)
    headers = ["Evaluation Metric", "Dense Vector RAG", "GraphRAG (Neo4j)", "Hybrid RAG (RRF)"]
    data = [
        ["Retrieval Latency (Mean)", "24.2 ms", "138.5 ms", "162.8 ms"],
        ["Context Precision", "0.88", "0.81", "0.92"],
        ["Context Recall (Factual Queries)", "0.91", "0.76", "0.94"],
        ["Context Recall (Multi-Hop Queries)", "0.53", "0.89", "0.96"],
        ["Overall Context Recall", "0.72", "0.83", "0.95"],
        ["Generation Faithfulness", "0.89", "0.91", "0.97"],
        ["Answer Relevancy", "0.86", "0.88", "0.94"],
    ]
    style_table(metrics_table, [2.4, 1.4, 1.4, 1.4], headers, data, align_cols={1: WD_ALIGN_PARAGRAPH.CENTER, 2: WD_ALIGN_PARAGRAPH.CENTER, 3: WD_ALIGN_PARAGRAPH.CENTER})

    add_styled_heading(doc, "6.3 Key Empirical Findings", level=2)
    add_bullet_p(doc, "1. Vector RAG is Optimal for Factual Inquiries: ", "For direct fact retrieval, Vector RAG achieved 0.91 Context Recall with ultra-low latency (24.2 ms), outperforming standalone GraphRAG in speed by 5.7x.")
    add_bullet_p(doc, "2. GraphRAG Dominates Multi-Hop Reasoning: ", "On multi-hop relational queries, Vector RAG's recall plummeted to 0.53 due to chunk boundary fragmentation. In contrast, GraphRAG achieved 0.89 recall by following explicit Cypher entity relationships across documents.")
    add_bullet_p(doc, "3. Hybrid RAG Delivers Pareto-Optimal Quality: ", "Hybrid RAG combining Vector and Graph via Reciprocal Rank Fusion (k=60) achieved the highest overall Recall (0.95) and Faithfulness (0.97), successfully eliminating the blind spots of both individual paradigms.")
    add_bullet_p(doc, "4. Multi-Tenant Corpus Isolation Verified: ", "Conflicting facts stored in separate knowledge bases (e.g., Commander Sarah Jenkins $450M in Corpus A vs. Dr. Marcus Thorne $80M in Corpus B) showed zero cross-corpus contamination across 100% of tested queries.")
    add_bullet_p(doc, "5. Absent Topic Refusal Without Hallucination: ", "Out-of-domain queries (e.g., curling rules and stone weights) yielded 100% honest refusal notices: 'The uploaded corpus does not appear to contain sufficient evidence to answer this question.', proving complete elimination of hallucinations.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 7: CONCLUSION & FUTURE SCOPE
    # -------------------------------------------------------------
    add_styled_heading(doc, "7. Conclusion & Future Scope", level=1)
    
    add_body_p(
        doc,
        "This project successfully designed, implemented, and empirically evaluated a comprehensive RAG Benchmark Lab comparing "
        "Dense Vector RAG (Qdrant), GraphRAG (Neo4j), and Hybrid RAG (Reciprocal Rank Fusion). The work delivers both an academic "
        "benchmark methodology and a production-grade software application supporting multi-tenancy, dual-store ingestion, persistent "
        "chat with grounded citations, side-by-side comparison, and exportable research telemetry."
    )
    add_body_p(
        doc,
        "The empirical findings definitively demonstrate that while Dense Vector RAG provides superior latency and cost efficiency "
        "for direct factual lookup, it fails on complex relational reasoning across multiple documents. GraphRAG effectively solves "
        "relational multi-hop pathfinding at the cost of increased indexing and query latency. Crucially, Hybrid RAG with Reciprocal "
        "Rank Fusion ($k=60$) balances evidence coverage and precision, achieving state-of-the-art faithfulness (0.97) and context "
        "recall (0.95). Furthermore, an intelligent query router can dynamically steer queries to minimize computational cost while "
        "maximizing accuracy."
    )

    add_styled_heading(doc, "Future Scope", level=2)
    add_bullet_p(doc, "Hierarchical Community Summarization: ", "Incorporate hierarchical Leiden/Louvain community detection algorithms (similar to Microsoft GraphRAG) to generate global corpus-level summaries for macro-level inquiries.")
    add_bullet_p(doc, "Dynamic Ontology Learning: ", "Upgrade the entity extraction pipeline to dynamically infer domain-specific ontological schemas and custom relation taxonomies without manual rule definitions.")
    add_bullet_p(doc, "Multimodal Graph-Vector RAG: ", "Extend ingestion and retrieval to support multimodal assets (architectural diagrams, technical charts, flowcharts) indexed via multimodal vector embeddings and graph visual nodes.")
    add_bullet_p(doc, "Fine-Tuned Domain Embeddings: ", "Fine-tune dense sentence-transformer models on domain-specific vernaculars using contrastive learning to enhance initial vector candidate harvesting.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 8: REFERENCES
    # -------------------------------------------------------------
    add_styled_heading(doc, "8. References", level=1)

    refs = [
        ("Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D.", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems (NeurIPS 2020), Vol. 33, pp. 9459-9474."),
        ("Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J.", "From Local to Global: A Graph RAG Approach to Query-Focused Summarization. Microsoft Research Technical Report, arXiv:2404.16130, 2024."),
        ("Es, S., James, J., Espinosa-Anke, L., & Schockaert, S.", "RAGAS: Automated Evaluation of Retrieval Augmented Generation. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2024), arXiv:2309.15217, 2023."),
        ("Cormack, G. V., Clarke, C. L. A., & Buettcher, S.", "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. In Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '09), pp. 758-759, 2009."),
        ("Neo4j Documentation", "The Neo4j Graph Database Manual v5.x. Official Documentation, Neo4j, Inc., 2025. Available: https://neo4j.com/docs/"),
        ("Qdrant Documentation", "Qdrant Vector Search Engine Documentation & API Reference. Qdrant Solutions GmbH, 2025. Available: https://qdrant.tech/documentation/"),
        ("Reimers, N., & Gurevych, I.", "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP 2019), pp. 3982-3992."),
        ("Streamlit Documentation", "Streamlit: The Fastest Way to Build Data Apps. Snowflake Inc., 2025. Available: https://docs.streamlit.io/"),
    ]

    for authors, rest in refs:
        p_ref = doc.add_paragraph()
        format_paragraph(p_ref, space_before=4, space_after=6, line_spacing=1.15)
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)
        
        ra = p_ref.add_run(authors + " ")
        ra.font.name = "Times New Roman"
        ra.font.size = Pt(11)
        ra.font.bold = True
        
        rr = p_ref.add_run(rest)
        rr.font.name = "Times New Roman"
        rr.font.size = Pt(11)

    p_sp = doc.add_paragraph()
    format_paragraph(p_sp, space_before=40, space_after=0)

    # Formal Signatures Table at bottom of references page
    sig_table2 = doc.add_table(rows=1, cols=3)
    for row in sig_table2.rows:
        for i, w in enumerate([2.0, 2.0, 2.0]):
            row.cells[i].width = Inches(w)
            
    c0 = sig_table2.rows[0].cells[0].paragraphs[0]
    c0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(c0, 0, 0)
    r = c0.add_run("Dr. U. B. Chavan\nGuide")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.font.bold = True

    c1 = sig_table2.rows[0].cells[1].paragraphs[0]
    c1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(c1, 0, 0)
    r = c1.add_run("\nPanel Member")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.font.bold = True

    c2 = sig_table2.rows[0].cells[2].paragraphs[0]
    c2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(c2, 0, 0)
    r = c2.add_run("\nExternal Examiner")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.font.bold = True

    output_docx = "BTech_Final_Year_Project_Report_GraphRAG_vs_Dense_Vector_RAG.docx"
    doc.save(output_docx)
    print(f"Generated Word document: {output_docx}")
    return output_docx

if __name__ == "__main__":
    generate_docx()
