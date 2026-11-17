"""Generate a professional Word report from Markdown and actual run artifacts."""
import argparse
import json
from pathlib import Path
import re
from config import ROOT,RESULTS_DIR,MODEL_PATH


def replace_block(text,name,content):
    return re.sub(rf"<!-- {name}_START -->.*?<!-- {name}_END -->",
                  lambda _: f"<!-- {name}_START -->\n{content}\n<!-- {name}_END -->",text,flags=re.S)


def update_evidence(text):
    if (RESULTS_DIR/"dataset.json").is_file():
        data=json.loads((RESULTS_DIR/"dataset.json").read_text())
        rows="\n".join(f"| {name} | {count} |" for name,count in data["class_distribution"].items())
        content=(f"The downloaded resized ZIP contains {data['archive_bytes']:,} bytes "
                 f"({data['archive_bytes']/1024/1024:.2f} MiB). The validated inventory has {data['unique_images']:,} unique decoded images. "
                 f"Archive SHA-256: `{data['archive_sha256']}`.\n\n| Category | Validated images |\n|---|---:|\n{rows}")
        if "raw_class_distribution" in data:
            content+="\n\nRaw archive folder counts: "+", ".join(f"{k} {v}" for k,v in data["raw_class_distribution"].items())+"."
            content+=f" A total of {len(data.get('excluded_images', []))} files were excluded as exact duplicates or members of conflicting-label duplicate groups. Full paths, hashes and reasons are recorded in the dataset metadata; no conflicting label is chosen arbitrarily."
        text=replace_block(text,"DATASET",content)
    if (RESULTS_DIR/"metrics.json").is_file() and MODEL_PATH.is_file():
        from smartwaste.evaluation import file_sha256
        data=json.loads((RESULTS_DIR/"metrics.json").read_text())
        if data["model_sha256"]!=file_sha256(MODEL_PATH):
            raise ValueError("Refusing to report metrics for a different model checkpoint.")
        content=(f"The actual held-out test set contains {data['test_samples']} images. The checkpoint is selected using validation loss only.\n\n"
                 "| Metric | Measured value |\n|---|---:|\n"+
                 "\n".join(f"| {label} | {data[key]:.4f} |" for label,key in [("Accuracy","test_accuracy"),("Macro precision","macro_precision"),("Macro recall","macro_recall"),("Macro F1","macro_f1")]))
        report=data["classification_report"]
        content+="\n\n| Class | Precision | Recall | F1 | Test support |\n|---|---:|---:|---:|---:|\n"
        content+="\n".join(f"| {name} | {report[name]['precision']:.4f} | {report[name]['recall']:.4f} | {report[name]['f1-score']:.4f} | {int(report[name]['support'])} |" for name in data["class_names"])
        content+="\n\n![Held out confusion matrix](../results/confusion_matrix.png)\n\n![Training and validation accuracy](../results/accuracy.png)\n\n![Training and validation loss](../results/loss.png)"
        text=replace_block(text,"RESULTS",content)
    return text


def inline(paragraph,text):
    """Preserve simple code/bold emphasis without exposing Markdown syntax."""
    for i,part in enumerate(re.split(r"(`[^`]+`|\*\*[^*]+\*\*)",text)):
        if part.startswith("`"):
            run=paragraph.add_run(part[1:-1]); run.font.name="Consolas"
        elif part.startswith("**"):
            paragraph.add_run(part[2:-2]).bold=True
        else:
            paragraph.add_run(part)


def build_docx(text,destination):
    from docx import Document
    from docx.shared import Inches,Pt,RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    document=Document()
    document.core_properties.title="SmartWaste AI Project Report"
    document.core_properties.author=""
    document.core_properties.subject="CNN waste classification and Mamdani fuzzy decision support"
    section=document.sections[0]
    section.page_width=Inches(8.27); section.page_height=Inches(11.69)
    section.top_margin=section.bottom_margin=Inches(.8)
    section.left_margin=section.right_margin=Inches(.85)
    normal=document.styles["Normal"]
    normal.font.name="Calibri"; normal.font.size=Pt(11)
    normal.paragraph_format.line_spacing=1.12
    normal.paragraph_format.space_after=Pt(7)
    for name in ("Title","Heading 1","Heading 2"):
        style=document.styles[name]
        style.font.name="Calibri"; style.font.color.rgb=RGBColor(0,0,0)
    document.styles["Title"].font.size=Pt(28)
    document.styles["Heading 1"].font.size=Pt(16)
    document.styles["Heading 2"].font.size=Pt(12)
    lines=text.splitlines(); i=0; title_done=False
    while i<len(lines):
        line=lines[i].strip()
        if not line or line.startswith("<!--"):
            i+=1; continue
        if line.startswith("# "):
            document.add_heading(line[2:],0); title_done=True
            i+=1; continue
        if line.startswith(("Student:", "Roll number:", "Programme and department:",
                            "Institution:", "Supervisor:", "Academic year:")):
            document.add_paragraph(line)
            i+=1; continue
        if line.startswith("## "):
            if line.startswith("## 1 "):
                document.add_page_break()
                document.add_heading("Contents",1)
                for heading in [x[3:] for x in lines if x.startswith("## ")]:
                    document.add_paragraph(heading)
                document.add_page_break()
            document.add_heading(line[3:],1); i+=1; continue
        image=re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)",line)
        if image:
            path=(ROOT/"docs"/image[2]).resolve()
            if path.is_file():
                paragraph=document.add_paragraph(); paragraph.alignment=1
                paragraph.paragraph_format.keep_with_next=True
                picture=paragraph.add_run().add_picture(str(path),width=Inches(6.3))
                picture._inline.docPr.set("descr",image[1])
                caption=document.add_paragraph(image[1],style="Caption"); caption.alignment=1
            i+=1; continue
        if line.startswith("|"):
            table_lines=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                row=[x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?",x) for x in row): table_lines.append(row)
                i+=1
            if table_lines:
                table=document.add_table(rows=1,cols=len(table_lines[0])); table.style="Light Shading Accent 1"
                for cell,label in zip(table.rows[0].cells,table_lines[0]): cell.text=label
                header=OxmlElement("w:tblHeader"); table.rows[0]._tr.get_or_add_trPr().append(header)
                for row in table_lines[1:]:
                    for cell,value in zip(table.add_row().cells,row): cell.text=value.replace("`","")
                for row in table.rows:
                    for cell in row.cells:
                        for paragraph in cell.paragraphs:
                            paragraph.paragraph_format.space_after=Pt(4)
                            for run in paragraph.runs: run.font.size=Pt(9)
                document.add_paragraph()
            continue
        parts=[line]
        while i+1<len(lines) and lines[i+1].strip() and not lines[i+1].lstrip().startswith(("#","|","!", "<!--")):
            i+=1; parts.append(lines[i].strip())
        paragraph=document.add_paragraph()
        inline(paragraph," ".join(parts))
        i+=1
    # Actual screenshots are appended only if captured from a running app.
    screenshots=sorted((ROOT/"assets/screenshots").glob("*.png"))
    if screenshots:
        document.add_page_break(); document.add_heading("Application screenshots",1)
        for path in screenshots:
            from PIL import Image
            with Image.open(path) as image:
                width,height=image.size
            scale=min(6.3/width,7.5/height)
            paragraph=document.add_paragraph()
            paragraph.paragraph_format.keep_with_next=True
            paragraph.add_run().add_picture(str(path),width=Inches(width*scale),height=Inches(height*scale))
            document.add_paragraph(path.stem.replace("_"," ").title(),style="Caption")
    document.save(destination)
    print(f"Generated {destination}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-figures",action="store_true",help="Reuse already generated figures.")
    args=parser.parse_args()
    if not args.skip_figures:
        from scripts.generate_figures import main as figures
        figures()
    source=ROOT/"docs/SmartWaste_AI_Project_Report.md"
    text=update_evidence(source.read_text(encoding="utf-8"))
    source.write_text(text,encoding="utf-8")
    build_docx(text,ROOT/"docs/SmartWaste_AI_Project_Report.docx")


if __name__=="__main__": main()
