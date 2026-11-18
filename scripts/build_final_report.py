"""Build the CSN441 report from existing evidence; never train or modify the app."""
import argparse, ast, base64, hashlib, io, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from config import ROOT, CLASS_NAMES, MATERIAL_POTENTIAL
from smartwaste.fuzzy_engine import infer

NAME="SmartWaste_AI_Final_Project_Report"
OUT=ROOT/"docs"
BT=chr(96)
GREEN="173F30"
def read(p): return (ROOT/p).read_text(encoding="utf-8")
def sha(p): return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def table(headers,rows):
    return "\n".join(["| "+" | ".join(map(str,headers))+" |","|"+"---|"*len(headers)]+["| "+" | ".join(map(str,r))+" |" for r in rows])
def extract_function(path,name):
    text=read(path)
    node=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(text,node)
def prepare_screens():
    dest=OUT/"screenshots";dest.mkdir(exist_ok=True)
    for p in (ROOT/"assets/screenshots").glob("*.png"): shutil.copy2(p,dest/p.name)
    source=dest/"classification.png"
    # These are crops of real existing screenshots, never synthetic UI.
    with Image.open(source) as im:
        im.crop((835,525,min(im.width,1395),min(im.height,980))).save(dest/"cnn_scores.png")
    from playwright.sync_api import sync_playwright, expect
    log=open("/tmp/smartwaste-report-app.log","w")
    proc=subprocess.Popen([sys.executable,"-m","streamlit","run","app.py","--server.headless","true","--server.port","8520"],cwd=ROOT,stdout=log,stderr=log)
    try:
        from urllib.request import urlopen
        for _ in range(60):
            try:
                if urlopen("http://localhost:8520/_stcore/health",timeout=1).status==200: break
            except OSError: time.sleep(1)
        else: raise RuntimeError("Report screenshot app did not start")
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=shutil.which("google-chrome") or None)
            page=browser.new_page(viewport={"width":1440,"height":1100},device_scale_factor=1)
            page.goto("http://localhost:8520",wait_until="networkidle")
            page.get_by_text("Fuzzy Logic Laboratory",exact=True).first.click()
            page.locator("[data-testid='stApp'][data-test-script-state='notRunning']").wait_for()
            expect(page.get_by_test_id("stSkeleton")).to_have_count(0)
            time.sleep(1)
            chart=page.get_by_test_id("stPlotlyChart").nth(2)
            chart.screenshot(path=str(dest/"fuzzy_aggregate.png"))
            page.get_by_role("heading",name="Rule firing strengths",exact=True).scroll_into_view_if_needed()
            time.sleep(.5)
            page.get_by_test_id("stDataFrame").nth(1).screenshot(path=str(dest/"fuzzy_rules.png"))
            browser.close()
    finally:
        proc.terminate();proc.wait(timeout=15);log.close()

def content():
    dataset=json.loads(read("results/dataset.json"))
    metrics=json.loads(read("results/metrics.json"))
    cfg=json.loads(read("models/training_config.json"))
    manifest=json.loads(read("models/model_manifest.json"))
    assert sha("models/smartwaste.keras")==metrics["model_sha256"]==manifest["model_sha256"]
    pred=np.load(ROOT/"results/test_predictions.npz")
    labels=pred["labels"]; scores=pred["scores"]; guess=scores.argmax(1)
    matrix=np.zeros((6,6),int)
    np.add.at(matrix,(labels,guess),1)
    assert matrix.tolist()==metrics["confusion_matrix"] and len(labels)==379
    assert abs((labels==guess).mean()-metrics["test_accuracy"])<1e-12
    import csv
    splits=list(csv.DictReader(io.StringIO(read("results/splits.csv"))))
    tests=[r for r in splits if r["split"]=="test"]
    assert len(tests)==379
    base=read("docs/SmartWaste_AI_Project_Report.md")
    chunks=re.split(r"^## (\d+) ([^\n]+)\n",base,flags=re.M)
    sections={int(chunks[i]):chunks[i+2].strip() for i in range(1,len(chunks),3)}
    blocks=[]
    def add(k,t,key=None): blocks.append((k,t,key))
    def markdown(t):
        lines=t.splitlines();i=0
        while i<len(lines):
            line=lines[i].strip()
            if not line or line.startswith("<!--"):i+=1;continue
            image=re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)",line)
            if image:add("image",(image[1],image[2]));i+=1;continue
            if line.startswith("|"):
                rows=[]
                while i<len(lines) and lines[i].strip().startswith("|"):
                    cells=[c.strip() for c in lines[i].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r":?-+:?",c) for c in cells):rows.append(cells)
                    i+=1
                add("table",rows);continue
            lines2=[line]
            while i+1<len(lines) and lines[i+1].strip() and not lines[i+1].strip().startswith(("|","!","<!--")):
                i+=1;lines2.append(lines[i].strip())
            add("p"," ".join(lines2));i+=1
    def chapter(n,title):
        add("h1",f"CHAPTER {n} — {title.upper()}",f"chapter{n}")
    def section(n,title,text):
        add("h2",n+" "+title);markdown(text)
    def fig(title,p):add("image",(title,p))
    chapter(1,"Introduction")
    for no,sec,title in [(1,2,"Background and motivation"),(2,3,"Problem definition"),(3,4,"Project objectives"),(4,5,"Scope and intended application")]:
        section(f"1.{no}",title,sections[sec])
    chapter(2,"Literature Review and Existing Approaches")
    section("2.1","Image recognition and waste datasets",sections[6])
    section("2.2","Choice of an integrated approach","Image recognition assigns a material label, whereas recycling advice depends on additional assumptions. SmartWaste AI therefore keeps the learned classifier and the explicit fuzzy decision layer separate. This design makes a category score auditable without treating it as a measured recycling rate. The implementation follows a closed-set image-classification task rather than object detection or industrial sorting.")
    markdown(table(["Approach","Useful capability","Limitation for this project"],[
        ["Manual visual segregation","Human context and judgement","Consistency and material knowledge vary"],
        ["CNN-only classification","Learns material appearance","A class score does not encode contamination"],
        ["Rule-only suitability","Inspectable assumptions","Requires a material input"],
        ["Implemented hybrid","CNN label plus manual contamination and Mamdani trace","Assumptions and domain limits remain explicit"]]))
    chapter(3,"Dataset and Preprocessing")
    section("3.1","Dataset collection and validation",sections[10])
    markdown(table(["Class","Raw","Retained","Excluded"],[[c,dataset["raw_class_distribution"][c],dataset["class_distribution"][c],dataset["raw_class_distribution"][c]-dataset["class_distribution"][c]] for c in CLASS_NAMES]))
    section("3.2","Image processing and split isolation",sections[11])
    rows=[]
    for c in CLASS_NAMES:
        rows.append([c]+[sum(r["category"]==c and r["split"]==s for r in splits) for s in ("train","validation","test")])
    markdown(table(["Class","Training","Validation","Test"],rows+[["Total",1764,378,379]]))
    section("3.3","Augmentation and imbalance handling","Only training batches receive random horizontal flips, random rotation of 0.08 turns and random zoom of 0.10, with reflection filling. Validation and test images are not augmented. Training uses inverse-frequency class weights N/(K × n_c), calculated only from training labels. The largest weight belongs to trash, which has the smallest modelling inventory. These choices reduce training imbalance without duplicating evaluation images.")
    markdown(table(["Class","Training class weight"],[[c,f"{cfg['class_weights'][str(i)]:.4f}"] for i,c in enumerate(CLASS_NAMES)]))
    chapter(4,"System Architecture and Methodology")
    section("4.1","Proposed system",sections[7])
    section("4.2","Software and hardware",sections[8])
    section("4.3","Architecture and data flow",sections[9])
    section("4.4","CNN model development",sections[12])
    markdown(table(["Layer / block","Output for one image","Role"],[
        ["RGB input","160 × 160 × 3","Official MobileNetV2 preprocessing"],
        ["Frozen MobileNetV2, alpha 1.0","5 × 5 × 1280","ImageNet feature extraction"],
        ["Global average pooling","1280","Spatial feature summary"],
        ["Dropout, rate 0.3","1280","Training-time regularisation"],
        ["Dense + softmax","6","Original TrashNet labels"]]))
    section("4.5","Transfer learning",sections[13])
    section("4.6","Recorded training configuration","The completed seeded run used a CPU configuration with two threads. It trained the added classification head for 12 epochs with the ImageNet base frozen; it did not train the whole network from scratch or fine-tune the base. The minimum validation loss selected epoch 10. Early stopping and learning-rate reduction were configured, but the 12-epoch history records a learning rate of 0.001 throughout.")
    markdown(table(["Setting","Recorded value"],[["Input / batch / seed","160 × 160 RGB / 32 / 42"],["Optimizer / loss","Adam 0.001 / sparse categorical cross-entropy"],["Early stopping","Validation loss, patience 4"],["Checkpoint","Best validation loss, epoch 10"],["Learning-rate reduction","Factor 0.5, patience 2, minimum 0.00001"],["Run framework","Python 3.11.18; TensorFlow 2.20.0"],["Saved checkpoint","Keras 3.15.1; 9,718,514 bytes"]]))
    chapter(5,"Fuzzy Logic Implementation")
    for no,sec,title in [(1,14,"Variables and design assumptions"),(2,15,"Membership functions"),(3,16,"Complete nine-rule base"),(4,17,"Mamdani rule evaluation"),(5,18,"Centroid defuzzification")]:
        section(f"5.{no}",title,sections[sec])
    section("5.6","Worked numerical inference","For potential 80 and contamination 20, High-potential membership and Low-contamination membership are both 1. Rule R7 activates High suitability with strength 1; the other rules do not fire. The centroid of the full High shoulder is 725/9 = 80.5556. For potential 90 and contamination 100, High contamination activates Low suitability and the output centroid is 175/9 = 19.4444. These are decision scores under the supplied assumptions.")
    rows=[]
    for p,c in [(80,20),(60,50),(37.5,62.5),(90,100)]:
        result=infer(p,c); rows.append([p,c,f"{result.score:.4f}",result.status,", ".join(r["Rule"] for r in result.rules if r["Firing strength"]>0)])
    markdown(table(["Potential","Contamination","Centroid","Status","Active rules"],rows))
    section("5.7","Confidence and suitability are different","The six CNN softmax scores rank material categories. They are uncalibrated and do not guarantee correctness. The fuzzy score is calculated from an assumed material potential and a user's contamination input; it is neither a probability nor an observed recycling percentage. Changing contamination does not retrain or replace the CNN.")
    fig("Actual Mamdani aggregate and centroid in the laboratory","screenshots/fuzzy_aggregate.png")
    fig("Actual nine-rule firing strengths in the laboratory","screenshots/fuzzy_rules.png")
    chapter(6,"Application Development")
    section("6.1","Modules, contracts and session state",sections[19])
    section("6.2","Five application sections",sections[20])
    section("6.3","Interface refinement and local operation","The completed interface uses forest green, off-white and sage. Native Streamlit toolbar clearance prevents the page eyebrow from being obscured. The classifier presents a three-step empty state, a framed image preview, structured top-three scores and a separate fuzzy assessment. The sidebar preserves the five sections, readable branding and a distinct active item. Windows setup.bat and run.bat remain the launch workflow; the application opens in a browser at localhost:8501. No desktop executable, installer, authentication or database is introduced.")
    for title,p in [("Dashboard with a genuine one-classification session","dashboard.png"),("Classifier before upload with workflow guidance","classification_empty.png"),("Fuzzy Logic Laboratory with live membership charts","fuzzy_laboratory.png"),("About and Methodology with the implemented architecture","methodology.png")]: fig(title,"screenshots/"+p)
    chapter(7,"Testing and Model Evaluation")
    section("7.1","Experimental protocol","The held-out test partition contains 379 images and is separate from the 1,764 training and 378 validation images. Validation loss alone selects the checkpoint; the test partition does not choose epochs, weights or hyperparameters. Model and split SHA-256 digests bind evaluation to the saved artifacts.")
    section("7.2","Evaluation measures","Accuracy is the number of correct predictions divided by test support. Per-class precision = TP/(TP+FP), recall = TP/(TP+FN), and F1 = 2 × precision × recall/(precision+recall). Macro averaging gives all six classes equal weight. This is useful for the imbalanced dataset, although per-class support must also be considered.")
    markdown(table(["Held-out metric","Measured result"],[[title,f"{100*metrics[key]:.2f}%"] for title,key in [("Accuracy","test_accuracy"),("Macro precision","macro_precision"),("Macro recall","macro_recall"),("Macro F1-score","macro_f1")]]))
    markdown(table(["Class","Precision","Recall","F1","Support"],[[c]+[f"{metrics['classification_report'][c][k]:.4f}" for k in ("precision","recall","f1-score")]+[int(metrics["classification_report"][c]["support"])] for c in CLASS_NAMES]))
    fig("Recorded held-out confusion matrix, actual rows and predicted columns","../results/confusion_matrix.png")
    fig("Actual model-evaluation application overview","screenshots/model_evaluation.png")
    section("7.3","Training behaviour","Training accuracy rises from 46.43% at epoch 1 to 85.71% at epoch 12. Validation loss is lowest at epoch 10, rather than at the last epoch; that saved checkpoint is used for test evaluation. The curves describe the training and validation partitions, not the held-out test accuracy. No additional fine-tuning experiment is claimed.")
    fig("Recorded training and validation accuracy","../results/accuracy.png")
    fig("Recorded training and validation loss","../results/loss.png")
    section("7.4","Software and integration testing",sections[24])
    markdown(table(["Executed check","Observed outcome"],[
        ["Local, Linux and Windows Pytest suites","34 passed; 2 expected fuzzy failures; 6 subtests passed"],
        ["Real Windows launch and browser classification","Passed with the shipped checkpoint"],
        ["Contamination and CSV export","Confirmed input 100 and matching exported record"],
        ["Five pages at three requested sizes","18 responsive checks and 7 overview checks passed"],
        ["Header, sidebar and horizontal-overflow checks","Passed at 1920×1080, 1366×768 and 1280×720"],
        ["Model / fuzzy / metrics / launch-file integrity","Protected checksums unchanged during UI refinement"]]))
    section("7.5","Expected failures are not unexpected defects","The two xfailed cases deliberately retain known global monotonicity counterexamples for the exact required membership functions and inference operators. They are visible unmet mathematical properties, not passing guarantees. They do not indicate a crashed application, omitted rule or fabricated computation. All other executed tests pass; unavailable-dependency skips are not being presented as completed checks.")
    chapter(8,"Results and Discussion")
    section("8.1","Interpretation of held-out performance","The checkpoint correctly classifies 309 of 379 test images, leaving 70 errors. Cardboard and paper have the strongest F1 values in this run. Trash has an F1 of 0.5000 on only 20 test examples. The macro F1 of 77.70% is lower than the overall accuracy of 81.53%, illustrating the need to examine minority-class performance. Twelve glass examples are predicted as plastic in the recorded confusion matrix. These observations describe this split, not deployment guarantees.")
    rows=[]
    chosen=[next(i for i in range(len(labels)) if labels[i]==guess[i])]
    wrong=[i for i in range(len(labels)) if labels[i]!=guess[i]]
    chosen+=sorted(wrong,key=lambda i:float(scores[i,guess[i]]),reverse=True)[:2]
    for i in chosen:rows.append([tests[i]["path"],CLASS_NAMES[int(labels[i])],CLASS_NAMES[int(guess[i])],f"{scores[i,guess[i]]:.4f}","Correct" if labels[i]==guess[i] else "Incorrect"])
    markdown(table(["Saved test image","Actual label","Predicted label","Top score","Outcome"],rows))
    markdown("These examples are derived from results/test_predictions.npz in the recorded test-split order. The incorrect examples show that a high softmax score can accompany a wrong material label. Their photographs are not invented or substituted; the table reports saved evidence.")
    section("8.2","Fuzzy results and mathematical limitation",sections[23])
    section("8.3","Scope of the evidence",sections[25])
    chapter(9,"Conclusion and Future Scope")
    section("9.1","Conclusion",sections[26].replace("when training completes","for the completed training run"))
    section("9.2","Objectives achieved","The completed prototype provides six-class recognition with a saved, auditable checkpoint; an independent nine-rule Mamdani system; interactive visual explanations; held-out metrics; session history and export; and reproducible Windows execution. The assignment stages from problem definition to demonstration are all addressed. Its contribution is the transparent connection between a neural classifier and a declared decision model, rather than a claim of universal recycling expertise.")
    section("9.3","Future improvements",sections[27])
    chapter(10,"Demonstration")
    section("10.1","Launch and upload","From a normal Windows terminal in the project root, run setup.bat once to create the environment and install pinned packages, then run run.bat. Open the printed local browser URL, normally http://localhost:8501. Choose Waste Classification and upload a valid JPEG or PNG of one waste object against a clear background.")
    add("code",("Windows launch","setup.bat\nrun.bat"))
    section("10.2","Observe a real classification","The recorded demonstration uses metal/metal392.jpg from the held-out split. The shipped model identifies metal, with its top score displayed as approximately 0.989. The interface also displays glass and cardboard as the next-ranked categories. The screenshot records actual model output; softmax scores are not guarantees of correctness.")
    fig("Real held-out photograph and the resulting material classification","screenshots/classification.png")
    fig("Detail of the genuine CNN prediction and top-three scores","screenshots/cnn_scores.png")
    section("10.3","Change contamination and explain the result","For the predicted metal category, the configured material potential is 90. At manual contamination 20, suitability is 80.56/100 and High. Set contamination to 100: the result becomes 19.44/100 and Low, with updated guidance. The material prediction stays fixed while the fuzzy calculation changes. Open the calculation expander to inspect memberships, rules and centroid.")
    fig("Actual metal decision at manual contamination 20","screenshots/decision_support.png")
    fig("Same material prediction with manual contamination 100","screenshots/decision_support_contaminated.png")
    section("10.4","Complete the viva demonstration","Open the Fuzzy Logic Laboratory to vary both assumptions independently. Explain minimum conjunction, clipped output sets, maximum aggregation and centroid. Open Model Evaluation to show the held-out evidence and minority-class limitations. Return to Dashboard to inspect the real session record, export CSV, and demonstrate clear history. Mention the documented monotonicity counterexamples rather than claiming a universal monotone score.")
    add("h1","REFERENCES","references")
    markdown(sections[28])
    markdown("[9] DIT University. CSN441 Fuzzy Logic and Neural Network, ODD 2026–27, Mini Project assignment brief supplied with this report. Selected topic: Project 4 — Waste Classification Using Images.\n\n[10] SmartWaste AI source and verified artifacts. https://github.com/prxns/SmartWaste-AI . Application/UI reference revision d55982a; documentation prepared 18 November 2026.")
    add("h1","APPENDIX A — SELECTED SOURCE CODE","appendixA")
    markdown("The following numbered listings are excerpts from the implemented project. They illustrate the computational contracts without reproducing the whole repository. Long lines may wrap for printing; the repository contains executable originals.")
    for title,path,func in [
        ("Identical MobileNetV2 preprocessing","smartwaste/preprocessing.py","preprocess_image"),
        ("Frozen base and six-class head","smartwaste/model.py","build_model"),
        ("Mapped prediction scores","smartwaste/model.py","predict"),
        ("Triangular and shoulder membership functions","smartwaste/fuzzy_engine.py","trapezoid"),
        ("Nine min/max rules and aggregate","smartwaste/fuzzy_engine.py","infer"),
        ("Piecewise-linear centroid","smartwaste/fuzzy_engine.py","centroid")]:
        code=extract_function(path,func)
        if func=="infer":code=code[code.index("    for i, material"):code.index("    return FuzzyResult")]
        if func=="trapezoid":code=extract_function(path,"triangle")+"\n\n"+code
        add("code",(title+" — "+path,code))
    add("h1","APPENDIX B — REPRODUCIBILITY AND SUBMISSION","appendixB")
    section("B.1","Commands and key artifacts","The checkpoint is already supplied, so a demonstration does not require retraining. The training and evaluation commands below reproduce the documented pipeline when the dataset and dependencies are available. No retraining was performed for this report.")
    add("code",("Reproducible project commands","setup.bat\nrun.bat\npython -m pytest -q\npython -m scripts.download_dataset\npython -m scripts.train_model --epochs 12 --threads 2\npython -m scripts.evaluate_model"))
    markdown(table(["Artifact","Purpose"],[["models/smartwaste.keras","Actual saved checkpoint"],["models/class_names.json","Fixed original label order"],["models/training_config.json","Recorded run settings"],["results/splits.csv","Disjoint image/hash/split record"],["results/test_predictions.npz","379 saved labels and score vectors"],["results/metrics.json","Measured test metrics and hashes"],["docs/FUZZY_VALIDATION.md","Analytic checks and monotonicity limitations"],["docs/THIRD_PARTY_NOTICES.md","Dataset and framework attribution"]]))
    section("B.2","Provenance and attribution","Model SHA-256: "+metrics["model_sha256"]+"\n\nSplit SHA-256: "+metrics["split_sha256"]+"\n\nTrashNet is attributed to Gary Thung and Mindy Yang. The upstream MIT notice is retained in docs/THIRD_PARTY_NOTICES.md. TensorFlow and Keras retain their upstream Apache 2.0 notices. ImageNet images are not redistributed. Application screenshots in this report are genuine recorded or freshly captured views of the implemented application.")
    section("B.3","Submission preparation","Replace the two students' placeholder names and ERP IDs, faculty name, programme and department before submission. No third student entry or signature is invented. The supplied assignment asks for a PDF and suggests the filename Miniproject_FL&NN_ERPID.docx; after completing the details, use the naming convention required by the LMS for the final submission. This report records the completed implementation without making a claim about submission acceptance or deadline extensions.")
    # Assign stable global figure, table and listing numbers.
    counts={"image":0,"table":0,"code":0};current="Summary";enriched=[]
    for kind,text,key in blocks:
        if kind in ("h1","h2"):current=text
        if kind in counts:
            counts[kind]+=1
            prefix={"image":"Figure","table":"Table","code":"Listing"}[kind]
            if kind=="image":title=text[0]
            elif kind=="code":title=text[0]
            else:title=current
            key=kind+str(counts[kind])
            enriched.append((kind,text,key,f"{prefix} {counts[kind]}. {title}"))
        else:enriched.append((kind,text,key,None))
    return enriched,metrics

def inline(p,text):
    for part in re.split(r"(\*\*[^*]+\*\*|"+BT+r"[^"+BT+r"]+"+BT+r"|https?://[^\s]+)",text):
        if part.startswith("http"):
            target=part.rstrip(".,")
            link=OxmlElement("w:hyperlink")
            from docx.opc.constants import RELATIONSHIP_TYPE as RT
            link.set(qn("r:id"),p.part.relate_to(target,RT.HYPERLINK,is_external=True))
            r=OxmlElement("w:r");pr=OxmlElement("w:rPr");color=OxmlElement("w:color");color.set(qn("w:val"),GREEN);pr.append(color);r.append(pr)
            t=OxmlElement("w:t");t.text=target;r.append(t);link.append(r);p._p.append(link)
        elif part.startswith("**"):p.add_run(part[2:-2]).bold=True
        elif part.startswith(BT):r=p.add_run(part[1:-1]);r.font.name="Liberation Mono";r.font.size=Pt(9)
        else:p.add_run(part)
def field(p,instruction):
    f=OxmlElement("w:fldSimple");f.set(qn("w:instr"),instruction);p._p.append(f)
def document(blocks,pages=None):
    pages=pages or {};d=Document()
    s=d.sections[0];s.page_width=Inches(8.2677);s.page_height=Inches(11.6929)
    s.top_margin=Inches(.8);s.bottom_margin=Inches(.75);s.left_margin=Inches(.85);s.right_margin=Inches(.8)
    s.header_distance=s.footer_distance=Inches(.35);s.different_first_page_header_footer=True
    for style in ("Normal","Title","Heading 1","Heading 2","Caption"):
        st=d.styles[style];st.font.name="Liberation Serif";st.font.color.rgb=RGBColor(0,0,0)
    st=d.styles["Normal"];st.font.size=Pt(11);st.paragraph_format.line_spacing=1.18;st.paragraph_format.space_after=Pt(7)
    for name,size in [("Title",25),("Heading 1",17),("Heading 2",12)]:
        st=d.styles[name];st.font.size=Pt(size);st.font.color.rgb=RGBColor.from_string(GREEN);st.paragraph_format.keep_with_next=True
        st.paragraph_format.space_before=Pt(12);st.paragraph_format.space_after=Pt(8)
    d.styles["Caption"].font.size=Pt(9);d.styles["Caption"].font.italic=True
    for b in list(d.styles.element.xpath(".//w:pBdr")):b.getparent().remove(b)
    hp=s.header.paragraphs[0];hp.add_run("SMARTWASTE AI  |  CSN441").font.size=Pt(8);hp.alignment=2
    fp=s.footer.paragraphs[0];fp.add_run("DIT University • ODD 2026–27").font.size=Pt(8);fp.add_run(" "*10+"Page ");field(fp,"PAGE");fp.alignment=2
    d.core_properties.title="SmartWaste AI — CSN441 Final Project Report"
    d.core_properties.author="[STUDENT 1 NAME]; [TO BE FILLED IN LATER]"
    d.core_properties.subject="Project 4 — Waste Classification Using Images"
    def para(t,style=None):
        p=d.add_paragraph(style=style);inline(p,t);return p
    def newpage(title):
        d.add_page_break();d.add_heading(title,1)
    def tbl(rows):
        t=d.add_table(rows=0,cols=len(rows[0]));t.style="Table Grid"
        for ri,row in enumerate(rows):
            cells=t.add_row().cells
            trpr=t.rows[-1]._tr.get_or_add_trPr();trpr.append(OxmlElement("w:cantSplit"))
            if ri==0:trpr.append(OxmlElement("w:tblHeader"))
            for cell,value in zip(cells,row):
                p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(4);p.paragraph_format.space_before=Pt(3)
                inline(p,str(value).replace(BT,""))
                for run in p.runs:run.font.size=Pt(9)
                if ri==0:
                    sh=OxmlElement("w:shd");sh.set(qn("w:fill"),"E9F0E5");cell._tc.get_or_add_tcPr().append(sh)
                    for run in p.runs:run.bold=True
        d.add_paragraph()
    # Formal cover; text branding is used rather than inventing a university logo.
    for title,size in [("DIT UNIVERSITY",24),("CSN441 — Fuzzy Logic and Neural Network",13),("ODD 2026–27  |  Mini Project",11)]:
        p=d.add_paragraph();p.alignment=1;r=p.add_run(title);r.bold=True;r.font.size=Pt(size);r.font.color.rgb=RGBColor.from_string(GREEN)
    d.add_paragraph()
    p=d.add_paragraph();p.alignment=1;r=p.add_run("SmartWaste AI");r.bold=True;r.font.size=Pt(28);r.font.color.rgb=RGBColor.from_string(GREEN)
    p=d.add_paragraph("CNN-Based Waste Classification with Mamdani Fuzzy Logic Decision Support for Intelligent Waste Segregation");p.alignment=1
    for r in p.runs:r.font.size=Pt(17)
    p=d.add_paragraph("Project 4 — Waste Classification Using Images");p.alignment=1;p.runs[0].bold=True
    d.add_paragraph()
    t=d.add_table(rows=1,cols=2);t.style="Table Grid"
    for i,c in enumerate(t.rows[0].cells,1):
        p=c.paragraphs[0];p.add_run(f"Student {i}").bold=True
        c.add_paragraph("Name: [STUDENT 1 NAME]" if i==1 else "Name: [TO BE FILLED IN LATER]")
        c.add_paragraph("ERP ID: [STUDENT 1 ERP ID]" if i==1 else "ERP ID: [TO BE FILLED IN LATER]")
        c.add_paragraph()
    d.add_paragraph()
    for text in ("Faculty / Supervisor: [FACULTY NAME]","Programme / Department: [PROGRAMME / DEPARTMENT]"):
        p=d.add_paragraph(text);p.alignment=1
    p=d.add_paragraph("Project repository: https://github.com/prxns/SmartWaste-AI");p.alignment=1
    newpage("Student Details")
    tbl([["Group member","Name","ERP ID"],["Student 1","[STUDENT 1 NAME]","[STUDENT 1 ERP ID]"],["Student 2","[TO BE FILLED IN LATER]","[TO BE FILLED IN LATER]"]])
    para("Institution: DIT University\nCourse: CSN441 — Fuzzy Logic and Neural Network\nAcademic session: ODD 2026–27\nSelected topic: Project 4 — Waste Classification Using Images")
    d.add_heading("Assignment-stage coverage",2)
    tbl([["Required stage","Report coverage"]]+[[a,b] for a,b in [("Problem definition","Chapter 1"),("Dataset collection","Chapter 3"),("Data preprocessing","Chapter 3"),("Model development","Chapter 4"),("Model training","Chapter 4"),("Model evaluation","Chapter 7"),("Result analysis","Chapter 8"),("Conclusion","Chapter 9"),("Future scope","Chapter 9"),("Demonstration","Chapter 10")]])
    newpage("Declaration")
    para("We declare that this report documents the SmartWaste AI implementation submitted for the CSN441 mini project. Dataset, pretrained-model, software and research sources are acknowledged. The reported evaluation values are drawn from the recorded project artifacts, and limitations are disclosed.")
    para("We understand that this declaration must be reviewed and signed by both group members after completing the student details. No signature, faculty approval or institutional endorsement has been supplied or reproduced in this document.")
    para("Student 1 signature: ____________________\nName: [STUDENT 1 NAME]\nERP ID: [STUDENT 1 ERP ID]")
    para("Student 2 signature: ____________________\nName: [TO BE FILLED IN LATER]\nERP ID: [TO BE FILLED IN LATER]\nDate: [DATE]")
    newpage("Acknowledgement")
    para("We acknowledge the CSN441 course framework and the open research and software that make this academic prototype reproducible. We thank the authors of TrashNet, MobileNetV2, TensorFlow/Keras, NumPy, Pillow, Scikit-learn, Streamlit, Plotly and Matplotlib. Their contributions are identified in the references and third-party notices.")
    para("Faculty / Supervisor: [FACULTY NAME]. This acknowledgement may be personalised by the two students before submission. It does not constitute a faculty endorsement or a certification of the system's recycling recommendations.")
    newpage("Abstract")
    para("SmartWaste AI combines a CNN material classifier with an inspectable Mamdani Fuzzy Inference System for recycling decision support. The application identifies cardboard, glass, metal, paper, plastic and trash from a single uploaded image. An ImageNet-pretrained MobileNetV2 with a frozen base and a trained six-class head operates on 160 × 160 RGB inputs.")
    para("The resized TrashNet archive contains 2,527 images. Conflicting duplicate-label handling retains 2,521 unique, unambiguous images, split into 1,764 training, 378 validation and 379 test examples. The selected epoch-10 checkpoint attains 81.53% held-out accuracy, 77.73% macro precision, 78.21% macro recall and 77.70% macro F1. These are measured prototype results, not deployment guarantees.")
    para("The CNN category selects an illustrative material potential; the user supplies contamination. Nine rules use minimum conjunction and implication, maximum aggregation and centroid defuzzification. The interface exposes membership degrees, firing strengths and the numerical output. Suitability is a decision score rather than a measured recycling rate. Documented global monotonicity counterexamples remain explicit. A local five-section Streamlit interface provides classification, laboratory exploration, evaluation, session history and CSV export.")
    para("Keywords: waste classification; MobileNetV2; transfer learning; Mamdani inference; centroid; explainable decision support.")
    toc=[(key,text.replace("CHAPTER ","").title()) for kind,text,key,cap in blocks if kind=="h1"]
    indexes=[("Table of Contents",toc),("List of Figures",[(key,cap.replace("Figure ","",1)) for kind,text,key,cap in blocks if kind=="image"]),("List of Tables",[(key,cap.replace("Table ","",1)) for kind,text,key,cap in blocks if kind=="table"])]
    for title,items in indexes:
        newpage(title)
        for key,label in items:
            p=para(label);p.paragraph_format.tab_stops.add_tab_stop(Inches(6.3),2,1)
            p.add_run("\t"+str(pages.get(key,"—")))
            p.paragraph_format.space_after=Pt(4)
            for run in p.runs:run.font.size=Pt(9.5)
    newpage("Abbreviations")
    tbl([["Term","Meaning"],["AI","Artificial Intelligence"],["CNN","Convolutional Neural Network"],["FIS","Fuzzy Inference System"],["RGB","Red, Green and Blue"],["GAP","Global Average Pooling"],["CSV","Comma-Separated Values"],["ERP","Enterprise Resource Planning student identifier"],["TP / FP / FN","True positives / false positives / false negatives"],["F1","Harmonic mean of precision and recall"],["SHA-256","Secure Hash Algorithm, 256-bit digest"],["CPU","Central Processing Unit"],["UI","User Interface"],["XFAIL","Documented expected test failure"]])
    for kind,text,key,caption in blocks:
        if kind=="h1":newpage(text)
        elif kind=="h2":d.add_heading(text,2)
        elif kind=="p":para(text)
        elif kind=="table":
            p=para(caption,"Caption");p.paragraph_format.keep_with_next=True;tbl(text)
        elif kind=="image":
            title,rel=text;p=(OUT/rel).resolve()
            if not p.is_file():raise FileNotFoundError(p)
            with Image.open(p) as im:w,h=im.size
            scale=min(6.25/w,6.9/h)
            ip=d.add_paragraph();ip.alignment=1;ip.paragraph_format.keep_with_next=True
            ip.add_run().add_picture(str(p),width=Inches(w*scale),height=Inches(h*scale))
            cp=para(caption,"Caption");cp.alignment=1
        elif kind=="code":
            title,code=text;p=para(caption,"Caption");p.paragraph_format.keep_with_next=True
            for line in code.splitlines():
                p=d.add_paragraph();p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1
                r=p.add_run(line or " ");r.font.name="Liberation Mono";r.font.size=Pt(8)
            d.add_paragraph()
    d.save(OUT/(NAME+".docx"))
    # Editable text source mirrors all technical blocks and source assets.
    md=["# SmartWaste AI — Final Project Report","DIT University | CSN441 | ODD 2026–27","Project 4 — Waste Classification Using Images","Student 1: [STUDENT 1 NAME] | ERP ID: [STUDENT 1 ERP ID]","Student 2: [TO BE FILLED IN LATER] | ERP ID: [TO BE FILLED IN LATER]","Faculty / Supervisor: [FACULTY NAME]"]
    for kind,text,key,cap in blocks:
        if kind in ("h1","h2"):md.append(("# " if kind=="h1" else "## ")+text)
        elif kind=="p":md.append(text)
        elif kind=="table":md.extend([cap,table(text[0],text[1:])])
        elif kind=="image":md.append(f"![{cap}]({text[1]})")
        elif kind=="code":md.append(cap+"\n\n"+BT*3+"python\n"+text[1]+"\n"+BT*3)
    (OUT/(NAME+".md")).write_text("\n\n".join(md)+"\n",encoding="utf-8")

def convert():
    p=subprocess.run(["libreoffice","-env:UserInstallation=file:///tmp/smartwaste-final-report-lo","--headless","--convert-to","pdf","--outdir",str(OUT),str(OUT/(NAME+".docx"))],capture_output=True,text=True,timeout=180)
    if p.returncode:raise RuntimeError(p.stdout+p.stderr)
    if not (OUT/(NAME+".pdf")).is_file():raise RuntimeError(p.stdout+p.stderr)

def paginate(blocks):
    import fitz
    pdf=fitz.open(OUT/(NAME+".pdf"));mapping={}
    headings={key:text for kind,text,key,cap in blocks if kind=="h1"}
    for i,page in enumerate(pdf,1):
        text=" ".join(page.get_text().split())
        large=" ".join(span["text"] for b in page.get_text("dict")["blocks"] for line in b.get("lines",[]) for span in line["spans"] if span["size"]>=16)
        large=" ".join(large.split())
        for key,title in headings.items():
            if title in large:mapping.setdefault(key,i)
        for match in re.finditer(r"\b(Figure|Table)\s+(\d+)\.",text):
            mapping.setdefault(("image" if match[1]=="Figure" else "table")+match[2],i)
    assert mapping.get("chapter1",0)>8, mapping
    return mapping

def verify_and_preview(blocks):
    import fitz
    pdf=fitz.open(OUT/(NAME+".pdf"));texts=[p.get_text() for p in pdf]
    full="\n".join(texts)
    assert 25<=len(pdf)<=65, len(pdf)
    for title in [t for k,t,key,c in blocks if k=="h1"]:assert title in " ".join(full.split()),title
    assert all(len(t.strip())>40 for t in texts), "Accidental blank page"
    assert "Student 3" not in full and "81.53%" in full and "77.70%" in full
    assert "monotonicity" in full and "expected" in full
    issues=[]
    for i,p in enumerate(pdf,1):
        for b in p.get_text("dict")["blocks"]:
            for line in b.get("lines",[]):
                for span in line["spans"]:
                    x0,y0,x1,y1=span["bbox"]
                    if x0<10 or y0<10 or x1>p.rect.width-8 or y1>p.rect.height-8:issues.append([i,span["text"]])
    assert not issues,issues
    target=ROOT/"report-preview-final";target.mkdir(exist_ok=True)
    thumbs=[]
    for i,p in enumerate(pdf,1):
        pix=p.get_pixmap(matrix=fitz.Matrix(1.1,1.1),alpha=False);pix.save(target/f"page-{i:02}.png")
        im=Image.open(io.BytesIO(pix.tobytes("png")));im.thumbnail((260,370));thumbs.append(im.copy())
    for batch in range(0,len(thumbs),12):
        board=Image.new("RGB",(4*280,3*400),"#DADDD7")
        from PIL import ImageDraw
        draw=ImageDraw.Draw(board)
        for j,im in enumerate(thumbs[batch:batch+12]):
            x=(j%4)*280+10;y=(j//4)*400+20
            board.paste(im,(x,y));draw.text((x,y-15),f"Page {batch+j+1}",fill="black")
        board.save(target/f"contact-{batch//12+1}.png")
    record={"pages":len(pdf),"figures":sum(k=="image" for k,*_ in blocks),"tables":sum(k=="table" for k,*_ in blocks),"model_sha256":sha("models/smartwaste.keras"),"pdf_sha256":sha("docs/"+NAME+".pdf"),"checks":"chapters, figures, tables, metrics, placeholders, no blank pages, text bounds"}
    (target/"validation.json").write_text(json.dumps(record,indent=2));print("REPORT_VALIDATION",json.dumps(record))
    for p in sorted(target.glob("contact-*.png")):
        encoded=base64.b64encode(p.read_bytes()).decode()
        for n,o in enumerate(range(0,len(encoded),60000)):print("REPORT_REVIEW",p.name,n,encoded[o:o+60000])
    # Files are transmitted as text only for this restricted authoring environment.
    for name in [NAME+".pdf",NAME+".docx",NAME+".md","screenshots/cnn_scores.png","screenshots/fuzzy_aggregate.png","screenshots/fuzzy_rules.png"]:
        encoded=base64.b64encode((OUT/name).read_bytes()).decode()
        for n,o in enumerate(range(0,len(encoded),60000)):print("REPORT_FILE",name,n,encoded[o:o+60000])
def main():
    prepare_screens()
    blocks,metrics=content();document(blocks);convert();mapping=paginate(blocks)
    for _ in range(3):
        document(blocks,mapping);convert();new=paginate(blocks)
        if new==mapping:break
        mapping=new
    else:raise RuntimeError("Contents pagination did not stabilise")
    assert len(mapping)==sum(k in ("h1","image","table") for k,*_ in blocks),mapping
    verify_and_preview(blocks)
if __name__=="__main__":main()
