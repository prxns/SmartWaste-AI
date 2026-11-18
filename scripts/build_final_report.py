"""Build final academic documentation from the completed SmartWaste AI project."""
import ast, csv, hashlib, io, json, re, shutil, sys, textwrap, zipfile
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smartwaste.fuzzy_engine import infer
from config import MATERIAL_POTENTIAL, RULE_TABLE
DOCS=ROOT/"docs"; ASSETS=DOCS/"screenshots"; ASSETS.mkdir(exist_ok=True)
OUT=DOCS/"SmartWaste_AI_Final_Project_Report"
WORK=ROOT/"report-preview";WORK.mkdir(exist_ok=True)
M=json.loads((ROOT/"results/metrics.json").read_text())
D=json.loads((ROOT/"results/dataset.json").read_text())
C=json.loads((ROOT/"models/training_config.json").read_text())
MAN=json.loads((ROOT/"models/model_manifest.json").read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ROOT/"models/smartwaste.keras")==M["model_sha256"]==MAN["model_sha256"]
assert sha(ROOT/"results/splits.csv")==M["split_sha256"]
S={}
original=(DOCS/"SmartWaste_AI_Project_Report.md").read_text()
for m in re.finditer(r"^## (\d+) ([^\n]+)\n(.*?)(?=^## \d+ |\Z)",original,re.M|re.S):
    t=re.sub(r"<!--.*?-->","",m[3],flags=re.S)
    t=re.sub(r"!\[.*?\]\(.*?\)","",t)
    t="\n".join(l for l in t.splitlines() if not l.startswith("|"))
    S[int(m[1])]=t.strip()
for pth in (ROOT/"assets/screenshots").glob("*.png"):shutil.copy2(pth,ASSETS/pth.name)
for n in ["architecture.png","memberships.png","fuzzy_example.png"]:shutil.copy2(ROOT/"assets/figures"/n,ASSETS/n)
for n in ["confusion_matrix.png","accuracy.png","loss.png"]:shutil.copy2(ROOT/"results"/n,ASSETS/n)
rows=list(csv.DictReader((ROOT/"results/splits.csv").open()))
test=[r for r in rows if r["split"]=="test"]
pr=np.load(ROOT/"results/test_predictions.npz");y=pr["labels"];scores=pr["scores"];yp=scores.argmax(1)
assert np.array_equal(y,np.array([int(r["label"]) for r in test]))
assert len(y)==379 and (y==yp).sum()==309
blob=urlopen(D["source"],timeout=120).read()
assert hashlib.sha256(blob).hexdigest()==D["archive_sha256"]
arc=zipfile.ZipFile(io.BytesIO(blob))
font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",20)
sheet=Image.new("RGB",(1560,1000),"white");draw=ImageDraw.Draw(sheet);examples=[]
for k,i in enumerate(list(np.flatnonzero(y==yp)[:3])+list(np.flatnonzero(y!=yp)[:3])):
    r=test[int(i)];name=r["path"];member=next(n for n in arc.namelist() if n.endswith("/"+name) or n==name)
    im=Image.open(io.BytesIO(arc.read(member))).convert("RGB");im.thumbnail((460,350))
    xx=(k%3)*520;yy=(k//3)*500
    sheet.paste(im,(xx+(520-im.width)//2,yy+10))
    a=M["class_names"][int(y[i])];b=M["class_names"][int(yp[i])];v=f"{scores[i,yp[i]]:.4f}"
    draw.text((xx+15,yy+380),f"Actual: {a} | Predicted: {b}",font=font,fill="black")
    draw.text((xx+15,yy+415),f"{Path(name).name} | Score {v}",font=font,fill="black")
    examples.append([Path(name).name,a,b,v])
sheet.save(ASSETS/"held_out_examples.png")
pages=[]
def page(title,*blocks):pages.append((title,blocks))
def p(t):return ("p",t)
def h(t):return ("h",t)
def s(n):return p(S[n])
def f(cap,name,height=3.4):return ("f",cap,name,height)
def t(cap,heads,rows,widths=None):return ("t",cap,heads,rows,widths)
def c(cap,path,name):
    source=(ROOT/path).read_text();tree=ast.parse(source)
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    return ("c",cap,path,"\n".join(source.splitlines()[node.lineno-1:node.end_lineno]))
def rawcode(cap,path,text):return ("c",cap,path,text)

page("SmartWaste AI",
 p("DIT UNIVERSITY"),p("CSN441 - Fuzzy Logic and Neural Network"),
 p("ODD 2026-27 | Final Year Mini Project"),
 p("Project 4 - Waste Classification Using Images"),
 p("CNN-Based Waste Classification with Mamdani Fuzzy Logic Decision Support for Intelligent Waste Segregation"),
 h("Student 1"),p("Name: [STUDENT 1 NAME]\nERP ID: [STUDENT 1 ERP ID]"),
 h("Student 2"),p("Name: [TO BE FILLED IN LATER]\nERP ID: [TO BE FILLED IN LATER]"),
 p("Faculty / Supervisor: [FACULTY OR SUPERVISOR NAME]"),
 p("Programme / Department: [PROGRAMME AND DEPARTMENT]"),p("November 2026"))
page("Student details and declaration",
 t("Table F1  Student details",["Group member","Name","ERP ID"],[
 ["Student 1","[STUDENT 1 NAME]","[STUDENT 1 ERP ID]"],
 ["Student 2","[TO BE FILLED IN LATER]","[TO BE FILLED IN LATER]"]]),
 h("Declaration"),
 p("We declare that this report describes our SmartWaste AI mini project for CSN441 Fuzzy Logic and Neural Network, ODD 2026-27. The implementation, experimental results and limitations are presented to the best of our knowledge. The dataset, pretrained architecture, software libraries and research used in the work are acknowledged."),
 p("The CNN uses pretrained ImageNet features and a trained six-class classifier head. We do not claim authorship of TrashNet or MobileNetV2. The fuzzy potentials are illustrative design assumptions, and suitability is a decision-support score rather than a measured recycling rate."),
 p("Student 1 signature: ____________________    Date: ____________________"),
 p("Student 2 signature: ____________________    Date: ____________________"),
 h("Acknowledgement"),
 p("We thank our course faculty for the opportunity to apply neural networks and fuzzy logic in a complete project. We acknowledge Gary Thung and Mindy Yang for TrashNet, the MobileNetV2 authors, and the maintainers of TensorFlow, Keras, NumPy, Streamlit and the supporting open-source libraries. Their datasets, publications and software provide the foundation for this implementation."),
 p("Faculty / Supervisor: [FACULTY OR SUPERVISOR NAME]"))
page("Abstract",s(1),
 p("The original archive contains 2,527 images; 2,521 unambiguous images remain after conflicting-label duplicates are excluded. The seeded split contains 1,764 training, 378 validation and 379 test images. The actual saved model achieves 81.53% accuracy, 77.73% macro precision, 78.21% macro recall and 77.70% macro F1 on the isolated test partition."),
 p("The project demonstrates a working academic prototype whose reasoning and empirical results can be inspected. Contamination is manually entered, the classifier is limited to the six original labels, and the fuzzy score is conditional on declared assumptions."),
 p("Keywords: waste classification; MobileNetV2; transfer learning; TrashNet; Mamdani inference; centroid defuzzification; Streamlit."))
page("Table of contents",("toc",))
page("List of figures",("figlist",))
page("List of tables and abbreviations",("tablist",),
 p("AI: Artificial intelligence. CNN: Convolutional neural network. FIS: Fuzzy inference system. GAP: Global average pooling. RGB: Red, green and blue. CSV: Comma-separated values. GUI: Graphical user interface. SHA-256: 256-bit Secure Hash Algorithm. ERP ID: Institutional student identifier."))
page("Chapter 1  Introduction",h("1.1 Background and problem statement"),s(2),s(3),
 h("1.2 Objectives"),s(4),h("1.3 Scope and intended application"),s(5))
page("Chapter 2  Literature review and existing approaches",s(6),
 t("Table 2.1  Existing approaches and their role",["Approach","Relevance and limitation"],[
 ["Manual segregation","Uses human material and contamination judgement; depends on experience."],
 ["CNN classification","Learns visual material features; does not determine collection eligibility."],
 ["MobileNetV2 transfer learning","Reuses efficient pretrained features with a task-specific head."],
 ["Mamdani reasoning","Exposes linguistic assumptions and rule activations; needs independent validation."]],[1.7,4.4]),
 p("The project integrates these methods with image validation, an explicit manual contamination input and saved evaluation evidence. It did not run a controlled benchmark against other architectures, so superiority over alternatives is not claimed."))
dist=[[n.title(),D["raw_class_distribution"][n],D["class_distribution"][n],D["raw_class_distribution"][n]-D["class_distribution"][n]] for n in M["class_names"]]
page("Chapter 3  Dataset and preprocessing",
 h("3.1 Dataset source and collection"),
 p("TrashNet, created by Gary Thung and Mindy Yang [1], supplies the six original categories. Its resized archive contains 2,527 images and occupies 42,834,870 bytes (40.85 MiB). Individual objects against simple backgrounds support a controlled classification prototype but do not establish performance in mixed municipal waste."),
 t("Table 3.1  Original and validated image counts",["Category","Original","Validated","Excluded"],dist+[["Total",2527,2521,6]]),
 h("3.2 Data validation and conflicting labels"),
 p("The pipeline decodes each image and fingerprints RGB bytes plus dimensions. Three identical-image pairs have contradictory category labels. All six images are excluded before splitting, avoiding an arbitrary choice of ground truth. The retained modelling inventory contains 2,521 unique, unambiguous images."),
 p("Excluded pairs are glass115.jpg / metal91.jpg, glass176.jpg / plastic152.jpg, and glass389.jpg / plastic332.jpg. Full paths, decoded hashes and reasons are recorded in results/dataset.json and the audit metadata."),
 p("Before extraction, the downloader checks ZIP paths, symlinks, encryption, duplicate targets and total size limits. Only a successfully validated six-folder dataset is installed."))
counts={}
for r in rows:counts[(r["category"],r["split"])]=counts.get((r["category"],r["split"]),0)+1
page("3.3 Partitioning and preprocessing",
 t("Table 3.2  Actual seeded stratified split",["Category","Train","Validation","Test"],
 [[n.title(),counts[n,"train"],counts[n,"validation"],counts[n,"test"]] for n in M["class_names"]]+[["Total",1764,378,379]]),
 s(11),
 p("The augmentation settings are RandomFlip(horizontal), RandomRotation(0.08) and RandomZoom(0.10), with reflected borders. Rotation is a fraction of a full turn, approximately 28.8 degrees in either direction. Balanced class weights use N_train/(6 x N_train,class), calculated only from training labels. Trash receives weight 3.0625 and paper 0.7067."))
page("Chapter 4  System architecture and methodology",h("4.1 Overall workflow"),s(7),
 f("Figure 4.1  System architecture from the actual project","architecture.png",3.3),
 p("The UI calls distinct preprocessing, CNN, fuzzy, chart and recommendation modules. The pure NumPy FIS is independent of TensorFlow and Streamlit. A saved checkpoint and class mapping connect the offline training path to application inference; a UI action never retrains the network."))
page("4.2 CNN architecture and transfer learning",s(12),s(13),
 t("Table 4.1  Implemented model layers",["Stage","Shape or configuration"],[
 ["Input","160 x 160 x 3 preprocessed RGB"],
 ["MobileNetV2","ImageNet weights; include_top=False; alpha=1.0; frozen"],
 ["Final convolutional map","5 x 5 x 1280"],
 ["Global average pooling","1280 features"],
 ["Dropout","Rate 0.3"],
 ["Dense softmax","6 class scores; 7,686 trainable head parameters"]],[2.1,4.0]))
page("4.3 Training configuration and model inference",
 t("Table 4.2  Recorded training configuration",["Parameter","Actual value"],[
 ["Image size / batch size","160 x 160 RGB / 32"],
 ["Seed / CPU threads","42 / two intra- and inter-op threads"],
 ["Initialization","ImageNet-pretrained MobileNetV2; frozen convolutional base"],
 ["Loss / optimizer","Sparse categorical cross entropy / Adam"],
 ["Initial learning rate","0.001"],
 ["Actual epochs / best epoch","12 completed / epoch 10"],
 ["Checkpoint selection","Minimum validation loss"],
 ["Early stopping","Patience 4; restore best weights"],
 ["Learning-rate reduction","Factor 0.5; patience 2; minimum 0.00001"],
 ["Class weights","Inverse-frequency weights from training labels"],
 ["Recorded runtime","Python 3.11.18; TensorFlow 2.20.0"]],[2.1,4.0]),
 h("4.4 Inference contract"),
 p("The loader validates the six-class order and square RGB input. It loads the real Keras checkpoint in safe mode, calls the model with training=False, and requires six finite, nonnegative scores summing approximately to one. Descending score order determines the category and top-three outputs."),
 p("Softmax scores are uncalibrated outputs, not guaranteed correctness probabilities. The predicted category selects material potential from config.py. Contamination is entered independently by the user and never inferred by the CNN. These two values then enter the fuzzy engine."))
page("Chapter 5  Fuzzy logic implementation",h("5.1 Inputs and material assumptions"),s(14),
 t("Table 5.1  Configured material potentials",["CNN category","Potential on 0 to 100"],
 [[n.title(),MATERIAL_POTENTIAL[n]] for n in M["class_names"]]),
 p("Material recyclability potential and contamination level are the two inputs. Recycling suitability is the output. All use [0,100]. Potential is an academic category-level assumption, contamination is a manual assessment, and suitability is a rule-system output. None measures an actual mass-recovery percentage."),
 p("Combining the CNN with a separate fuzzy stage preserves this distinction: recognition supplies a material category; human input supplies its condition; linguistic rules express the decision-support assumptions. The laboratory makes the reasoning visible."))
page("5.2 Membership functions and fuzzification",
 t("Table 5.2  Exact membership definitions for inputs and output",["Set","Shape","Parameters"],[
 ["Low","Trapezoid","[0, 0, 25, 50]"],["Medium","Triangle","[25, 50, 75]"],["High","Trapezoid","[50, 75, 100, 100]"]]),
 f("Figure 5.1  Actual configured membership functions","memberships.png",2.65),s(15))
rules=[]
for i,a in enumerate(["Low","Medium","High"]):
 for j,b in enumerate(["Low","Medium","High"]):rules.append([f"R{i*3+j+1}",a,b,RULE_TABLE[i][j]])
page("5.3 Rule base and Mamdani inference",
 t("Table 5.3  All nine implemented IF THEN rules",["Rule","Material","Contamination","Suitability"],rules),
 s(16),s(17))
cases=[(80,20),(90,20),(90,100),(37.5,62.5)]
page("5.4 Centroid defuzzification and numerical results",s(18),
 f("Figure 5.2  Computed aggregate and centroid","fuzzy_example.png",2.55),
 t("Table 5.4  Outputs computed by the actual engine",["Potential","Contamination","Suitability","Status"],
 [[a,b,f"{infer(a,b).score:.6f}",infer(a,b).status] for a,b in cases]),
 p("For potential 80 and contamination 20, only R7 fires at one. The High trapezoid has area 37.5 and first moment 3,020.833333, giving centroid 80.555556. At potential 90 and contamination 100, R9 activates the full Low set: first moment 729.166667 divided by area 37.5 gives 19.444444."))
page("Chapter 6  Application development",h("6.1 Local Streamlit application and dashboard"),
 p("The five sections are Dashboard, Waste Classification, Fuzzy Logic Laboratory, Model Evaluation, and About & Methodology. The application uses native Streamlit widgets, Plotly charts and the existing forest-green visual theme."),
 f("Figure 6.1  Real dashboard with session activity","dashboard.png",3.8),s(19),
 p("The dashboard displays actual session totals, category distribution and recent history. CSV export and clearing operate on the session. A contamination change updates the same activity instead of adding a CNN prediction; changing the upload invalidates stale output."))
page("6.2 Classification input and CNN results",
 f("Figure 6.2  Real classification screen before upload","classification_empty.png",3.2),
 f("Figure 6.3  Real held-out upload and top-three scores","classification.png",3.65),
 p("The actual metal392.jpg image is 512 x 384 pixels. The application predicts Metal with displayed scores metal 0.989, glass 0.004 and cardboard 0.003. Three omitted class scores and display rounding explain why the top three do not sum to one."))
page("6.3 Fuzzy laboratory and activated rules",
 f("Figure 6.4  Real laboratory inputs and membership plots","fuzzy_laboratory.png",3.65),
 p("The laboratory example uses potential 80 and contamination 20. It displays High suitability 80.56 / 100 and one active rule. Both inputs can be changed independently of the classifier; membership degrees, nine rule strengths, aggregate and centroid are recalculated."),
 f("Figure 6.5  Real laboratory aggregate and activated-rule table","fuzzy_trace.png",3.0))
page("6.4 Evaluation and methodology sections",
 f("Figure 6.6  Real model evaluation dashboard","model_evaluation.png",3.25),
 f("Figure 6.7  Real About and Methodology section","methodology.png",3.25),
 p("Evaluation reads recorded artifacts, while methodology explains the two computational paths and their assumptions. Missing or incompatible models produce explicit errors, never fabricated predictions. Uploaded photos remain in memory."))
page("Chapter 7  Testing and model evaluation",h("7.1 Experimental setup"),s(21),
 h("7.2 Software testing"),s(24),
 p("The completed full-environment project validation reports 34 passed tests, 2 xfailed cases and 6 passed subtests, with no dependency or model skips. The two xfailed cases explicitly document unmet global monotonicity properties; they are neither passing guarantees nor unexpected software failures."),
 p("The Windows and Linux validation records additionally cover actual browser upload, model loading, contamination changes, session CSV export and Streamlit startup. Five sections were checked at 1920 x 1080, 1366 x 768 and 1280 x 720. These completed records are in docs/VALIDATION.md."))
page("7.3 Held-out performance",
 t("Table 7.1  Actual aggregate metrics on 379 test images",["Metric","Measured value"],
 [[label,f"{M[key]*100:.2f}%"] for label,key in [("Accuracy","test_accuracy"),("Macro precision","macro_precision"),("Macro recall","macro_recall"),("Macro F1","macro_f1")]]),
 t("Table 7.2  Per-class metrics rounded to four decimals",["Class","Precision","Recall","F1","Support"],
 [[n.title()]+[f"{M['classification_report'][n][k]:.4f}" for k in ["precision","recall","f1-score"]]+[int(M["classification_report"][n]["support"])] for n in M["class_names"]]),
 p("The saved checkpoint correctly classifies 309 of 379 images, with 70 errors. Accuracy is 309/379 = 0.8153034301. Validation loss selects epoch 10; test data are reserved for evaluation."),
 p("Precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is their harmonic mean. Macro averages weight every category equally. Macro F1 averages class F1 values rather than taking the harmonic mean of macro precision and recall. Cardboard has the highest F1 at 0.9153, paper follows at 0.9032, and trash has F1=0.5000 on only 20 test images."))
page("7.4 Confusion matrix",
 f("Figure 7.1  Actual held-out confusion matrix","confusion_matrix.png",4.3),
 t("Table 7.3  Matrix with actual rows and predicted columns",["Actual","Card.","Glass","Metal","Paper","Plastic","Trash"],
 [[n.title()]+r for n,r in zip(M["class_names"],M["confusion_matrix"])],[1.5,.76,.76,.76,.76,.76,.76]),
 p("Rows are actual labels and columns are predictions. The diagonal sums to 309. Twelve glass images are predicted as plastic and nine as metal. Seven plastic images are predicted as glass and seven as trash. Similar appearance is a plausible explanation, but the matrix alone cannot establish the visual cause of an individual error."))
page("7.5 Training and validation curves",
 f("Figure 7.2  Actual accuracy across 12 training epochs","accuracy.png",2.8),
 f("Figure 7.3  Actual training and validation loss","loss.png",2.8),
 p("Training accuracy rises from 46.43% to 85.71%. Minimum validation loss is 0.570204 at epoch 10, with validation accuracy 80.42%. Validation accuracy peaks separately at epoch 11 at 80.95%; epoch selection follows loss. Epoch 12 has validation loss 0.582194 and accuracy 79.63%."),
 p("Later training improvements without sustained validation-loss improvement suggest a developing generalization gap. No comparison experiment isolates the individual benefit of augmentation, dropout or class weights. The test result remains tied to the selected checkpoint."))
page("Chapter 8  Results and discussion",
 f("Figure 8.1  Correct and incorrect saved held-out predictions","held_out_examples.png",3.5),
 t("Table 8.1  Actual example images and saved predictions",["Image","Actual","Predicted","Top score"],examples),
 p("The first row contains three correct predictions, and the second contains three misclassifications. Images come from the verified original archive. Labels and scores are read from test_predictions.npz and matched to the saved split order; these are existing test outputs, not manually assigned or newly fitted predictions."),
 p("The application example metal392.jpg demonstrates a functioning pipeline, while full-test metrics quantify its overall performance. Small category support and simple backgrounds limit the strength of broader deployment claims."))
page("8.2 Fuzzy results and limitations",
 t("Table 8.2  Monotonicity counterexamples from the actual engine",["Potential","Contamination","Suitability"],
 [[a,b,f"{infer(a,b).score:.6f}"] for a,b in [(15,25),(15,26),(38,100),(39,100)]]),
 p("For potential 15, contamination increasing from 25 to 26 raises suitability from 19.444444 to approximately 19.631579. Only Low consequents are active. Maximum firing strength falls from 1 to 0.96; clipping the asymmetric shoulder at a lower height moves its centroid rightward. At contamination 100, raising potential from 38 to 39 reduces suitability from approximately 21.9119 to 21.6899 through the same geometric mechanism."),
 p("These documented counterexamples are not rounding errors. The specified min AND, clipped implication, max aggregation and centroid calculation remain intact. Two expected-failure tests expose the unmet global properties. The system does not guarantee that every increase in contamination lowers the exact score or every increase in potential raises it."),
 h("8.3 Strengths and limitations"),s(25),
 p("The implementation's strengths are its real checkpoint, saved evaluation provenance, independent fuzzy engine and visible trace. Its suitability assumptions have not been validated against actual recycling outcomes. A future redesign must declare and test any changed membership functions, rule base or inference method."))
page("Chapter 9  Conclusion and future scope",h("9.1 Conclusion"),s(26),
 t("Table 9.1  Objectives and achieved evidence",["Objective","Evidence"],[
 ["Image recognition","Real checkpoint; six-class held-out evaluation"],
 ["Fuzzy decision support","Nine rules; inspectable memberships and centroid"],
 ["Application demonstration","Five sections; actual upload and session CSV export"],
 ["Reproducibility","Splits, configuration, predictions and hashes"],
 ["Honest validation","Per-class errors and monotonicity limitations disclosed"]],[2.0,4.1]),
 h("9.2 Future scope"),s(27))
page("Chapter 10  Demonstration",
 t("Table 10.1  Demonstration procedure",["Step","Action and expected evidence"],[
 ["1 Launch","Run setup.bat once, then run.bat; open localhost:8501."],
 ["2 Upload","Choose a genuine JPEG/PNG, such as held-out metal392.jpg."],
 ["3 Predict","Click Classify image; inspect Metal and the top-three scores."],
 ["4 Inputs","Material potential is 90; contamination is a separate manual slider."],
 ["5 Fuzzy result","At contamination 20, R7 fires at one; suitability is 80.555556."],
 ["6 Change condition","Set contamination to 100; the same Metal prediction now gives Low suitability 19.444444."],
 ["7 Explain","Inspect the membership plots, rule strengths, aggregate and centroid."],
 ["8 Export","Verify one session classification and download its CSV."]],[.7,5.4]),
 f("Figure 10.1  Real material and manual-contamination decision output","decision_support.png",2.6),
 p("Contamination changes update the same classification record. The guidance is qualified by the assumptions and local collection requirements; the fuzzy score is not a recycling percentage."))
page("10.2 Output interpretation and demonstration checks",
 f("Figure 10.2  Real output after contamination is changed to 100","decision_support_contaminated.png",2.8),
 p("The captured material remains Metal, and potential remains 90. At contamination 100, R9 activates the full Low output trapezoid. Area 37.5 and first moment 729.166667 produce centroid 19.444444, displayed as 19.44 / 100. The CNN is not asked to estimate contamination."),
 h("10.3 Laboratory demonstration"),
 p("Enter potential 37.5 and contamination 62.5 in the laboratory. Low and Medium material degrees are both 0.5; Medium and High contamination degrees are both 0.5. Rules R2, R3, R5 and R6 fire. Inspect the clipped Low/Medium aggregate and compare its computed centroid with Table 5.4."),
 h("10.4 Functional checks"),
 p("An invalid input must show a validation message without a fabricated prediction. A new upload invalidates the old active result. Clear History must return the session count to zero. Demonstrate that the laboratory is usable independently of classification."),
 h("10.5 Viva explanation"),
 p("Explain transfer learning, the fixed class mapping, validation-based checkpoint selection and actual held-out metrics. Then distinguish CNN scores, fuzzy memberships and suitability. State that contamination is manual, potentials are illustrative, and global monotonicity is not guaranteed. The appendices identify the actual implementation of these calculations."))
page("References",s(28),
 p("[9] SmartWaste AI implementation and recorded experimental artifacts. https://github.com/prxns/SmartWaste-AI"),
 p("[10] DIT University. CSN441 Fuzzy Logic and Neural Network, ODD 2026-27 Mini Project. Supplied assignment document."),
 h("Dataset and software attribution"),
 p("TrashNet is attributed to Gary Thung and Mindy Yang. MobileNetV2 and its pretrained implementation are credited to their authors and TensorFlow/Keras. The cited fuzzy research supports the mathematical framework, not the project-specific material potentials. Software notices are retained in docs/THIRD_PARTY_NOTICES.md."))
page("Appendix A  CNN and preprocessing implementation",
 c("Listing A.1  Shared resizing","smartwaste/preprocessing.py","resize_pixels"),
 c("Listing A.2  Inference normalization","smartwaste/preprocessing.py","preprocess_image"),
 c("Listing A.3  Actual MobileNetV2 construction","smartwaste/model.py","build_model"),
 p("The convolutional base is frozen and called with training=False. The task head learns the six categories. The shared preprocessing maps raw RGB pixels to MobileNetV2's input range exactly once."))
page("Appendix B  Prediction and membership functions",
 c("Listing B.1  Actual prediction and score validation","smartwaste/model.py","predict"),
 c("Listing B.2  Triangular membership","smartwaste/fuzzy_engine.py","triangle"),
 c("Listing B.3  Trapezoidal membership with shoulders","smartwaste/fuzzy_engine.py","trapezoid"))
centroid=c("Listing C.2  Centroid area and first moment excerpt","smartwaste/fuzzy_engine.py","centroid")
page("Appendix C  Inference and centroid",
 c("Listing C.1  Actual nine-rule inference","smartwaste/fuzzy_engine.py","infer"),
 rawcode(centroid[1],centroid[2],"\n".join(centroid[3].splitlines()[-8:])),
 p("The preceding centroid lines validate the arrays and increasing output universe. The excerpt calculates exact area and first moment for the polygon defined by sampled linear segments. Zero aggregate area raises an error."))
page("Appendix D  Execution and reproducibility",
 rawcode("Listing D.1  Windows setup and launch","Repository root","setup.bat\nrun.bat\n\nREM Alternative after setup\n.venv\\Scripts\\activate\nstreamlit run app.py"),
 rawcode("Listing D.2  Training evaluation and test commands","Repository root","python -m scripts.download_dataset\npython -m scripts.train_model --epochs 12 --threads 2\npython -m scripts.evaluate_model\npython -m pytest -q\npython -m unittest discover -s tests -v"),
 p("Training commands are supplied for reproducibility. The completed CNN was not retrained to prepare this report. Use its existing checkpoint for the demonstration."),
 h("D.1 Artifact provenance"),
 p("Model SHA-256: "+M["model_sha256"]),
 p("Split manifest SHA-256: "+M["split_sha256"]),
 p("Dataset archive SHA-256: "+D["archive_sha256"]),
 p("Repository: https://github.com/prxns/SmartWaste-AI"),
 h("D.2 Assignment coverage"),
 p("Problem definition: Chapter 1. Dataset collection and preprocessing: Chapter 3. Model development and training: Chapter 4. Evaluation: Chapter 7. Result analysis: Chapter 8. Conclusion and future scope: Chapter 9. Demonstration: Chapter 10. Selected code and input/output snapshots appear in the appendices and Chapters 6 and 10."),
 p("Complete both student names and ERP IDs, faculty and programme fields in the editable Word copy before submission. Apply signatures only after reviewing the declaration. The supplied assignment requests the LMS name Miniproject_FL&NN_ERPID."))

# DOCX and Markdown share one page/content plan.
doc=Document();section=doc.sections[0]
section.page_width=Inches(8.2677);section.page_height=Inches(11.6929)
section.left_margin=section.right_margin=Inches(.85)
section.top_margin=Inches(.7);section.bottom_margin=Inches(.65)
section.header_distance=section.footer_distance=Inches(.28)
section.different_first_page_header_footer=True
for name in ["Normal","Title","Heading 1","Heading 2","Caption"]:
 st=doc.styles[name];st.font.name="Liberation Serif";st.font.color.rgb=RGBColor(0,0,0)
 st.paragraph_format.space_after=Pt(6)
doc.styles["Normal"].font.size=Pt(11)
doc.styles["Normal"].paragraph_format.line_spacing=1.06
doc.styles["Title"].font.size=Pt(28);doc.styles["Title"].font.name="Liberation Sans"
for name,size in [("Heading 1",17),("Heading 2",12)]:
 st=doc.styles[name];st.font.name="Liberation Sans";st.font.size=Pt(size)
 st.font.color.rgb=RGBColor.from_string("163E33");st.paragraph_format.space_before=Pt(7)
doc.styles["Caption"].font.size=Pt(9);doc.styles["Caption"].font.italic=True
for border in list(doc.styles.element.xpath(".//w:pBdr")):border.getparent().remove(border)
header=section.header.paragraphs[0]
header.text="DIT UNIVERSITY  |  CSN441                                      SMARTWASTE AI"
for run in header.runs:run.font.name="Liberation Sans";run.font.size=Pt(8)
footer=section.footer.paragraphs[0];footer.alignment=1
footer.add_run("SmartWaste AI  |  ODD 2026-27  |  ")
fld=OxmlElement("w:fldSimple");fld.set(qn("w:instr"),"PAGE");footer._p.append(fld)
for run in footer.runs:run.font.size=Pt(8)
doc.core_properties.title="SmartWaste AI Final Project Report";doc.core_properties.author=""
doc.core_properties.subject="CSN441 CNN waste classification and Mamdani fuzzy inference"
figures=[(b[1],i+1) for i,(_,bs) in enumerate(pages) for b in bs if b[0]=="f"]
tables=[(b[1],i+1) for i,(_,bs) in enumerate(pages) for b in bs if b[0]=="t"]
def para(text,style=None):
 pp=doc.add_paragraph(style=style)
 for part in re.split(r"(https?://[^\s]+)",text.replace(chr(96),"")):
  if part.startswith(("http://","https://")):
   rel=doc.part.relate_to(part,"http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",is_external=True)
   link=OxmlElement("w:hyperlink");link.set(qn("r:id"),rel)
   rr=OxmlElement("w:r");tt=OxmlElement("w:t");tt.text=part;rr.append(tt);link.append(rr);pp._p.append(link)
  else:pp.add_run(part)
 return pp
def add_table(b):
 _,cap,heads,data,widths=b
 cp=para(cap,"Caption");cp.paragraph_format.keep_with_next=True
 tt=doc.add_table(rows=1,cols=len(heads));tt.autofit=False
 widths=widths or [6.1/len(heads)]*len(heads)
 for j,w in enumerate(widths):tt.columns[j].width=Inches(w)
 for idx,vals in enumerate([heads]+data):
  cells=tt.rows[0].cells if idx==0 else tt.add_row().cells
  for j,(cell,value) in enumerate(zip(cells,vals)):
   cell.width=Inches(widths[j]);cell.text=str(value);cell.vertical_alignment=1
   pr=cell._tc.get_or_add_tcPr();fill=OxmlElement("w:shd");fill.set(qn("w:fill"),"163E33" if idx==0 else ("EFF4F1" if idx%2 else "FFFFFF"));pr.append(fill)
   borders=OxmlElement("w:tcBorders")
   for edge in ["top","left","bottom","right"]:
    el=OxmlElement("w:"+edge);el.set(qn("w:val"),"single");el.set(qn("w:sz"),"4");el.set(qn("w:color"),"D9D9D9");borders.append(el)
   pr.append(borders)
   marg=OxmlElement("w:tcMar")
   for edge in ["top","left","bottom","right"]:
    el=OxmlElement("w:"+edge);el.set(qn("w:w"),"65");el.set(qn("w:type"),"dxa");marg.append(el)
   pr.append(marg)
   for pp in cell.paragraphs:
    pp.paragraph_format.space_after=Pt(2);pp.paragraph_format.space_before=Pt(2);pp.paragraph_format.line_spacing=1
    if j>0 and len(str(value))<18:pp.alignment=1
    for rr in pp.runs:
     rr.font.name="Liberation Sans";rr.font.size=Pt(9);rr.bold=idx==0
     rr.font.color.rgb=RGBColor(255,255,255) if idx==0 else RGBColor(0,0,0)
  keep=OxmlElement("w:cantSplit");tt.rows[idx]._tr.get_or_add_trPr().append(keep)
 repeat=OxmlElement("w:tblHeader");tt.rows[0]._tr.get_or_add_trPr().append(repeat)
 para("").paragraph_format.space_after=Pt(1)
md=[]
for index,(title,blocks) in enumerate(pages):
 if index:doc.add_page_break();md.append("\n<!-- PAGE BREAK -->\n")
 hh=doc.add_heading(title,0 if index==0 else 1);md.append("# "+title+"\n")
 if index==0:hh.alignment=1
 for b in blocks:
  if b[0]=="p":
   for pt in b[1].split("\n\n"):
    pp=para(pt);md.append(pt+"\n")
    if index==0:
     pp.alignment=1;pp.paragraph_format.space_after=Pt(15)
     if pt=="DIT UNIVERSITY":
      for rr in pp.runs:rr.bold=True;rr.font.size=Pt(20)
  elif b[0]=="h":
   pp=doc.add_heading(b[1],2);md.append("## "+b[1]+"\n")
   if index==0:pp.alignment=1
  elif b[0]=="t":
   add_table(b)
   md.append(b[1]+"\n\n| "+" | ".join(map(str,b[2]))+" |\n| "+" | ".join(["---"]*len(b[2]))+" |")
   md.extend("| "+" | ".join(map(str,row))+" |" for row in b[3]);md.append("")
  elif b[0]=="f":
   path=ASSETS/b[2];assert path.is_file(),path
   with Image.open(path) as im:ww,hh=im.size
   scale=min(6.1/ww,b[3]/hh)
   pp=doc.add_paragraph();pp.alignment=1;pp.paragraph_format.keep_with_next=True;pp.paragraph_format.space_after=Pt(3)
   pic=pp.add_run().add_picture(str(path),width=Inches(ww*scale),height=Inches(hh*scale));pic._inline.docPr.set("descr",b[1])
   pp=para(b[1],"Caption");pp.alignment=1
   md.append("!["+b[1]+"](screenshots/"+b[2]+")\n")
  elif b[0]=="c":
   pp=para(b[1]+" | "+b[2],"Caption");pp.paragraph_format.keep_with_next=True
   lines=[]
   for line in b[3].splitlines():
    lines.extend(textwrap.wrap(line,width=93,subsequent_indent="        ",break_long_words=False,break_on_hyphens=False,replace_whitespace=False,drop_whitespace=False) if len(line)>93 else [line])
   pp=doc.add_paragraph();pp.paragraph_format.line_spacing=1;pp.paragraph_format.keep_together=True
   rr=pp.add_run("\n".join(lines));rr.font.name="Liberation Mono";rr.font.size=Pt(8)
   md.append(b[1]+" | "+b[2]+"\n\n~~~python\n"+b[3]+"\n~~~\n")
  elif b[0] in ["toc","figlist","tablist"]:
   items=figures if b[0]=="figlist" else tables if b[0]=="tablist" else [(tt,i+1) for i,(tt,_) in enumerate(pages) if i in [1,2,4,5] or tt.startswith(("Chapter","Appendix")) or tt=="References"]
   for label,num in items:
    pp=para(label+"\t"+str(num));pp.paragraph_format.tab_stops.add_tab_stop(Inches(6),WD_TAB_ALIGNMENT.RIGHT)
    pp.paragraph_format.space_after=Pt(4 if b[0]!="toc" else 9)
    if b[0]!="toc":
     for rr in pp.runs:rr.font.size=Pt(9)
    md.append(label+" - "+str(num)+"\n")
doc.save(str(OUT)+".docx");OUT.with_suffix(".md").write_text("\n".join(md),encoding="utf-8")
(WORK/"plan.json").write_text(json.dumps({"pages":[tt for tt,_ in pages],"figures":figures,"tables":tables},indent=2))
print("REPORT_BUILT",len(pages),"planned pages",len(figures),"figures",len(tables),"tables")
