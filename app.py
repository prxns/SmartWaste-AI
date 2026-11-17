"""SmartWaste AI local Streamlit application. Run: streamlit run app.py."""
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from config import ROOT, CLASS_NAMES, MATERIAL_POTENTIAL, MODEL_PATH, RESULTS_DIR, RULE_TABLE
from smartwaste.fuzzy_engine import infer
from smartwaste.preprocessing import read_image
from smartwaste.recommendations import guidance
from smartwaste.charts import membership_chart, aggregate_chart, style

st.set_page_config(page_title="SmartWaste AI", page_icon="assets/icon.svg", layout="wide")
st.markdown(f"<style>{(ROOT/'assets/style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

PAGES = ("Dashboard", "Waste Classification", "Fuzzy Logic Laboratory", "Model Evaluation", "About & Methodology")
TRAIN_COMMAND = "python -m scripts.download_dataset\npython -m scripts.train_model --epochs 12"


@st.cache_resource
def load_classifier(stamp):
    from smartwaste.model import WasteClassifier
    return WasteClassifier()


def navigate_classifier():
    st.session_state.page = "Waste Classification"


def introduction(title, description, label):
    st.markdown(
        f'<header class="page-intro"><p class="eyebrow">{escape(label)}</p>'
        f'<h1>{escape(title)}</h1><p>{escape(description)}</p></header>',
        unsafe_allow_html=True)


def fuzzy_details(result):
    c1, c2 = st.columns(2)
    c1.plotly_chart(membership_chart(result, "Material potential", result.potential), width="stretch")
    c2.plotly_chart(membership_chart(result, "Contamination", result.contamination), width="stretch")
    st.plotly_chart(aggregate_chart(result), width="stretch")
    a, b = st.columns([1, 2])
    with a:
        st.subheader("Membership degrees")
        st.dataframe(pd.DataFrame(result.memberships).round(3), width="stretch")
    with b:
        st.subheader("Rule firing strengths")
        st.dataframe(pd.DataFrame(result.rules), hide_index=True, width="stretch",
                     column_config={"Firing strength": st.column_config.ProgressColumn(min_value=0,max_value=1,format="%.3f")})
    st.caption("AND = minimum · implication = clipping · aggregation = maximum · crisp score = centroid of the aggregate.")


def dashboard():
    introduction("Waste insights, clearly explained.", "Intelligent Waste Classification & Recycling Decision Support.", "SmartWaste AI / Overview")
    st.markdown('<div class="hero"><span class="pill">CNN + Mamdani fuzzy logic</span><h2>From an image to an informed decision.</h2><p>Identify one of six waste materials, assess contamination yourself, and inspect every step of the recycling suitability calculation.</p></div>', unsafe_allow_html=True)
    history = pd.DataFrame(st.session_state.history)
    columns = st.columns(3)
    columns[0].metric("Session classifications", len(history))
    columns[1].metric("Mean suitability", f"{history.suitability.mean():.1f} / 100" if len(history) else "—")
    columns[2].metric("Material categories seen", history.category.nunique() if len(history) else 0)
    st.button("Open waste classifier", type="primary", on_click=navigate_classifier)
    if history.empty:
        st.markdown('<div class="empty-state"><h3>Your session starts here</h3><p>Classify a photograph to see your material distribution, suitability summary and recent activity. Only your actual results appear here.</p></div>', unsafe_allow_html=True)
    else:
        a,b = st.columns(2)
        counts = history.category.value_counts().rename_axis("Category").reset_index(name="Count")
        a.plotly_chart(style(px.bar(counts, x="Category", y="Count", color_discrete_sequence=["#32704F"]), "Category distribution"), width="stretch")
        b.plotly_chart(style(px.histogram(history, x="suitability", nbins=10, range_x=[0,100], color_discrete_sequence=["#829E68"]), "Suitability in this session"), width="stretch")
        st.subheader("Recent classifications")
        st.dataframe(history.iloc[::-1], hide_index=True, width="stretch")
        st.download_button("Export session CSV", history.to_csv(index=False).encode("utf-8"), "smartwaste_session.csv", "text/csv")
        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.pop("prediction", None)
            st.rerun()
    st.caption("Session activity is held in memory. Uploaded photographs are not written to disk.")


def classify():
    introduction("Classify a waste image", "Identify the material, assess contamination and explore an explained recycling decision.", "SmartWaste AI / Classifier")
    with st.container(border=True, key="upload_panel"):
        st.subheader("Your waste photograph")
        upload = st.file_uploader("Upload a JPEG or PNG", type=["jpg", "jpeg", "png"], help="Maximum 10 MB and 24 megapixels. Contamination is entered manually.")
        st.caption("For a useful prediction, photograph one item in even light against a clear background. Keep the whole item in view. Maximum 10 MB / 24 megapixels.")
    if upload is None:
        st.session_state.pop("prediction", None)
        st.markdown(
            '<div class="workflow-intro"><h3>From photograph to recycling guidance</h3>'
            '<p>Your image preview and analysis will appear here after upload.</p></div>'
            '<ol class="workflow" aria-label="Classification workflow">'
            '<li><span class="step-number">01</span><h3>Upload one item</h3><p>Choose a clear photograph of the waste material you want to identify.</p></li>'
            '<li><span class="step-number">02</span><h3>Classify the material</h3><p>Run the CNN and review its three highest material prediction scores.</p></li>'
            '<li><span class="step-number">03</span><h3>Assess suitability</h3><p>Set contamination yourself. Mamdani fuzzy logic explains the resulting decision score.</p></li></ol>'
            '<p class="supported-materials">Supported materials: cardboard, glass, metal, paper, plastic and trash. Photographs stay in memory during your session.</p>',
            unsafe_allow_html=True)
        if not MODEL_PATH.is_file():
            st.info("The trained CNN is not available yet. The fuzzy laboratory works independently.")
            st.code(TRAIN_COMMAND, language="bash")
        return
    raw = upload.getvalue()
    digest = sha256(raw).hexdigest()
    if st.session_state.get("prediction", {}).get("image_hash") != digest:
        st.session_state.pop("prediction", None)
    try:
        image = read_image(raw)
    except ValueError as exc:
        st.error(str(exc)); return
    left, right = st.columns([1, 1.15], gap="large")
    with left, st.container(border=True, key="image_preview"):
        st.subheader("Uploaded photograph")
        st.image(image, caption=f"Uploaded image · {image.width} x {image.height} pixels", width="stretch")
        st.caption("The model was trained on individual items on simple backgrounds. Mixed waste or unrelated images can still produce a score.")
    with right, st.container(border=True, key="cnn_result"):
        st.subheader("1 · CNN material classification")
        if not MODEL_PATH.is_file():
            st.warning("No trained model found. Train the CNN before classifying images.")
            st.code(TRAIN_COMMAND, language="bash")
            return
        if st.button("Classify image", type="primary", width="stretch"):
            try:
                with st.spinner("Analyzing material with MobileNetV2."):
                    prediction = load_classifier(MODEL_PATH.stat().st_mtime_ns).predict(image)
                prediction.update(image_hash=digest, activity_id=len(st.session_state.history)+1)
                st.session_state.prediction = prediction
                st.session_state.history.append({"id": prediction["activity_id"],
                    "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "category": prediction["category"], "model_score": prediction["top_three"][0]["score"],
                    "potential": MATERIAL_POTENTIAL[prediction["category"]], "contamination": 20,
                    "suitability": infer(MATERIAL_POTENTIAL[prediction["category"]], 20).score})
            except (FileNotFoundError, ValueError, ImportError, OSError, RuntimeError) as exc:
                st.error(f"Classification could not run: {exc}")
        prediction = st.session_state.get("prediction")
        if prediction is None:
            st.markdown('<div class="empty-state"><h3>Ready for analysis</h3><p>Select Classify image to identify the material. Your prediction scores and recycling assessment will follow.</p></div>', unsafe_allow_html=True)
            return
        category = prediction["category"]
        st.markdown(f'<div class="result">Predicted material<strong>{category.title()}</strong></div>', unsafe_allow_html=True)
        scores = pd.DataFrame(prediction["top_three"])
        st.dataframe(scores, hide_index=True, width="stretch", column_config={"score": st.column_config.ProgressColumn("Prediction score", min_value=0,max_value=1,format="%.3f")})
        st.caption(f"Uncalibrated softmax scores, not guaranteed correctness. CNN input: {' x '.join(map(str,prediction['input_dimensions']))}.")
    with st.container(border=True, key="decision_panel"):
        st.subheader("2 · Fuzzy recycling decision support")
        st.caption("The CNN identifies material. You provide contamination; the fuzzy engine calculates suitability.")
        contamination = st.slider("Contamination level - your manual assessment", 0, 100, 20, key=f"contamination_{prediction['activity_id']}", help="0 = visibly clean; 100 = heavily contaminated. The CNN does not detect contamination.")
        result = infer(MATERIAL_POTENTIAL[category], contamination)
        a,b,c = st.columns(3)
        a.metric("Material potential assumption", f"{result.potential:.0f} / 100")
        b.metric("User contamination input", f"{contamination} / 100")
        c.metric(f"{result.status} suitability", f"{result.score:.2f} / 100")
        st.info(guidance(category, result.status))
        st.caption("Material potential values are illustrative academic assumptions. Suitability is a fuzzy decision score, not an empirical recycling rate or local acceptance guarantee.")
        for row in st.session_state.history:
            if row["id"] == prediction["activity_id"]:
                row.update(contamination=contamination, suitability=result.score)
        with st.expander("Inspect the fuzzy inference calculation"):
            fuzzy_details(result)


def laboratory():
    introduction("Fuzzy Logic Laboratory", "Adjust both inputs independently and inspect a complete Mamdani inference trace.", "SmartWaste AI / Explainability")
    with st.container(border=True, key="input_panel"):
        st.subheader("Explore the two inputs")
        a,b = st.columns(2)
        potential = a.slider("Material recyclability potential", 0, 100, 80)
        contamination = b.slider("Contamination level", 0, 100, 20)
        st.caption("Both inputs are independent here. Every change recalculates the rules and centroid immediately.")
    result = infer(potential, contamination)
    c,d,e = st.columns(3)
    c.metric("Centroid suitability", f"{result.score:.2f} / 100")
    d.metric("Linguistic status", result.status)
    e.metric("Activated rules", sum(r["Firing strength"] > 0 for r in result.rules))
    fuzzy_details(result)
    st.subheader("Complete nine-rule base")
    st.dataframe(pd.DataFrame(RULE_TABLE, index=["Low potential", "Medium potential", "High potential"], columns=["Low contamination", "Medium contamination", "High contamination"]), width="stretch")
    st.write("Each input belongs to Low, Medium and High sets to a degree between 0 and 1. Each rule takes the smaller input degree. Its output set is clipped at that strength. The largest clipped degree is kept at each output point; the centre of area gives the crisp result.")
    st.latex(r"s^*=\frac{\int_0^{100} s\,\mu_{agg}(s)\,ds}{\int_0^{100}\mu_{agg}(s)\,ds}")
    st.caption("Status uses the largest output membership at the centroid. Scores are not probabilities.")
    st.warning("Mathematical limitation: this exact min/max rule base with asymmetric shoulder sets is not globally monotonic. For example, potential 15 gives 19.44 at contamination 25 and 19.63 at 26. This is a centroid/clipping effect; the rule definitions have not been silently changed.")


def evaluation():
    introduction("Model evaluation", "Held-out test performance, recorded by the training pipeline.", "SmartWaste AI / Evidence")
    path = RESULTS_DIR/"metrics.json"
    if not path.is_file():
        st.info("No held-out evaluation has been produced. Train the model to generate real metrics and charts.")
        st.code(TRAIN_COMMAND, language="bash")
        return
    try:
        metrics = json.loads(path.read_text())
        from smartwaste.evaluation import file_sha256
        if not MODEL_PATH.is_file() or file_sha256(MODEL_PATH) != metrics["model_sha256"]:
            st.warning("Evaluation artifacts do not match the current model. Run python -m scripts.evaluate_model."); return
        columns = st.columns(4)
        for col, key, title in zip(columns, ["test_accuracy","macro_precision","macro_recall","macro_f1"], ["Test accuracy","Macro precision","Macro recall","Macro F1"]):
            col.metric(title, f"{metrics[key]:.1%}")
        st.caption(f"Untouched test split · {metrics['test_samples']} images · best checkpoint selected only on validation loss.")
        st.subheader("Per-class performance")
        report = pd.DataFrame(metrics["classification_report"]).T
        st.dataframe(report.loc[list(CLASS_NAMES)], width="stretch")
        matrix = px.imshow(metrics["confusion_matrix"], x=list(CLASS_NAMES), y=list(CLASS_NAMES), text_auto=True, color_continuous_scale="Greens", labels=dict(x="Predicted",y="Actual",color="Images"))
        st.plotly_chart(style(matrix,"Test confusion matrix",430), width="stretch")
        st.subheader("Training history")
        history = pd.DataFrame(json.loads((RESULTS_DIR/"history.json").read_text()))
        history.index = history.index+1
        a,b = st.columns(2)
        a.plotly_chart(style(px.line(history, y=["accuracy","val_accuracy"], labels={"index":"Epoch","value":"Accuracy"}),"Training and validation accuracy"), width="stretch")
        b.plotly_chart(style(px.line(history, y=["loss","val_loss"], labels={"index":"Epoch","value":"Loss"}),"Training and validation loss"), width="stretch")
        st.subheader("Dataset distribution")
        distribution = pd.read_csv(RESULTS_DIR/"class_distribution.csv")
        st.plotly_chart(style(px.bar(distribution,x="category",y="count",color_discrete_sequence=["#32704F"]),"Validated dataset distribution"), width="stretch")
        with st.expander("Training configuration and provenance"):
            st.json(json.loads((ROOT/"models/training_config.json").read_text()))
            st.write(f"Model SHA-256: `{metrics['model_sha256']}`")
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Evaluation artifacts are incomplete or invalid: {exc}")


def methodology():
    introduction("About & methodology", "CNN-Based Waste Classification with Mamdani Fuzzy Logic Decision Support for Intelligent Waste Segregation.", "SmartWaste AI / Methodology")
    st.subheader("Purpose and scope")
    st.write("This academic prototype identifies material categories from images and provides an inspectable recycling suitability assessment. A CNN supplies the category; the user supplies contamination. The fuzzy engine combines an assumed material potential and the manual contamination value.")
    st.subheader("Methodology")
    architecture, explanation = st.columns([1.1, 1], gap="large")
    with architecture:
        st.graphviz_chart('digraph { rankdir=TB; node [shape=box,style="rounded,filled",fillcolor="#EAF0E7",color="#739179",fontname="Arial"]; image [label="JPEG / PNG image"]; cnn [label="MobileNetV2 + six-class head"]; material [label="Category → potential assumption"]; user [label="User contamination input"]; fuzzy [label="Mamdani min / max inference"]; score [label="Centroid → suitability and guidance"]; image -> cnn -> material -> fuzzy -> score; user -> fuzzy; }')
    with explanation, st.container(border=True, key="methodology_steps"):
        st.subheader("Two inputs. One explained assessment.")
        st.markdown("**Identify the material**")
        st.write("MobileNetV2 assigns one of the six TrashNet categories. That category selects the configured, illustrative material potential.")
        st.markdown("**Add your contamination assessment**")
        st.write("The contamination slider is a manual input. It is independent of the CNN prediction and can be adjusted without retraining.")
        st.markdown("**Inspect the fuzzy decision**")
        st.write("The nine Mamdani rules combine both inputs. Minimum clipping, maximum aggregation and the centroid produce the suitability score and guidance.")
        st.caption("A transparent academic decision score. Confirm local recycling and disposal requirements.")
    st.subheader("Dataset and CNN")
    st.write("TrashNet by Gary Thung and Mindy Yang contains cardboard, glass, metal, paper, plastic and trash. The pipeline validates images, removes exact decoded duplicates, and stratifies a seeded 70/15/15 split. ImageNet MobileNetV2 is frozen; a global average pooling, dropout and six-class softmax head is trained with Adam and class weights. Only training images are augmented. MobileNetV2 preprocessing maps RGB pixels to [-1,1].")
    st.link_button("TrashNet source and attribution", "https://github.com/garythung/trashnet")
    st.subheader("Fuzzy assumptions")
    st.dataframe(pd.DataFrame({"Category": CLASS_NAMES, "Illustrative material potential": [MATERIAL_POTENTIAL[x] for x in CLASS_NAMES]}), hide_index=True, width="stretch")
    st.write("Low uses trapezoid [0,0,25,50]; Medium uses triangle [25,50,75]; High uses trapezoid [50,75,100,100]. Minimum evaluates AND and clips output sets; maximum aggregates rules. Numerical centroid defuzzification integrates the piecewise linear output on 1,001 points. These shapes and all nine rules live separately from the UI.")
    st.subheader("Limitations")
    st.write("A small imbalanced dataset of simple backgrounds cannot represent all waste streams. Predictions are uncalibrated and lack an out-of-distribution rejection mechanism. Contamination is subjective. Material values and rules are demonstration assumptions. The prescribed centroid system has local nonmonotonic regions; it has not been validated against recycling outcomes. Local rules determine disposal eligibility.")
    st.caption("Python · TensorFlow / Keras · Streamlit · Pillow · NumPy · Pandas · Scikit-learn · Plotly · Matplotlib · Pytest")


st.session_state.setdefault("history", [])
with st.sidebar:
    brand_icon = (ROOT/"assets/icon.svg").read_text(encoding="utf-8")
    st.markdown(f'<div class="sidebar-brand">{brand_icon}<span class="brand-name">SmartWaste AI</span></div><p class="brand-tagline">Intelligent waste classification<br>and recycling decision support.</p><p class="nav-label">WORKSPACE</p>', unsafe_allow_html=True)
    st.radio("Workspace", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.markdown('<p class="sidebar-note">Academic prototype<br>Six material classes. Explainable fuzzy decisions.</p>', unsafe_allow_html=True)
    st.markdown(f'<span class="model-status">{"CNN ready" if MODEL_PATH.is_file() else "CNN training required"}</span>', unsafe_allow_html=True)
dict(zip(PAGES, (dashboard, classify, laboratory, evaluation, methodology)))[st.session_state.page]()
st.markdown('<div class="footnote">SmartWaste AI · Academic decision support · Confirm local disposal and recycling rules.</div>', unsafe_allow_html=True)
