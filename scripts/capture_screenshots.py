"""Capture actual application pages for the report (optional Playwright dependency)."""
import argparse
import base64
import os
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
from config import ROOT


def inspect_layout(page, label):
    """Check real browser geometry; never mask overflow to make this pass."""
    page.get_by_test_id("stMain").evaluate("element => { element.scrollTop = 0; }")
    geometry=page.evaluate("""() => {
        const main=document.querySelector('[data-testid="stMain"]');
        const toolbar=document.querySelector('[data-testid="stHeader"]');
        const eyebrow=document.querySelector('.eyebrow');
        const brand=document.querySelector('.sidebar-brand');
        const heading=document.querySelector('.page-intro h1');
        const nav=[...document.querySelectorAll('[data-testid="stSidebar"] label[data-baseweb="radio"]')];
        return {
            viewport: innerWidth,
            documentWidth: document.documentElement.scrollWidth,
            mainWidth: main.clientWidth,
            mainScrollWidth: main.scrollWidth,
            eyebrowTop: eyebrow.getBoundingClientRect().top,
            toolbarBottom: toolbar.getBoundingClientRect().bottom,
            headingTop: heading.getBoundingClientRect().top,
            eyebrowBottom: eyebrow.getBoundingClientRect().bottom,
            brandClipped: brand.scrollWidth > brand.clientWidth + 1,
            navCount: nav.length,
            navClipped: nav.some(el => el.scrollHeight > el.clientHeight + 1 || el.scrollWidth > el.clientWidth + 1),
            zoom: devicePixelRatio,
        };
    }""")
    assert geometry["documentWidth"] <= geometry["viewport"]+1, (label,geometry)
    assert geometry["mainScrollWidth"] <= geometry["mainWidth"]+1, (label,geometry)
    assert geometry["eyebrowTop"] >= geometry["toolbarBottom"]+4, (label,geometry)
    assert geometry["headingTop"] >= geometry["eyebrowBottom"], (label,geometry)
    assert not geometry["brandClipped"] and not geometry["navClipped"], (label,geometry)
    assert geometry["navCount"] == 5 and geometry["zoom"] == 1, (label,geometry)
    print("Layout verified:",label,geometry)


def responsive_pages(page, target, wait_until_idle, classified=False):
    pages=[("Waste Classification","classification_result")] if classified else [
        ("Dashboard","dashboard"),("Waste Classification","classification_empty"),
        ("Fuzzy Logic Laboratory","fuzzy_laboratory"),("Model Evaluation","model_evaluation"),
        ("About & Methodology","methodology")]
    review=target/"responsive"; review.mkdir(exist_ok=True)
    for width,height in ((1920,1080),(1366,768),(1280,720)):
        page.set_viewport_size({"width":width,"height":height})
        for label,filename in pages:
            page.get_by_text(label,exact=True).first.click()
            wait_until_idle(page)
            if page.get_by_test_id("stException").count():
                raise RuntimeError(f"Application exception on {label} at {width}x{height}")
            inspect_layout(page,f"{filename} {width}x{height}")
            page.screenshot(path=str(review/f"{filename}_{width}x{height}.png"),full_page=True)
    page.set_viewport_size({"width":1440,"height":1050})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-launcher",action="store_true",
                        help="Exercise the installed Windows run.bat launcher.")
    parser.add_argument("--responsive",action="store_true",
                        help="Check all five sections at the three demonstration resolutions.")
    parser.add_argument("--review-previews",action="store_true",
                        help="Include actual screenshots in CI logs for offline visual review.")
    args=parser.parse_args()
    if args.windows_launcher and os.name!="nt":
        parser.error("--windows-launcher requires Windows")
    from playwright.sync_api import sync_playwright, expect
    def wait_until_idle(page):
        # Use Streamlit's own end-to-end test contract. Headings may appear
        # before a rerun completes and old page elements are removed.
        page.wait_for_timeout(250)
        page.locator("[data-testid='stApp'][data-test-script-state='notRunning']").wait_for(timeout=60000,state="attached")
        expect(page.get_by_test_id("stSkeleton")).to_have_count(0,timeout=60000)
        page.wait_for_timeout(500)
    target=ROOT/"assets/screenshots"; target.mkdir(parents=True,exist_ok=True)
    options=["--server.headless","true","--server.port","8510"]
    command=([os.environ["COMSPEC"],"/d","/c","run.bat",*options] if args.windows_launcher
             else [sys.executable,"-m","streamlit","run","app.py",*options])
    process=subprocess.Popen(command,cwd=ROOT,
                             stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    try:
        for _ in range(45):
            try:
                with urlopen("http://localhost:8510/_stcore/health",timeout=1) as response:
                    if response.status==200: break
            except OSError: time.sleep(1)
        else: raise RuntimeError("Streamlit server did not start.")
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch()
            page=browser.new_page(viewport={"width":1440,"height":1050},device_scale_factor=1)
            page.goto("http://localhost:8510",wait_until="networkidle")
            page.get_by_role("heading",name="Waste insights, clearly explained.").wait_for()
            wait_until_idle(page)
            inspect_layout(page,"dashboard overview")
            page.screenshot(path=str(target/"dashboard.png"),full_page=False)
            if args.responsive:
                responsive_pages(page,target,wait_until_idle)
            # Exercise a real held-out image through the browser, not a mock prediction.
            import pandas as pd
            test=pd.read_csv(ROOT/"results/splits.csv")
            sample=test[test.split=="test"].iloc[0]
            page.get_by_text("Waste Classification",exact=True).first.click()
            wait_until_idle(page)
            inspect_layout(page,"classification empty overview")
            page.screenshot(path=str(target/"classification_empty.png"),full_page=False)
            page.locator('input[type="file"]').set_input_files(str(ROOT/"data/dataset-resized"/sample.path))
            page.get_by_role("button",name="Classify image",exact=True).click()
            page.get_by_role("heading",name="2 · Fuzzy recycling decision support",exact=True).wait_for(timeout=60000)
            wait_until_idle(page)
            inspect_layout(page,"classification result overview")
            page.screenshot(path=str(target/"classification.png"),full_page=False)
            page.locator(".st-key-decision_panel").screenshot(path=str(target/"decision_support.png"))
            if args.responsive:
                responsive_pages(page,target,wait_until_idle,classified=True)
            slider=page.get_by_role("slider")
            # BaseWeb's slider implements arrow keys; End is not portable.
            # Each change reruns Streamlit. A burst of keypresses can race the
            # widget remount, so acknowledge the rendered value before the next.
            for value in range(int(slider.get_attribute("aria-valuenow"))+1,101):
                slider.press("ArrowRight")
                expect(page.get_by_test_id("stMetric").filter(has_text="User contamination input")
                       .get_by_test_id("stMetricValue")).to_have_text(f"{value} / 100",timeout=10000)
                wait_until_idle(page)
            expect(slider).to_have_attribute("aria-valuenow","100")
            print("Classifier contamination after keyboard interaction:",slider.get_attribute("aria-valuenow"))
            page.get_by_text("Low suitability",exact=True).wait_for()
            wait_until_idle(page)
            page.locator(".st-key-decision_panel").screenshot(path=str(target/"decision_support_contaminated.png"))
            page.get_by_text("Dashboard",exact=True).first.click()
            page.get_by_role("heading",name="Waste insights, clearly explained.").wait_for()
            wait_until_idle(page)
            inspect_layout(page,"populated dashboard overview")
            page.screenshot(path=str(target/"dashboard.png"),full_page=False)
            with tempfile.TemporaryDirectory() as temporary:
                with page.expect_download() as event:
                    page.get_by_role("button",name="Export session CSV",exact=True).click()
                export=__import__("pathlib").Path(temporary)/"session.csv"
                event.value.save_as(str(export))
                activity=pd.read_csv(export)
                assert len(activity)==1 and activity.iloc[0].contamination==100
                assert activity.iloc[0].category in ("cardboard","glass","metal","paper","plastic","trash")
                print("Real session CSV export verified:",len(activity),"classification")
            wait_until_idle(page)
            for label,filename in [("Fuzzy Logic Laboratory","fuzzy_laboratory.png"),("Model Evaluation","model_evaluation.png"),("About & Methodology","methodology.png")]:
                page.get_by_text(label,exact=True).first.click()
                page.get_by_role("heading",name={"fuzzy_laboratory.png":"Fuzzy Logic Laboratory","model_evaluation.png":"Model evaluation","methodology.png":"About & methodology"}[filename],exact=True).wait_for()
                wait_until_idle(page)
                if page.get_by_test_id("stException").count(): raise RuntimeError("Application rendered an exception.")
                inspect_layout(page,filename)
                page.screenshot(path=str(target/filename),full_page=False)
            browser.close()
            if args.review_previews:
                for path in sorted(target.glob("*.png")):
                    encoded=base64.b64encode(path.read_bytes()).decode("ascii")
                    for index,offset in enumerate(range(0,len(encoded),60000)):
                        print(f"UI_PREVIEW {path.name} {index} {encoded[offset:offset+60000]}")
    finally:
        if os.name=="nt":
            # A .bat launcher has a child Python process; stop only this test's tree.
            subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW,check=False)
        else:
            process.terminate()
        process.wait(timeout=15)


if __name__=="__main__": main()
