"""Capture actual application pages for the report (optional Playwright dependency)."""
import os
import subprocess
import sys
import time
from urllib.request import urlopen
from config import ROOT


def main():
    from playwright.sync_api import sync_playwright
    target=ROOT/"assets/screenshots"; target.mkdir(parents=True,exist_ok=True)
    process=subprocess.Popen([sys.executable,"-m","streamlit","run","app.py","--server.headless","true","--server.port","8510"],cwd=ROOT,
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
            page.screenshot(path=str(target/"dashboard.png"),full_page=True)
            # Exercise a real held-out image through the browser, not a mock prediction.
            import pandas as pd
            test=pd.read_csv(ROOT/"results/splits.csv")
            sample=test[test.split=="test"].iloc[0]
            page.get_by_text("Waste Classification",exact=True).first.click()
            page.locator('input[type="file"]').set_input_files(str(ROOT/"data/dataset-resized"/sample.path))
            page.get_by_role("button",name="Classify image",exact=True).click()
            page.get_by_role("heading",name="2 · Fuzzy recycling decision support",exact=True).wait_for(timeout=60000)
            page.screenshot(path=str(target/"classification.png"),full_page=True)
            page.get_by_role("slider").press("End")
            page.get_by_text("Low suitability",exact=True).wait_for()
            page.get_by_text("Dashboard",exact=True).first.click()
            page.get_by_role("heading",name="Waste insights, clearly explained.").wait_for()
            page.screenshot(path=str(target/"dashboard.png"),full_page=True)
            for label,filename in [("Fuzzy Logic Laboratory","fuzzy_laboratory.png"),("Model Evaluation","model_evaluation.png")]:
                page.get_by_text(label,exact=True).first.click()
                page.get_by_role("heading",name="Fuzzy Logic Laboratory" if filename.startswith("fuzzy") else "Model evaluation",exact=True).wait_for()
                page.wait_for_timeout(1500)
                if page.get_by_test_id("stException").count(): raise RuntimeError("Application rendered an exception.")
                page.screenshot(path=str(target/filename),full_page=True)
            browser.close()
    finally:
        process.terminate()
        process.wait(timeout=15)


if __name__=="__main__": main()
