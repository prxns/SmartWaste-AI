import importlib.util
import unittest

HAS_UI=all(importlib.util.find_spec(x) is not None for x in ("streamlit","plotly"))


@unittest.skipUnless(HAS_UI,"Streamlit/Plotly dependencies are unavailable")
class AppTests(unittest.TestCase):
    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file("app.py",default_timeout=45).run()

    def test_all_pages_render(self):
        app=self.app()
        self.assertFalse(app.exception)
        for page in ("Waste Classification","Fuzzy Logic Laboratory","Model Evaluation","About & Methodology","Dashboard"):
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception, str(app.exception))

    def test_laboratory_reacts_to_input(self):
        app=self.app()
        app.sidebar.radio[0].set_value("Fuzzy Logic Laboratory").run()
        before=app.metric[0].value
        app.slider[1].set_value(95).run()
        self.assertFalse(app.exception)
        self.assertNotEqual(app.metric[0].value,before)

    def test_dashboard_uses_actual_session_and_clears_it(self):
        app=self.app()
        self.assertEqual(app.metric[0].value,"0")
        app.session_state["history"]=[{"id":1,"time_utc":"2026-01-01T00:00:00+00:00", "category":"metal","model_score":.8,"potential":90,"contamination":20,"suitability":80}]
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value,"1")
        next(x for x in app.button if x.label=="Clear session history").click().run()
        self.assertEqual(app.metric[0].value,"0")

    def test_navigation_action_opens_classifier(self):
        app=self.app()
        next(x for x in app.button if x.label=="Open waste classifier").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.sidebar.radio[0].value,"Waste Classification")
