import unittest
import numpy as np
from smartwaste.fuzzy_engine import triangle, trapezoid, fuzzify, centroid, infer


class MembershipTests(unittest.TestCase):
    def test_triangle_values_and_boundaries(self):
        np.testing.assert_allclose(triangle([0,25,37.5,50,62.5,75,100],25,50,75), [0,0,.5,1,.5,0,0])

    def test_shoulder_endpoints(self):
        np.testing.assert_allclose(trapezoid([-1,0,25,37.5,50,100],0,0,25,50),[0,1,1,.5,0,0])
        np.testing.assert_allclose(trapezoid([0,50,62.5,75,100,101],50,75,100,100),[0,0,.5,1,1,0])

    def test_invalid_definitions(self):
        for fn, params in [(triangle,(1,1,2)),(trapezoid,(4,3,2,1))]:
            with self.assertRaises(ValueError): fn(2,*params)

    def test_fuzzification(self):
        self.assertEqual(fuzzify(37.5), {"Low":.5,"Medium":.5,"High":0})

    def test_invalid_inputs(self):
        for bad in (-1,101,float("nan"),float("inf")):
            with self.assertRaises(ValueError): infer(bad,30)
            with self.assertRaises(ValueError): infer(30,bad)


class InferenceTests(unittest.TestCase):
    def test_all_nine_rules_present(self):
        r=infer(80,20)
        self.assertEqual(len(r.rules),9)
        active=[x for x in r.rules if x["Firing strength"]>0]
        self.assertEqual(len(active),1)
        self.assertEqual(active[0]["Rule"],"R7")
        self.assertEqual(active[0]["Suitability"],"High")
        self.assertEqual(active[0]["Firing strength"],1)

    def test_min_implication_max_aggregation(self):
        r=infer(37.5,62.5)
        expected=np.maximum(np.minimum(.5,r.outputs["Low"]),np.minimum(.5,r.outputs["Medium"]))
        np.testing.assert_allclose(r.aggregate,expected)
        self.assertEqual(sum(x["Firing strength"]>0 for x in r.rules),4)

    def test_centroid_against_analytic_shoulders(self):
        # Full Low trapezoid area = 37.5, first moment = 729.1666...
        self.assertAlmostEqual(infer(0,0).score, 175/9, places=4)
        self.assertAlmostEqual(infer(100,0).score, 725/9, places=4)
        self.assertAlmostEqual(infer(50,0).score,50,places=8)

    def test_centroid_nonuniform_triangle(self):
        self.assertAlmostEqual(centroid([0,2,8],[0,1,0]),10/3,places=10)

    def test_zero_area_is_reported(self):
        with self.assertRaises(ValueError): centroid([0,50,100],[0,0,0])

    def test_bad_centroid_arrays(self):
        for x,y in [([0,0],[1,1]),([0,1],[1,-1]),([0,1],[1,float("nan")])]:
            with self.assertRaises(ValueError): centroid(x,y)

    def test_output_range_and_coverage(self):
        for p in np.linspace(0,100,21):
            for c in np.linspace(0,100,21):
                score=infer(p,c).score
                self.assertTrue(np.isfinite(score) and 0<=score<=100)

    def test_monotonicity_at_linguistic_peaks(self):
        for p in (0,50,100):
            scores=[infer(p,c).score for c in (0,50,100)]
            self.assertTrue(np.all(np.diff(scores)<=1e-8))
        for c in (0,50,100):
            scores=[infer(p,c).score for p in (0,50,100)]
            self.assertTrue(np.all(np.diff(scores)>=-1e-8))

    @unittest.expectedFailure
    def test_global_contamination_monotonicity_specification_conflict(self):
        # Required shapes/rules violate this desired property. Keep this as an
        # explicit expected failure rather than modifying the specified FIS.
        self.assertLessEqual(infer(15,26).score,infer(15,25).score+1e-8)

    @unittest.expectedFailure
    def test_global_potential_monotonicity_specification_conflict(self):
        self.assertGreaterEqual(infer(39,100).score,infer(38,100).score-1e-8)
