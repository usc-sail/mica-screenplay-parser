# author : Sabyasachee

# standard library
import unittest

# third party
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

class TestParsing(unittest.TestCase):

    def test_rule_parser_on_screenplays(self):
        import os
        from screenplayparser import ScreenplayParser
        
        parser = ScreenplayParser(use_rules=True)
        tagset = set("SNCDETMO")
        screenplays_folder = "tests/screenplays"

        scripts = []

        for filename in os.listdir(screenplays_folder):
            if filename.endswith(".txt"):
                filepath = os.path.join(screenplays_folder, filename)
                with open(filepath) as f:
                    script = f.read().splitlines()
                scripts.append(script)

        for script in scripts:
            n = len(script)
            tags = parser.parse(script)
            self.assertEqual(n, len(tags), "number of tags should equal the number of script lines")
            for tag in tags:
                self.assertIn(tag, tagset, "tag should belong to tagset")
    
    def test_rule_parser_on_noisy_screenplays(self):
        import pandas as pd
        from screenplayparser import ScreenplayParser
        
        parser = ScreenplayParser(use_rules=True)
        tagset = set("SNCDETMO")
        eval_tagset = list("SNCDET")
        df = pd.read_csv("tests/noisy.csv", index_col=None)

        scripts = []
        pred = []
        true = []

        for _, script_df in df.groupby(["movie", "error"]):
            script = script_df["text"].fillna("").tolist()
            scripts.append(script)
            true.extend(script_df["label"].tolist())
        
        for script in scripts:
            n = len(script)
            tags = parser.parse(script)
            self.assertEqual(n, len(tags), "number of tags should equal the number of script lines")
            for tag in tags:
                self.assertIn(tag, tagset, "tag should belong to tagset")
            pred.extend(tags)
        
        print()
        p, r, f1, _ = precision_recall_fscore_support(true, pred, labels=list(eval_tagset), average="micro")
        print(f"precision {p:.3f}, recall {r:.3f}, f1 {f1:.3f}")

    def test_robust_parser_on_noisy_screenplays(self):
        import pandas as pd
        from screenplayparser import ScreenplayParser
        
        parser = ScreenplayParser(device_id=0)
        tagset = set("SNCDETMO")
        eval_tagset = list("SNCDET")
        df = pd.read_csv("tests/noisy.csv", index_col=None)

        allscripts = []
        alltrue = []

        for _, script_df in df.groupby(["movie", "error"]):
            script = script_df["text"].fillna("").tolist()
            allscripts.append(script)
            alltrue.append(script_df["label"].tolist())
        
        index = np.random.permutation(len(allscripts))
        index = index[: min(len(index), 10)]
        scripts = [allscripts[i] for i in index]
        true = [label for i in index for label in alltrue[i]]
        pred = []
        print()

        for script in tqdm(scripts):
            n = len(script)
            tags = parser.parse(script)
            self.assertEqual(n, len(tags), "number of tags should equal the number of script lines")
            for tag in tags:
                self.assertIn(tag, tagset, "tag should belong to tagset")
            pred.extend(tags)
        
        print("performance on <= 10 random noisy scripts")
        p, r, f1, _ = precision_recall_fscore_support(true, pred, labels=list(eval_tagset), average="micro")
        print(f"precision {p:.3f}, recall {r:.3f}, f1 {f1:.3f}")
    
if __name__=="__main__":
    unittest.main()