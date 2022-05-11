# author : Sabyasachee

# standard library
import unittest

class TestEmpty(unittest.TestCase):

    def test_rule_parser_on_empty_list(self):
        import pandas as pd
        from screenplayparser import ScreenplayParser
        
        parser = ScreenplayParser(use_rules=True)

        tags = parser.parse([])
        self.assertListEqual(tags, [], "tags should be empty list")

    def test_robust_parser_on_empty_list(self):
        import pandas as pd
        from screenplayparser import ScreenplayParser
        
        parser = ScreenplayParser(device_id=0)

        tags = parser.parse([])
        self.assertListEqual(tags, [], "tags should be empty list")

if __name__=="__main__":
    unittest.main()