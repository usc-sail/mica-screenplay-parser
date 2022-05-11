import unittest

class TestInitialize(unittest.TestCase):

    def test_rule(self):
        from screenplayparser import ScreenplayParser
        parser = ScreenplayParser(use_rules=True)
        self.assertIsInstance(parser, ScreenplayParser, "initializing rule parser: should be ScreenplayParser object")

    def test_robust_cpu(self):
        from screenplayparser import ScreenplayParser
        parser = ScreenplayParser(use_rules=False, device_id=-1)
        self.assertIsInstance(parser, ScreenplayParser, "initializing robust parser on CPU: should be ScreenplayParser object")

    def test_robust_gpu(self):
        from screenplayparser import ScreenplayParser
        parser = ScreenplayParser(use_rules=False, device_id=0)
        self.assertIsInstance(parser, ScreenplayParser, "initializing robust parser on GPU: should be ScreenplayParser object")

if __name__=="__main__":
    unittest.main()