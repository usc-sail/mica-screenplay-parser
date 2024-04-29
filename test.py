from screenplayparser import ScreenplayParser

SCRIPT_PATH = "data/screenplays/screenplays/44_inch_chest.txt"

# instantiate a transformer-based parser by setting use_rules=False
# device_id is the GPU id the parser will use
trx_parser = ScreenplayParser(use_rules=False, device_id=1)

# instantiate a rule-based parser by setting use_rules=True
rule_parser = ScreenplayParser(use_rules=True)

# read a script and save it as a list of strings
# SCRIPT_PATH is the filepath of the movie script
with open(SCRIPT_PATH) as reader:
    script = reader.read().split("\n")

# trx_tags contains the tag per script line found by the transformer-based parser
# rule_tags contains the tag per script line found by the rule-based parser
trx_tags = trx_parser.parse(script)
rule_tags = rule_parser.parse(script)