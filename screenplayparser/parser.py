from .robust import RobustScreenplayParser
from .rule import parse_lines

from typing import List
import torch
import os

class ScreenplayParser:
    '''
    Screenplay Parser class. \\
    Instantiate objects of this class and specify the parsing method to tag screenplay lines. \\
    Give a device id if you choose to use GPU

    The parser takes as input an array of string and returns an array of string of same length. \\
    Each element of the returned array is one of the following:

        'S' : scene header
        'N' : scene description
        'C' : character name
        'D' : utterance
        'E' : expression
        'T' : transition
        'M' : metadata
        'O' : other

    For example, shown below is a typical screenplay portion and the tags found by the parser

        S: 36 EXT -- EXERCISE YARD -- DAY (1947) 36
        O:
        N: Exercise period. Red plays catch with Heywood and Jigger, 
        N: lazily tossing a baseball around. Red notices Andy off to 
        N: the side. Nods hello. Andy takes this as a cue to amble over. 
        N: Heywood and Jigger pause, watching.
        O:
        C: &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; ANDY
        E: &emsp;&emsp;&emsp; (offers his hand)
        D: &emsp;&emsp; Hello. I'm Andy Dufresne.
        O:
        N: Red glances at the hand, ignores it. The game continues.
        O:
        C: &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; RED
        D: &emsp;&emsp; The wife-killin' banker.
        O:
        C: &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; ANDY
        D: &emsp;&emsp; How do you know that?
        
    '''

    def __init__(self, use_rules=False, device_id=-1) -> None:
        '''
        initializer for ScreenplayParser class

        Parameters
        ==========

            `use_rules`
                type = bool \\
                set use_rules if you want to use a rule-based parser instead of the default transformer-based parser \\
                you will sacrifice accuracy but parsing will be much faster
            
            `device_id`
                type = int \\
                CUDA device index \\
                if you have a GPU, you can set device_id to a non-negative integer to use your GPU for parsing
        '''

        self.use_rules = use_rules
        if not self.use_rules:
            if torch.cuda.is_available() and isinstance(device_id, int) and device_id >= 0:
                device = torch.device(device_id)
            else:
                device_id = -1
                device = torch.device("cpu")
            self.parser = RobustScreenplayParser(38, 8, True, device_index=device_id)
            self.parser.load_state_dict(torch.load(os.path.join(os.getenv("PROJ_DIR"), 
                                                                "mica-screenplay-parser/screenplayparser/model.pt"), 
                                        map_location=device))
            self.parser.to(device)
            self.parser.eval()

    def parse(self, script: List[str]) -> List[str]:
        '''
        provides a list of tags for each line

        Parameters
        ==========

            `script`
                type = array of string

        Return
        ======

            the function returns an array of string of the same length as `script` \\
            each element can be one of the following

                'S' : scene header
                'N' : scene description
                'C' : character name
                'D' : utterance
                'E' : expression
                'T' : transition
                'M' : metadata
                'O' : other
        '''
        
        if self.use_rules:
            tags = parse_lines(script)
            tags = ["O" if tag == '0' else tag for tag in tags]
        
        else:
            i = 0
            n_empty_lines, cscript = [], []
            header_length, footer_length = 0, 0

            while i < len(script):
                if script[i].strip() == "":
                    j = i + 1
                    while j < len(script) and script[j].strip() == "":
                        j += 1
                    if j < len(script):
                        if i == 0:
                            header_length = j
                        else:
                            cscript.append("")
                            n_empty_lines.append(j - i)
                    else:
                        footer_length = j - i
                    i = j
                else:
                    cscript.append(script[i].strip())
                    n_empty_lines.append(0)
                    i += 1
            
            ctags = self.parser.parse(cscript)
            tags = []

            for ne, tag, line in zip(n_empty_lines, ctags, cscript):
                if ne > 0:
                    tags.extend(["O" for _ in range(ne)])
                elif line.strip() == "":
                    tags.append("O")
                else:
                    tags.append(tag)
        
            tags = ["O" for _ in range(header_length)] + tags + ["O" for _ in range(footer_length)]
        
        return tags