from movie_screenplay_parser.movieparser.scriptparser import ScriptParser
from typing import List
import torch

class MovieParser:

    def __init__(self, epoch = 6, device_id = 0) -> None:
        self.parser = ScriptParser(38, 8, True, device_index=device_id)
        self.parser.load_state_dict(torch.load(f"results/saved_models/epoch{epoch}.pt"))
        self.parser.to(torch.device(device_id))
        self.parser.eval()

    def parse(self, script) -> List[str]:
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