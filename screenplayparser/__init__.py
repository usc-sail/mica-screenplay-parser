from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

from mica_text_parser.screenplayparser.parser import ScreenplayParser