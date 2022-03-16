# author : Sabyasachee

# standard library
import os

# user library
from movieparser.read_args import read_args
from movieparser.evaluation import evaluate_parser
from movieparser.evaluation_gdi import evaluate_gdi
from movieparser.evaluation_robust import RobustEvaluation

if __name__=="__main__":
    args = read_args()

    annotator_1 = os.path.join(args["data"], "Annotator_1.xlsx")
    annotator_2 = os.path.join(args["data"], "Annotator_2.xlsx")
    annotator_3 = os.path.join(args["data"], "Annotator_3.xlsx")
    parsed_folder = os.path.join(args["data"], "SAIL_annotation_screenplays/parsed-screenplays")
    screenplays_folder = os.path.join(args["data"], "SAIL_annotation_screenplays/screenplays")
    line_indices = os.path.join(args["data"], "SAIL_annotation_screenplays/line_indices.txt")

    if args["mode"] == "evaluate-parser":
        evaluate_parser(annotator_1, annotator_2, annotator_3, parsed_folder, line_indices)
    
    elif args["mode"] == "evaluate-parser-robust":
        robust_eval = RobustEvaluation(annotator_1, annotator_2, annotator_3, \
                        screenplays_folder, line_indices, names_file=args["names_file"], data_folder=args["data"])
        robust_eval.robust_evaluate()

    elif args["mode"] == "evaluate-gdi":
        gdi_folder = os.path.join(args["data"], "SAIL Team Spellcheck")
        evaluate_gdi(gdi_folder, args["gdi_folders"])