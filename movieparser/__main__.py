# author : Sabyasachee

# standard library
import os

# user library
from movieparser.read_args import read_args
from movieparser.evaluation import evaluate_parser
from movieparser.evaluation_gdi import evaluate_gdi
from movieparser.evaluation_robust import RobustEvaluation
from movieparser.create_data import create_data
from movieparser.create_seq_data import create_seq_data
from movieparser.create_feats import create_features
from movieparser.train import train
from movieparser.save_model import train_and_save
from movieparser.print_results import print_results

if __name__=="__main__":
    args = read_args()

    annotator_1 = os.path.join(args["data"], "Annotator_1.xlsx")
    annotator_2 = os.path.join(args["data"], "Annotator_2.xlsx")
    annotator_3 = os.path.join(args["data"], "Annotator_3.xlsx")
    parsed_folder = os.path.join(args["data"], "SAIL_annotation_screenplays")
    screenplays_folder = os.path.join(args["data"], "SAIL_annotation_screenplays/screenplays")
    line_indices = os.path.join(args["data"], "SAIL_annotation_screenplays/line_indices.txt")

    if args["mode"] == "evaluate-parser":
        evaluate_parser(annotator_1, annotator_2, annotator_3, parsed_folder, line_indices, use_robust=args["robust"])
    
    elif args["mode"] == "evaluate-parser-robust":
        robust_eval = RobustEvaluation(annotator_1, annotator_2, annotator_3, screenplays_folder, line_indices, names_file=args["names_file"], results_folder=args["results"])
        robust_eval.robust_evaluate()

    elif args["mode"] == "evaluate-gdi":
        gdi_folder = os.path.join(args["data"], "SAIL Team Spellcheck")
        evaluate_gdi(gdi_folder, args["gdi_folders"], ignore_scripts = args["ignore_scripts"], use_robust_parser = args["robust"], ignore_existing_parse = args["ignore_existing_parse"])
    
    elif args["mode"] == "create-data":
        create_data(annotator_1, annotator_2, annotator_3, line_indices, names_file=args["names_file"], results_folder=args["results"], screenplays_folder=screenplays_folder)
    
    elif args["mode"] == "create-seq":
        create_seq_data(args["results"], args["seqlen"])

    elif args["mode"] == "create-feats":
        create_features(args["results"])
    
    elif args["mode"] == "train":
        train(data_folder=args["data"], 
        results_folder=args["results"], 
        seqlen=args["seqlen"], 
        bidirectional=args["bidirectional"],
        train_batch_size=args["train_batch_size"], 
        eval_movie=args["eval_movie"], 
        learning_rate=args["learning_rate"], 
        encoder_learning_rate=args["enc_learning_rate"], 
        max_epochs=args["max_epochs"], 
        max_norm=args["max_norm"], 
        parallel=args["parallel"], 
        n_folds_per_gpu=args["n_folds_per_gpu"], 
        segment_lengths=args["eval_segment_lengths"],
        verbose=args["verbose"])
    
    elif args["mode"] == "deploy":
        train_and_save(results_folder=args["results"], seqlen=args["seqlen"], bidirectional=args["bidirectional"], train_batch_size=args["train_batch_size"], learning_rate=args["learning_rate"], encoder_learning_rate=args["enc_learning_rate"], max_norm=args["max_norm"], max_epochs=args["max_epochs"])

    elif args["mode"] == "print":
        print_results(results_folder=args["results"])