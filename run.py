"""Research experiments for the movie screenplay parser development.

This code can be used to conduct different types of evaluations and train a sentence-transformer-based screenplay 
parser. The code runs in one of the following modes:

    1. evaluate_simple      : Evaluate the rule-based or transformer-based parser on the human-annotated screenplay
                              parse data.
    2. evaluate_issue       : Evaluate the rule-based or transformer-based parser on different synthetically
                              modified versions of the human-annotated data. The modifications reflect various
                              screenplay formatting issues. The error probabilities are varied.
    3. evaluate_linecount   : Evaluate the rule-based or transformer-based parser on the task of counting the speaking
                              lines of screenplay characters.
    4. create_data          : Create the synthetically modified versions of the clean human-annotated data. The error
                              probabilities are not varied for a specific formatting issue. This mode creates the
                              MovieParse dataset.
    5. create_seqdata       : Create sequences of screenplay lines from the MovieParse dataset used later for model
                              training.
    6. create_features      : Create the non-embedding-based features of the sequence data.
    7. train                : Train the transformer-based screenplay parser and perform leave-one-movie-out evaluation.
    8. deploy               : Train and save the weights of the transformer-based screenplay parser that will be used
                              by end users.
    9. print                : [TODO]

Usage:
    python run.py --mode=evaluate_simple [--data_dir=] [--results_dir=] [--trx]
    python run.py --mode=evaluate_issue [--data_dir=] [--results_dir=] [--trx] [--issue_file=] [--error=]+ 
        [--error_rate=]+ [--n_trials]
"""
from movie_screenplay_parser.movieparser import evaluation_simple
from movie_screenplay_parser.movieparser import evaluation_linecount
from movie_screenplay_parser.movieparser import evaluation_issue
from movie_screenplay_parser.movieparser import create_data
from movie_screenplay_parser.movieparser import create_seq_data
from movie_screenplay_parser.movieparser import create_feats
from movie_screenplay_parser.movieparser import train
from movie_screenplay_parser.movieparser import save_model
from movie_screenplay_parser.movieparser import print_results

from absl import app
from absl import flags
import os

error_values = ["replace_name_with_scene_kw", "replace_name_with_transition_kw", "remove_scene_kw",
                "lowercase_scene_line", "lowercase_character_line", "create_watermark_line",
                "insert_asterisks_or_numbers", "insert_dialogue_expressions"]
error_rates = [0.1, 0.5, 0.9, 0.01, 0.05, 0.15, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8]
movies = ["44_inch_chest","american_psycho","basic_instinct","bodyguard","bounty_hunter_the","broken_embraces",
          "burn_after_reading","candle_to_water","changeling","custody","dry_white_season_a","event_horizon","extract",
          "flight","gamer","grosse_pointe_blank","i_am_number_four","kids","lord_of_the_rings_fellowship_of","machete",
          "man_who_knew_too_much_the","memento","men_in_black_3","mirrors","nine","pandorum","petulia",
          "shawshank_redemption_the","spartan","star_trek_01_the_motion_picture","strange_days","stuntman_the",
          "suspect_zero","true_romance","up_in_the_air","up","willow","wolf_of_wall_street_the","xmen"]

FLAGS = flags.FLAGS
flags.DEFINE_enum("mode", "train", enum_values=["evaluate_simple", "evaluate_issue", "evaluate_linecount", 
                                                "create_data", "create_seq", "create_feats", "train", "deploy", 
                                                "print"], help="run mode")
flags.DEFINE_string("data_dir", default="data/", help="data directory")
flags.DEFINE_string("results_dir", default="results/", help="results directory")
flags.DEFINE_string("issue_file", default=None, help="output file for issue evaluation")
flags.DEFINE_string("linecounts_dir", default=None, help="directory containing scripts and word document files "
                    "containing character line counts")
flags.DEFINE_multi_string("ignore_script_linecount", default=[], help="script name that will be ignored from the "
                          "linecounts directory during line count evaluation")
flags.DEFINE_bool("trx", default=False, help="use transformer-based parser instead of the default rule-based parser")
flags.DEFINE_bool("reparse", default=False, help="overwrite existing parse output of scripts by reparsing them")
flags.DEFINE_bool("recount", default=False, help="overwrite existing line count dataframe by recounting line counts")
flags.DEFINE_integer("seqlen", default=10, help="number of sentences in a training sample")
flags.DEFINE_integer("batch_size", default=256, help="batch size used in training")
flags.DEFINE_enum("eval_movie", default=movies[0], enum_values=movies, 
                  help="movie left out in leave-one-movie-out validation")
flags.DEFINE_multi_integer("eval_seqlen",default=[10, 50, 100, 1000000], help="number of sentences in an inference "
                          "batch")
flags.DEFINE_multi_enum("error", default=error_values, enum_values=error_values + ["none"], 
                        help="error types for issue-based evaluation")
flags.DEFINE_multi_float("error_rate", default=error_rates, lower_bound=0, upper_bound=1, 
                         help="error rates for issue-based evaluation")
flags.DEFINE_integer("n_trials", default=10, help="number of trials for issue-based evaluation")
flags.DEFINE_float("lr", default=1e-3, help="model learning rate")
flags.DEFINE_float("sent_lr", default=1e-5, help="learning rate of the sentence encoder")
flags.DEFINE_float("max_norm", default=1, help="maximum norm of the weights used for gradient clipping")
flags.DEFINE_integer("max_epochs", default=10, help="maximum number of epochs")
flags.DEFINE_bool("parallel", default=False, help="train multiple folds of the leave-one-movie-out cross validation")
flags.DEFINE_integer("parallel_folds", default=1, help="number of folds to train parallelly in a single gpu")
flags.DEFINE_bool("bi", default=False, help="use bidirectional GRUs")
flags.DEFINE_bool("verbose", default=False, help="verbose output")

def movieparse(argv):
    if len(argv) > 1:
        print("too many command-line arguments")
        return
    
    mode = FLAGS.mode
    data_dir = FLAGS.data_dir
    results_dir = FLAGS.results_dir
    issue_file = FLAGS.issue_file
    linecounts_dir = FLAGS.linecounts_dir
    ignore_scripts_linecount = FLAGS.ignore_script_linecount
    trx = FLAGS.trx
    reparse = FLAGS.reparse
    recount = FLAGS.recount
    seqlen = FLAGS.seqlen
    batch_size = FLAGS.batch_size
    eval_movie = FLAGS.eval_movie
    eval_seqlen = FLAGS.eval_seqlen
    errors = FLAGS.error
    error_rates = FLAGS.error_rate
    n_trials = FLAGS.n_trials
    lr = FLAGS.lr
    sent_lr = FLAGS.sent_lr
    max_norm = FLAGS.max_norm
    max_epochs = FLAGS.max_epochs
    parallel = FLAGS.parallel
    parallel_folds = FLAGS.parallel_folds
    bi = FLAGS.bi
    verbose = FLAGS.verbose

    annotator_1 = os.path.join(data_dir, "Annotator_1.xlsx")
    annotator_2 = os.path.join(data_dir, "Annotator_2.xlsx")
    annotator_3 = os.path.join(data_dir, "Annotator_3.xlsx")
    parsed_folder = os.path.join(data_dir, "screenplays")
    screenplays_folder = os.path.join(data_dir, "screenplays/screenplays")
    line_indices = os.path.join(data_dir, "screenplays/line_indices.txt")
    names_file = os.path.join(data_dir, "names.txt")

    if mode == "evaluate_simple":
        evaluation_simple.evaluate(annotator_1, annotator_2, annotator_3, parsed_folder, line_indices, results_dir, 
                                   trx=trx)
    
    elif mode == "evaluate_issue":
        evaluation_issue.IssueEvaluation(annotator_1, annotator_2, annotator_3, screenplays_folder, line_indices, 
                                         results_dir, issue_file, names_file=names_file, errors=errors, 
                                         error_rates=error_rates, n_trials=n_trials, trx=trx)

    elif mode == "evaluate_linecount":
        gdi_folder = os.path.join(data_dir, "SAIL Team Spellcheck")
        evaluation_linecount.evaluate(gdi_folder, linecounts_dir, ignore_scripts=ignore_scripts_linecount, trx=trx,
                                      ignore_existing_parse=reparse, recalculate_line_counts=recount)
    
    elif mode == "create_data":
        create_data.create_data(annotator_1, annotator_2, annotator_3, line_indices, names_file=names_file, 
                                results_folder=results_dir, screenplays_folder=screenplays_folder)
    
    elif mode == "create_seq":
        create_seq_data.create_seq_data(results_dir, seqlen)

    elif mode == "create_feats":
        create_feats.create_features(results_dir)
    
    elif mode == "train":
        train.train(data_folder=data_dir, 
              results_folder=results_dir, 
              seqlen=seqlen, 
              bidirectional=bi,
              train_batch_size=batch_size, 
              eval_movie=eval_movie, 
              learning_rate=lr, 
              encoder_learning_rate=sent_lr, 
              max_epochs=max_epochs, 
              max_norm=max_norm, 
              parallel=parallel, 
              n_folds_per_gpu=parallel_folds, 
              segment_lengths=eval_seqlen,
              verbose=verbose)
    
    elif mode == "deploy":
        save_model.train_and_save(results_folder=results_dir, 
                       seqlen=seqlen, 
                       bidirectional=bi, 
                       train_batch_size=batch_size, 
                       learning_rate=lr, 
                       encoder_learning_rate=sent_lr, 
                       max_norm=max_norm, 
                       max_epochs=max_epochs)

    elif mode == "print":
        print_results.print_results(results_folder=results_dir)

if __name__ == '__main__':
    app.run(movieparse)