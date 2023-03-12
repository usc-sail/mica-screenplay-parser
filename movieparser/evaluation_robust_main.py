"""Main program for robust evaluation of robust parser"""
from mica_text_parser.movieparser.evaluation_robust import RobustEvaluation

from absl import app
from absl import flags
import os
import time

FLAGS = flags.FLAGS
data_dir = os.getenv("DATA_DIR")

flags.DEFINE_string("results_dir", default=None, help="results folder")
flags.DEFINE_string("output_file", default=None, help="output file")
flags.DEFINE_multi_enum("error", default=[], 
                        enum_values=["replace_name_with_scene_kw",
                                     "replace_name_with_transition_kw",
                                     "remove_scene_kw",
                                     "lowercase_scene_line",
                                     "lowercase_character_line",
                                     "create_watermark_line",
                                     "insert_asterisks_or_numbers",
                                     "insert_dialogue_expressions"], 
                        help="error type")
flags.DEFINE_multi_float("error_rate", default=[], lower_bound=0, upper_bound=1, help="error rate")
flags.DEFINE_integer("n_trials", default=10, help="Number of trials")

def main(argv):
    annotator_1_file = os.path.join(data_dir, "mica_text_parser/data/Annotator_1.xlsx")
    annotator_2_file = os.path.join(data_dir, "mica_text_parser/data/Annotator_2.xlsx")
    annotator_3_file = os.path.join(data_dir, "mica_text_parser/data/Annotator_3.xlsx")
    screenplays_folder = os.path.join(data_dir, "mica_text_parser/data/SAIL_annotation_screenplays/screenplays")
    screenplays_line_numbers_file = os.path.join(data_dir, 
                                                 "mica_text_parser/data/SAIL_annotation_screenplays/line_indices.txt")
    RobustEvaluation(annotator_1_file=annotator_1_file, 
                     annotator_2_file=annotator_2_file, 
                     annotator_3_file=annotator_3_file, 
                     screenplays_folder=screenplays_folder, 
                     screenplays_line_numbers_file=screenplays_line_numbers_file, 
                     results_folder=os.path.join(data_dir, "mica_text_parser/results/", FLAGS.results_dir), 
                     output_file=os.path.join(data_dir, "mica_text_parser/results/", FLAGS.results_dir, 
                                              FLAGS.output_file), 
                     names_file=os.path.join(data_dir, "mica_text_parser/data/names.txt"), 
                     errors=FLAGS.error, 
                     error_rates=FLAGS.error_rate, 
                     n_trials=FLAGS.n_trials)

if __name__=="__main__":
    app.run(main)