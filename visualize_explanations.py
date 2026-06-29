import saliency_evaluation.visualization
from saliency_evaluation.visualization import print_importance_dataset

INPUT_JSON = "/net/tscratch/people/plgpietron/agnews/Pullback_explanations.json"
OUTPUT_HTML = "/net/tscratch/people/plgpietron/agnews/pullback_s_.html"

print_importance_dataset(INPUT_JSON, "Bcos", OUTPUT_HTML, no_cls_sep=False, num_examples=-1)
