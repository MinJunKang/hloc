
from pathlib import Path
from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction, visualization


images = Path('datasets/sacre_coeur/mapping/')
outputs = Path('outputs/sfm/')
sfm_pairs = outputs / 'pairs-netvlad.txt'
sfm_dir = outputs / 'sfm_superpoint+sgmnet'

retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['r2d2']
matcher_conf = match_features.confs['adalam']

# feature_conf = extract_features.confs['superpoint_aachen']
# matcher_conf = match_features.confs['superglue']
# matcher_conf = match_features.confs['sgmnet_sp']
# use sift with sgmnet_root
# use superpoint with sgmnet_sp

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)


feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)
model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)
visualization.visualize_sfm_2d(model, images, color_by='visibility', n=5)
visualization.visualize_sfm_2d(model, images, color_by='track_length', n=5)
visualization.visualize_sfm_2d(model, images, color_by='depth', n=5)