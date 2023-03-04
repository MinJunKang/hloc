
from pathlib import Path
from hloc import extract_features, match_dense, pairs_from_retrieval, reconstruction, visualization


images = Path('datasets/sacre_coeur/mapping/')
outputs = Path('outputs/sfm/')
sfm_pairs = outputs / 'pairs-netvlad.txt'
sfm_dir = outputs / 'sfm_dkm_outdoor'

retrieval_conf = extract_features.confs['netvlad']
matcher_conf = match_dense.confs['dkm_outdoor']
# use sift with sgmnet_root
# use superpoint with sgmnet_sp

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)
features, matches = match_dense.main(matcher_conf, sfm_pairs, images, outputs)

model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches)
visualization.visualize_sfm_2d(model, images, color_by='visibility', n=5)
visualization.visualize_sfm_2d(model, images, color_by='track_length', n=5)
visualization.visualize_sfm_2d(model, images, color_by='depth', n=5)
