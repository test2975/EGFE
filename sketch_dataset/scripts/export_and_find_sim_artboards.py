import json
import logging
from os import path

from sketch_dataset.datasets import find_similar_sketch
from sketch_dataset.utils import create_folder

if __name__ == "__main__":

    input_sketch_folder = 'F:/dataset/first_unlabeled/sketches'

    export_artboard_folder = "F:/dataset/first_unlabeled/sketch_artboards"
    export_artboard_logging_folder = path.join(export_artboard_folder, "logging")
    export_artboard_profile_folder = path.join(export_artboard_folder, "profile")

    similarity_res_folder = "sim_res"
    similarity_logging_folder = path.join(similarity_res_folder, "logging")
    similarity_profile_folder = path.join(similarity_res_folder, "profile")
    sim_groups_json = path.join(similarity_res_folder, "sim_groups.json")

    for folder in [export_artboard_folder, export_artboard_logging_folder, export_artboard_profile_folder,
                   similarity_res_folder, similarity_logging_folder, similarity_profile_folder]:
        create_folder(folder)

    logging.basicConfig(filename=path.join(similarity_res_folder, 'main.log'), level=logging.INFO)

    # export_artboards(
    #     glob.glob(f"{input_sketch_folder}/*.sketch"),
    #     export_artboard_folder,
    #     similarity_logging_folder,
    #     similarity_profile_folder,
    #     12
    # )

    # groups = merge_artboard_group(
    #     glob.glob(f"{export_artboard_folder}/*{path.sep}"),
    #     similarity_logging_folder,
    #     similarity_profile_folder,
    #     12
    # )
    # json.dump(
    #     groups,
    #     open(sim_groups_json, 'w')
    # )

    groups = json.load(open(sim_groups_json, 'r'))
    # visualize_groups(groups, similarity_res_folder)

    for group in find_similar_sketch(groups):
        print(group)
