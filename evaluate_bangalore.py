
import os
import yaml
import numpy as np
import tensorflow as tf

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor

CONFIG = "data/model/dcrnn_bangalore_15min_smoke.yaml"

CHECKPOINT = (
    "data/model/bangalore_15min_smoke/"
    "dcrnn_DR_2_h_12_64-64_lr_0.01_bs_16_1103202622/"
    "models-0.0246-114"
)

def main():

    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    graph_file = config["data"]["graph_pkl_filename"]

    sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(
        graph_file
    )

    assert len(sensor_ids) == 100
    assert config["model"]["num_nodes"] == 100

    tf_config = tf.ConfigProto(
        device_count={"GPU": 0}
    )

    with tf.Session(config=tf_config) as sess:

        supervisor = DCRNNSupervisor(
            adj_mx=adj_mx,
            **config
        )

        # Initialize variables before restoring checkpoint.
        sess.run(tf.global_variables_initializer())

        print("\nRestoring best checkpoint...")
        supervisor.load(sess, CHECKPOINT)

        print("\nEvaluating test dataset...")
        supervisor.evaluate(sess)

        print("\nTEST EVALUATION COMPLETED.")

if __name__ == "__main__":
    main()
