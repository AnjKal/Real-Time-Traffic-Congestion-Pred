
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

SPEED_SCALE = 120.0


def calculate_metrics(pred, actual):
    # Match the original DCRNN evaluation mask: ignore zero speeds.
    mask = (actual != 0) & np.isfinite(actual) & np.isfinite(pred)

    errors = pred[mask] - actual[mask]

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors) / np.abs(actual[mask])) * 100

    return mae, rmse, mape


def main():

    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    _, _, adj_mx = load_graph_data(
        config["data"]["graph_pkl_filename"]
    )

    tf_config = tf.ConfigProto(
        device_count={"GPU": 0}
    )

    with tf.Session(config=tf_config) as sess:

        supervisor = DCRNNSupervisor(
            adj_mx=adj_mx,
            **config
        )

        sess.run(tf.global_variables_initializer())

        print("Restoring best DCRNN checkpoint...")
        supervisor.load(sess, CHECKPOINT)

        print("Evaluating DCRNN...")
        result = supervisor.evaluate(sess)

        # Predictions and actual values are inverse-standardized
        # by supervisor.evaluate(), but still divided by 120.
        dcrnn_pred = np.stack(
            result["predictions"], axis=1
        ) * SPEED_SCALE

        actual = np.stack(
            result["groundtruth"], axis=1
        ) * SPEED_SCALE

        # Persistence baseline:
        # Predict future speed as the last observed input speed.
        x_test = supervisor._data["x_test"]

        scaler = supervisor._data["scaler"]

        last_speed = scaler.inverse_transform(
            x_test[:, -1, :, 0]
        ) * SPEED_SCALE

        persistence_pred = np.repeat(
            last_speed[:, np.newaxis, :],
            actual.shape[1],
            axis=1
        )

        print("\n========== TEST RESULTS ==========")

        print(
            "\nDCRNN prediction shape:",
            dcrnn_pred.shape
        )

        print(
            "Actual speed shape:",
            actual.shape
        )

        print("\n--- Per-horizon comparison ---")

        for h in range(actual.shape[1]):

            d_mae, d_rmse, d_mape = calculate_metrics(
                dcrnn_pred[:, h, :],
                actual[:, h, :]
            )

            p_mae, p_rmse, p_mape = calculate_metrics(
                persistence_pred[:, h, :],
                actual[:, h, :]
            )

            print(
                "Horizon {:02d} ({:3d} min) | "
                "DCRNN MAE: {:.4f} km/h, "
                "RMSE: {:.4f}, MAPE: {:.2f}% | "
                "Persistence MAE: {:.4f} km/h, "
                "RMSE: {:.4f}, MAPE: {:.2f}%".format(
                    h + 1,
                    (h + 1) * 15,
                    d_mae, d_rmse, d_mape,
                    p_mae, p_rmse, p_mape
                )
            )

        print("\n--- Overall comparison ---")

        d_mae, d_rmse, d_mape = calculate_metrics(
            dcrnn_pred, actual
        )

        p_mae, p_rmse, p_mape = calculate_metrics(
            persistence_pred, actual
        )

        print(
            "DCRNN: MAE={:.4f} km/h, "
            "RMSE={:.4f} km/h, MAPE={:.2f}%".format(
                d_mae, d_rmse, d_mape
            )
        )

        print(
            "Persistence: MAE={:.4f} km/h, "
            "RMSE={:.4f} km/h, MAPE={:.2f}%".format(
                p_mae, p_rmse, p_mape
            )
        )

        if p_mae > 0:
            improvement = (
                (p_mae - d_mae) / p_mae
            ) * 100

            print(
                "DCRNN MAE improvement over persistence: "
                "{:.2f}%".format(improvement)
            )

        print("\nEVALUATION COMPLETED.")


if __name__ == "__main__":
    main()
