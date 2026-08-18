import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_log(csv_path, out_path):
    df = pd.read_csv(csv_path)

    fig, ax_loss = plt.subplots(figsize=(9, 5))
    ax_dice = ax_loss.twinx()

    ax_loss.plot(df["epoch"], df["loss"], color="tab:red", marker="o")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (log scale)", color="tab:red")

    ax_dice.plot(df["epoch"], df["dice"], color="tab:blue", marker="s")
    ax_dice.set_ylabel("Dice score", color="tab:blue")

    ax_loss.set_title("Training loss and validation Dice per epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training loss + validation Dice from a training_log.csv")
    parser.add_argument("csv_path", nargs="?", default="training_log.csv", help="Path to training_log.csv")
    parser.add_argument("-o", "--out", default="training_log.png", help="Output image path")
    args = parser.parse_args()

    plot_training_log(args.csv_path, args.out)
    print(f"wrote {args.out}")
