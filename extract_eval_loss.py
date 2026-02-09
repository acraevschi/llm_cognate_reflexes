import json
import matplotlib.pyplot as plt
import os


def extract_and_plot_losses(json_path):
    """
    Extract training and evaluation losses from trainer_state.json and plot them.
    """
    # Load the JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract data
    log_history = data["log_history"]

    steps = []
    train_loss = []
    eval_loss = []

    # In the log history, training entries and eval entries alternate
    # Extract them by pairs (training followed by evaluation)
    for i in range(0, len(log_history), 2):
        if i + 1 < len(log_history):
            train_entry = log_history[i]
            eval_entry = log_history[i + 1]

            # Verify these entries correspond to the same step
            if (
                train_entry["step"] == eval_entry["step"]
                and "loss" in train_entry
                and "eval_loss" in eval_entry
            ):
                steps.append(train_entry["step"])
                train_loss.append(train_entry["loss"])
                eval_loss.append(eval_entry["eval_loss"])

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_loss, label="Training Loss", marker="o", color="blue")
    plt.plot(steps, eval_loss, label="Evaluation Loss", marker="s", color="red")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(os.path.dirname(json_path), "loss_plot.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    # Also print the data
    print("\nStep\tTrain Loss\tEval Loss")
    print("----\t----------\t---------")
    for i in range(len(steps)):
        print(f"{steps[i]}\t{train_loss[i]:.4f}\t\t{eval_loss[i]:.4f}")

    return steps, train_loss, eval_loss


def main():
    json_path = r"checkpoints/gemma-3-1b-pt/checkpoint-900/trainer_state.json"

    # Check if file exists
    if not os.path.isfile(json_path):
        print(f"File not found: {json_path}")
        print("Please provide the correct path to the trainer_state.json file.")
        return

    extract_and_plot_losses(json_path)


if __name__ == "__main__":
    main()
