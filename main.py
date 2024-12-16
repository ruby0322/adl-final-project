from base import Agent
from execution_pipeline import main
from pathlib import Path
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

class ClassificationAgent(Agent):
    """
    An agent that classifies text into one of the labels in the given label set.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize your LLM here
        """
        # Save the configuration
        self.config = config

        # Define the model path
        self.model_path = Path(self.config.get("model_path", "./saved_models/qwen-7b"))

        # Check if the model is already downloaded
        if not self.model_path.exists():
            print(f"Model not found locally. Downloading and saving to {self.model_path}...")
            self.model_path.mkdir(parents=True, exist_ok=True)
            model_name = self.config.get("model", "Qwen/Qwen2.5-7B-Instruct")
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.save_pretrained(self.model_path)
            tokenizer.save_pretrained(self.model_path)
        else:
            print(f"Loading model from local path: {self.model_path}")

        # Initialize the text classification pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model=str(self.model_path),
            tokenizer=str(self.model_path),
            device=self.config.get("device", 1)  # Use -1 for CPU or specify GPU ID
        )

    def __call__(
        self,
        label2desc: dict[str, str],
        text: str
    ) -> str:
        """
        Classify the text into one of the labels.

        Args:
            label2desc (dict[str, str]): A dictionary mapping each label to its description.
            text (str): The text to classify.

        Returns:
            str: The label (should be a key in label2desc) that the text is classified into.
        """
        # Extract labels and their descriptions
        labels = list(label2desc.keys())
        label_descriptions = list(label2desc.values())

        # Perform zero-shot classification
        result = self.classifier(text, candidate_labels=label_descriptions)

        # Map the predicted description back to the label
        predicted_label = labels[label_descriptions.index(result["labels"][0])]

        return predicted_label

    def update(self, correctness: bool) -> bool:
        """
        Update your LLM agent based on the correctness of its own prediction at the current time step.

        Args:
            correctness (bool): Whether the prediction is correct.

        Returns:
            bool: Whether the prediction is correct.
        """
        # No update logic for zero-shot classification; return the correctness as-is
        return correctness

if __name__ == "__main__":
    from argparse import ArgumentParser
    from execution_pipeline import main

    parser = ArgumentParser()
    parser.add_argument('--bench_name', type=str, required=True)
    args = parser.parse_args()

    if args.bench_name.startswith("classification"):
        agent_name = ClassificationAgent
    else:
        raise ValueError(f"Invalid benchmark name: {args.bench_name}")

    bench_cfg = {
        'bench_name': args.bench_name
    }
    config = {
        "model": "Qwen/Qwen2.5-7B-Instruct",  # Specify the model name here
        "model_path": "./saved_models/qwen-7b",  # Local model directory
        "device": 1  # Use -1 for CPU, or specify GPU ID
    }
    agent = agent_name(config)
    main(agent, bench_cfg)