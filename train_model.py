from models import SpamClassifier
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train the spam classifier model')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    print(f"Training model with dataset: {args.dataset}")
    classifier = SpamClassifier()
    accuracy = classifier.train(args.dataset, args.test_size, args.random_state)
    print(f"Training complete! Final accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()