import argparse
from algorithms import run_sklearn_demo, run_tensorflow_demo, run_pytorch_demo

def main():
    parser = argparse.ArgumentParser(description="ML Education Lab CLI")
    parser.add_argument(
        "--framework",
        choices=["sklearn", "tensorflow", "pytorch"],
        required=True,
        help="Choose which framework demo to run"
    )
    parser.add_argument(
        "--dataset",
        choices=["iris", "mnist", "imdb"],
        default="iris",
        help="Choose which dataset to use (default: iris)"
    )
    args = parser.parse_args()

    print(f"🚀 Running {args.framework} demo on {args.dataset} dataset...")

    if args.framework == "sklearn":
        run_sklearn_demo(dataset=args.dataset)
    elif args.framework == "tensorflow":
        run_tensorflow_demo(dataset=args.dataset)
    elif args.framework == "pytorch":
        run_pytorch_demo(dataset=args.dataset)

    print("✅ Demo complete. Check logs/ and models/ for outputs.")

if __name__ == "__main__":
    main()
