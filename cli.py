"""Interactive command-line demo of the agent. Run with: python cli.py"""

from agent import Agent


def main() -> None:
    agent = Agent()
    print("Payment collection agent (Ctrl+C or 'quit' to exit)\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit"):
                break
            result = agent.next(user_input)
            print(f"Agent: {result['message']}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
