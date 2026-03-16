from langchain.agents import create_agent

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input


def run():
    # Get predefined attributes
    model = get_model(temperature=0.9, top_p=0.95)

    # Create agent
    agent = create_agent(
        model=model,
        system_prompt=(
            "Du är en hjälpsam assistent som svarar på användarens frågor."
            "Svara alltid på svenska och var koncis men informativ."
        ),
    )

    # Get user input
    conversation = []

    while True:
        user_input = get_user_input("Ställ din fråga(skriv 'exit' för att avsluta)")

        if user_input.lower() == "exit":
            break

        conversation.append({"role": "user", "content": user_input})

        # Call the agent
        process_stream = agent.stream(
            {"messages": conversation},
            stream_mode=STREAM_MODES,
        )

        # Stream the process
        handle_stream(process_stream)


if __name__ == "__main__":
    run()
