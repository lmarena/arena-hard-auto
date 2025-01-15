import argparse
import openai

def main(api_key):
    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",
        api_key=api_key,
    )

    examples = [
        [
            {'role': 'system', 'content': 'This is a system message that provides context, you are an assistent for user, answer his questions.'},
            {'role': 'user', 'content': 'How can I start coding?'},
        ],
        [
            {'role': 'user', 'content': 'How can I start coding?'},
            {'role': 'assistant', 'content': 'To do something, you can start by online courses and leetcode'},
            {'role': 'user', 'content': 'Could you give more details?'},
        ],
        [
            {'role': 'system', 'content': 'This is a system message that provides context, you are an assistent for user, answer his questions.'},
            {'role': 'user', 'content': 'How can I start coding?'},
            {'role': 'assistant', 'content': 'To do something, you can start by '},
        ],
        [
            {'role': 'user', 'content': 'How can I start coding?'},
            {'role': 'assistant', 'content': 'To do something, you can start by '},
        ],
        [
            {'role': 'system', 'content': 'This is a system message that provides context, you are an assistent for user, answer his questions.'},
            {'role': 'user', 'content': 'How can I start coding?'},
            {'role': 'assistant', 'content': 'To do something, you can start by online courses and leetcode'},
            {'role': 'user', 'content': 'Could you give more details?'},
        ]
    ]
    for example in examples:
        messages = example
        completion = None
        print(f"Testing messages: {messages}")
        try:
            completion = client.chat.completions.create(
                model="llama-2-7b-hf",
                messages=messages,
                temperature=0.5,
                max_tokens=200,
            )
        except Exception as e:
            print(f"Got error: {type(e)}, {e}")

        print(f"Got completion: {completion}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run OpenAI chat completions.')
    parser.add_argument('api_key', type=str, help='API key for OpenAI client')

    # Parse arguments
    args = parser.parse_args()

    # Call main function with parsed API key
    main(args.api_key)