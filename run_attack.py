import time
from attack import attack

def main():
    while True:
        try:
            prompt = input("Enter your prompt, or type 'exit' to quit: ")
            if not prompt:
                continue
            if prompt.lower() == 'exit':
                break

            start = time.time()
            original_response, attacked_prompt, attacked_response, num_calls = attack(prompt)
            elapsed = time.time() - start

            print(f"ORIGINAL PROMPT:   {prompt}")
            print(f"ORIGINAL RESPONSE: {original_response}")
            print(f"ATTACKED PROMPT:   {attacked_prompt}")
            print(f"ATTACKED RESPONSE: {attacked_response}")
            print(f"ELAPSED:           {elapsed:.3f}s")
            print(f"NUMBER CALLS:      {num_calls}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
