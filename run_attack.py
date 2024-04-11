import time
from attack import attack

def main():
    while True:
        try:
            prompt = input("Enter your prompt, or type 'exit' to quit: ")
            if prompt.lower() == 'exit':
                break

            start = time.time()
            original_response, attacked_prompt, attacked_response = attack(prompt)
            elapsed = time.time() - start

            print(f"Original prompt:   {prompt}")
            print(f"Original response: {original_response}")
            print(f"Attacked prompt:   {attacked_prompt}")
            print(f"Attacked response: {attacked_response}")
            print(f"Elapsed:           {elapsed:.3f}s")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
