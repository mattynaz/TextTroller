from attack import attack

def main():
    while True:
        try:
            prompt = input("Enter your prompt, or type 'exit' to quit: ")
            if prompt.lower() == 'exit':
                break

            new_prompt, original_label, new_label = attack(prompt)

            print(f"Original prompt: {prompt}")
            print(f"Original label: {original_label}")
            print(f"Perturbed prompt: {new_prompt}")
            print(f"New label: {new_label}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
