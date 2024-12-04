import pandas as pd

def process_file(file_path, case):
    # Read the text file
    with open(f"{file_path}.txt", "r") as file:
        data = file.read()

    # Split the text into sections based on the hashtags
    sections = data.split("#")
    sections = [section.strip() for section in sections if section.strip()]  # Remove empty and whitespace-only entries

    if case == "nq-noret":
        # Case 1: Single column 'text' with '###' appended
        formatted_sections = [section + " ###" for section in sections]
        df = pd.DataFrame(formatted_sections, columns=["text"])
    
    elif case.startswith("nq-rank"):
        # Case 2: Extract question, full text, and hardcoded state
        questions = []
        texts = []
        if case == "nq-rank-top1":
            state = "[[(False, False)]]"
        elif case == "nq-rank-top10":
            state = "[[(True, False)]]"
        elif case == "nq-rank-random":
            state = "[[(False, True)]]"

        for section in sections:
            # Extract the question
            question_start = section.find("Question:")
            question_end = section.find("Are follow up questions needed here:")
            if question_start != -1 and question_end != -1:
                question = section[question_start:question_end].strip()
                questions.append(question)
            else:
                questions.append("")  # Handle case where question might not be found

            # Add the full text section with '###' appended
            texts.append(section + " ###")

        # Create DataFrame for case 2
        df = pd.DataFrame({
            "question": questions,
            "text": texts,
            "10/Random": [state] * len(texts)  # Same hardcoded value for all rows
        })
    else:
        raise ValueError("Invalid case.")

    # Save to CSV
    df.to_csv(f"{file_path}.csv", index=False)
    print(f"Data saved to {file_path}.csv")



process_file("nq_top1_own", "nq-rank-top1")

process_file("nq_top10_own", "nq-rank-top10")

process_file("nq_random_own", "nq-rank-random")
