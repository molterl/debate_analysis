import argparse


def remove_extra_spaces(text):
    # Split the text into lines
    lines = text.split("\n")
    # Strip leading and trailing spaces from each line and join them into a single paragraph
    formatted_text = " ".join(line.strip() for line in lines if line.strip())
    return formatted_text


def main(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        unformatted_text = file.read()
    formatted_text = remove_extra_spaces(unformatted_text)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(formatted_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and format text.")
    parser.add_argument("input_file", type=str, help="Path to the input text file")
    parser.add_argument("output_file", type=str, help="Path to the output text file")
    args = parser.parse_args()

    main(args.input_file, args.output_file)
