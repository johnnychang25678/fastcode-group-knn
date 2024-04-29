import random
import argparse


def generate_data_and_write_to_file(N, file_path):
    with open(file_path, 'w') as file:
        file.write("3 2\n")  # write header
        for _ in range(N):
            # Generating random doubles for the first two columns
            col1 = random.uniform(0.0, 100.0)
            col2 = random.uniform(0.0, 100.0)
            # Generating a random integer for the third column
            col3 = random.randint(0, 10)
            file.write(f"{col1} {col2} {col3}\n")


def main():
    # Setting up argument parsing
    parser = argparse.ArgumentParser(
        description="Generate data and write to a file.")
    parser.add_argument("N", type=int, help="The number of rows to generate")
    parser.add_argument("file_name", type=str,
                        help="The name of the output file")

    # Parsing arguments
    args = parser.parse_args()

    # Generating data and writing to the file
    generate_data_and_write_to_file(args.N, args.file_name)

    print(
        f"Data has been written to {args.file_name}, including the header '3 2'.")


if __name__ == "__main__":
    main()
