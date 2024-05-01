import random
import argparse
import threading

def generate_data(file_path, row, dim):
    with open(file_path, 'w') as file:
        file.write(f"{row} {dim}\n")  # write header
        for _ in range(row):
            data = ""
            # Generating random doubles for data's feature
            for _ in range(dim):
                data += str(random.uniform(0.0, 100.0)) + " "
            # Generating a random integer for data's label
            data += str(random.randint(0, 1)) + "\n"
            file.write(data)

def generate_test_suite(args):
    threads = []
    for arg in args:
        threads.append(threading.Thread(target=generate_data, args=arg))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    # Setting up argument parsing
    parser = argparse.ArgumentParser(
        description="Generate data and write to a file."
    )
    parser.add_argument("file_name", type=str, help="The name of the output file")
    parser.add_argument("row", type=int, help="The number of rows to generate")
    parser.add_argument("dim", type=int, help="The number of data's feature")

    # Parsing arguments
    args = parser.parse_args()

    # Generating data and writing to the file
    generate_data(args.file_name, args.row, args.dim)
