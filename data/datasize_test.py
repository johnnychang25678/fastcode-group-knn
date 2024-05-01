from generate_data import generate_test_suite

def main():
    generate_test_suite([
        ("size_train1", 4096, 3),
        ("size_test1", 1924, 3),
        ("size_train2", 16384, 3),
        ("size_test2", 4096, 3),
        ("size_train3", 65536, 3),
        ("size_test3", 16384, 3)
    ])

if __name__ == "__main__":
    main()