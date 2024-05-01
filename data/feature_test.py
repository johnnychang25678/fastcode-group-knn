from generate_data import generate_test_suite

def main():
    generate_test_suite([
        ("feature_train1", 16384, 3),
        ("feature_test1", 4096, 3),
        ("feature_train2", 16384, 10),
        ("feature_test2", 4096, 10),
        ("feature_train3", 16384, 100),
        ("feature_test3", 4096, 100)
    ])

if __name__ == "__main__":
    main()