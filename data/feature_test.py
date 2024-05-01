import threading
from generate_data import generate_data

def main():
    threads = []

    # Create threads
    threads.append(threading.Thread(target=generate_data, args=("feature_train1", 8192, 3)))
    threads.append(threading.Thread(target=generate_data, args=("feature_test1", 2048, 3)))
    threads.append(threading.Thread(target=generate_data, args=("feature_train2", 8192, 10)))
    threads.append(threading.Thread(target=generate_data, args=("feature_test2", 2048, 10)))
    threads.append(threading.Thread(target=generate_data, args=("feature_train3", 8192, 100)))
    threads.append(threading.Thread(target=generate_data, args=("feature_test3", 2048, 100)))

    # Start threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()