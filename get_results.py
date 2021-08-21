from sin2eeg import get_results

if __name__ == "__main__":
    with open('signal.txt', 'r') as file:
        signal = file.read().replace('\n', '')
    get_results(signal)
