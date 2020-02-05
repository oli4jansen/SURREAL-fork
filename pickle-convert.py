import argparse
import joblib
import pickle

def main(filename):
    data = joblib.load(filename)
    pickle.dump(data, open(filename + "new", 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()

    main(args.file)
