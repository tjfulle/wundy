import argparse
import sys

from .model import Model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file", help="Wundy yaml file")
    args = p.parse_args(sys.argv[1:])

    model = Model.from_file(args.file)
    model.prepare()
    model.solve()
    print(model.solution)


if __name__ == "__main__":
    sys.exit(main())
