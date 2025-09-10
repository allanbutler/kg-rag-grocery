import sys

from sgs.app import hello_world


def main(args: list[str] | None = None) -> None:
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    print("This is the main routine and will serve as the package entrypoint.")
    hello_world()


if __name__ == "__main__":
    main()
