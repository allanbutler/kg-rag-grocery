import argparse
import uvicorn

from sgs.ingest.build_kg import main as build_graph
from sgs.ingest.build_index import main as build_index

def main():
    p = argparse.ArgumentParser("sgs")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare-data")
    sub.add_parser("run-server")

    args = p.parse_args()
    if args.cmd == "prepare-data":
        print("== Building KG =="); build_graph()
        print("== Building Vector Index =="); build_index()
        print("All set.")
    elif args.cmd == "run-server":
        uvicorn.run("sgs.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
