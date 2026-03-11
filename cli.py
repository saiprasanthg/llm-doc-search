import argparse
import json
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv

# Reuse the same store implementation and config as the API
import main

load_dotenv()


def _build_store() -> main.RagStore:
    store = main.RagStore(main.VECTORSTORE_PATH)
    store.load()
    return store


def _print_json(payload):
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _render_search(results, max_chars: int = 280):
    if not results:
        print("No results.")
        return
    for idx, item in enumerate(results, start=1):
        text = (item.get("text") or "").strip().replace("\n", " ")
        snippet = text[:max_chars] + ("..." if len(text) > max_chars else "")
        meta = item.get("metadata", {})
        source = meta.get("source", "unknown")
        chunk_id = meta.get("chunk_id", "?")
        score = item.get("score", None)
        print(f"{idx}. {snippet}")
        print(f"   source: {source} | chunk: {chunk_id} | score: {score}")


def cmd_ingest(args):
    store = _build_store()
    patterns = args.pattern if args.pattern else None
    result = store.ingest(args.source_dir, patterns=patterns, persist=not args.no_persist)
    _print_json(result)


def cmd_search(args):
    store = _build_store()
    results = store.search(args.query, args.top_k)
    if args.json:
        _print_json({"query": args.query, "results": results, "count": len(results)})
    else:
        _render_search(results)


def cmd_answer(args):
    store = _build_store()
    result = store.answer(args.query, args.top_k)
    if args.json:
        _print_json(result)
        return
    print("Answer:\n")
    print(result.get("answer", ""))
    print("\nSources:")
    for src in result.get("sources", []):
        print(f"- {src.get('source')} (chunk {src.get('chunk_id')})")


def cmd_stats(_args):
    store = _build_store()
    if store.vectorstore is None:
        _print_json({"indexed_vectors": 0, "persist_path": main.VECTORSTORE_PATH, "loaded": False})
        return
    _print_json({"indexed_vectors": int(store.vectorstore.index.ntotal), "persist_path": main.VECTORSTORE_PATH, "loaded": True})


def interactive_shell():
    print("\nLLM Document Search & Analysis System (CLI)")
    print("Type the number for an action. Ctrl+C to exit.\n")

    store = _build_store()

    while True:
        print("1. Ingest documents")
        print("2. Semantic search")
        print("3. Ask a RAG question")
        print("4. Stats")
        print("5. Exit")
        choice = input("\nSelect: ").strip()

        if choice == "1":
            source_dir = input("Source dir (default: data): ").strip() or "data"
            patterns_raw = input("Patterns (comma-separated, blank for defaults): ").strip()
            patterns = [p.strip() for p in patterns_raw.split(",") if p.strip()] or None
            try:
                result = store.ingest(source_dir, patterns=patterns, persist=True)
                _print_json(result)
            except Exception as exc:
                print(f"Error: {exc}")
        elif choice == "2":
            query = input("Query: ").strip()
            if not query:
                print("Query required.")
                continue
            top_k_raw = input(f"Top K (default: {main.TOP_K_DEFAULT}): ").strip()
            top_k = int(top_k_raw) if top_k_raw else main.TOP_K_DEFAULT
            try:
                results = store.search(query, top_k)
                _render_search(results)
            except Exception as exc:
                print(f"Error: {exc}")
        elif choice == "3":
            query = input("Question: ").strip()
            if not query:
                print("Question required.")
                continue
            top_k_raw = input(f"Top K (default: {main.TOP_K_DEFAULT}): ").strip()
            top_k = int(top_k_raw) if top_k_raw else main.TOP_K_DEFAULT
            try:
                result = store.answer(query, top_k)
                print("\nAnswer:\n")
                print(result.get("answer", ""))
                print("\nSources:")
                for src in result.get("sources", []):
                    print(f"- {src.get('source')} (chunk {src.get('chunk_id')})")
            except Exception as exc:
                print(f"Error: {exc}")
        elif choice == "4":
            if store.vectorstore is None:
                print("No index loaded.")
            else:
                print(f"Indexed vectors: {int(store.vectorstore.index.ntotal)}")
                print(f"Persist path: {main.VECTORSTORE_PATH}")
        elif choice == "5":
            print("Bye.")
            return
        else:
            print("Invalid selection.")

        print("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI for the LLM Document Search & Analysis System")
    sub = parser.add_subparsers(dest="command")

    ingest = sub.add_parser("ingest", help="Ingest documents into the vector store")
    ingest.add_argument("--source-dir", default="data", help="Directory containing docs")
    ingest.add_argument("--pattern", action="append", help="Glob pattern to include (repeatable)")
    ingest.add_argument("--no-persist", action="store_true", help="Do not persist index to disk")
    ingest.set_defaults(func=cmd_ingest)

    search = sub.add_parser("search", help="Semantic search")
    search.add_argument("--query", required=True, help="Search query")
    search.add_argument("--top-k", type=int, default=main.TOP_K_DEFAULT, help="Number of results")
    search.add_argument("--json", action="store_true", help="Print raw JSON")
    search.set_defaults(func=cmd_search)

    answer = sub.add_parser("answer", help="RAG answer generation")
    answer.add_argument("--query", required=True, help="Question to answer")
    answer.add_argument("--top-k", type=int, default=main.TOP_K_DEFAULT, help="Number of results")
    answer.add_argument("--json", action="store_true", help="Print raw JSON")
    answer.set_defaults(func=cmd_answer)

    stats = sub.add_parser("stats", help="Show index stats")
    stats.set_defaults(func=cmd_stats)

    sub.add_parser("shell", help="Interactive menu")

    return parser


def main_entry():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "shell":
        interactive_shell()
        return

    args.func(args)


if __name__ == "__main__":
    main_entry()
