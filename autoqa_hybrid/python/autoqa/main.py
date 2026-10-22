"""CLI entrypoint for the hybrid AutoQA agent."""
import argparse
import asyncio
import pathlib
import sys

from dotenv import load_dotenv

try:  # packaging import; works when installed via maturin
    from autoqa.agent import AutoQAAgent
except ModuleNotFoundError:
    _PKG_ROOT = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_PKG_ROOT.parent))
    from autoqa.agent import AutoQAAgent

load_dotenv()


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autoqa-main",
        description="AutoQA hybrid agent CLI (Python core + Rust native engine)",
    )
    parser.add_argument("--url", required=True, help="Target application URL")
    parser.add_argument(
        "--goal", required=True, help="User journey goal or assertion text"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Upper bound on agent steps (default: 10)",
    )
    args = parser.parse_args()

    agent = AutoQAAgent(goal=args.goal, max_steps=args.max_steps)
    history = await agent.run(start_url=args.url)
    print(f"\n[AutoQA Core] ✅ Finished with {len(history)} step(s) recorded.")


if __name__ == "__main__":
    asyncio.run(main())
