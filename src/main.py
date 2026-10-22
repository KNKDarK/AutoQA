import asyncio
import argparse
import pathlib
import sys
from dotenv import load_dotenv

try:  # `python -m src.main` from the repository root
    from src.agent.core import AutoQAAgent
except ModuleNotFoundError:
    # `python src/main.py` — put the repository root on sys.path so that the
    # `src.*` package imports inside agent/core.py and browser/driver.py work.
    _ROOT = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_ROOT))
    from src.agent.core import AutoQAAgent

load_dotenv()

async def main():
    parser = argparse.ArgumentParser(description="AutoQA Agent CLI")
    parser.add_argument("--url", required=True, help="Target application URL")
    parser.add_argument("--goal", required=True, help="User journey goal or assertion text")
    args = parser.parse_args()

    agent = AutoQAAgent(goal=args.goal)
    await agent.run(start_url=args.url)

if __name__ == "__main__":
    asyncio.run(main())
