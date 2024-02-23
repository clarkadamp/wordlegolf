import random
import sys
from argparse import ArgumentParser, Namespace
from typing import Callable, Iterable, Mapping, Sequence, Tuple

LetterGernerator = Callable[[bool], Iterable[Tuple[str, str]]]

all_letters = list(map(chr, range(ord("a"), ord("z") + 1)))

GENERATORS: Mapping[str, LetterGernerator] = {}


def random_letters() -> Sequence[str]:
    letters = all_letters.copy()
    random.shuffle(letters)
    return letters


def generator(func: LetterGernerator) -> LetterGernerator:
    GENERATORS[func.__name__.replace("_", "-")] = func
    return func


def skins_teams() -> None:
    adam_heather = ["adam", "heather"]
    random.shuffle(adam_heather)

    players = ["danc", "danz", "ina", "ross"]
    random.shuffle(players)

    team_a = adam_heather[:1] + players[:2]
    team_b = adam_heather[1:] + players[2:]

    print(f"Team A: {', '.join(sorted(team_a))}")
    print(f"Team B: {', '.join(sorted(team_b))}")


@generator
def no_repeat(allow_same: bool) -> Iterable[Tuple[str, str]]:
    for daily, bonus in zip(random_letters(), random_letters()):
        if not allow_same and daily == bonus:
            continue
        yield daily, bonus


@generator
def full_random(allow_same: bool) -> Iterable[Tuple[str, str]]:
    while True:
        daily = random_letters()[0]
        bonus = random_letters()[0]
        if not allow_same and daily == bonus:
            continue
        yield daily, bonus


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--generator", choices=sorted(GENERATORS), default="no-repeat")
    parser.add_argument("--allow-same", action="store_true", default=False)
    parser.add_argument("--skins", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.skins:
        skins_teams()

    print(f"Mode: {' '.join(sys.argv[1:])}")
    for hole, (d, b) in enumerate(GENERATORS[args.generator](args.allow_same), 1):
        print(f"Hole: {hole}, LOTD: {d.upper()} Bonus: {b.upper()}")
        if hole >= 18:
            break


if __name__ == "__main__":
    main()
