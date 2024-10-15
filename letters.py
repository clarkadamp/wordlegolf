#!/usr/bin/env python3

from __future__ import annotations

import csv
import logging
import random
import re
import string
import sys
import urllib.request
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    Iterable,
    List,
    MutableMapping,
    NamedTuple,
    Optional,
    Protocol,
    Type,
    TypedDict,
    cast,
)

WORDLE_WORD_LENGTH = 5
GENERATORS: MutableMapping[str, Type[Generator]] = {}
PUBLIC_WORD_LIST = "https://gist.githubusercontent.com/dracos/dd0668f281e685bad51479e5acaadb93/raw/6bfa15d263d6d5b63840a8e5b64e04b382fdb079/valid-wordle-words.txt"

logger = logging.getLogger("letters")


class Generator(Protocol):
    @classmethod
    def add_subcommand(cls, parser: ArgumentParser) -> None:
        """Add subcommand and generator specific options"""
        ...

    def generate_letters(self) -> Iterable[Letters]:
        """Actually generate the letter sequence"""
        ...


class Letters(NamedTuple):
    lotd: str
    bonus: str
    extra: Optional[str]


def generator(generator_cls: Type[Generator]) -> None:
    name = re.sub(r"([A-Z])", r"-\1", generator_cls.__name__).lower().strip("-")
    GENERATORS[name] = generator_cls


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debugging",
    )
    parser.add_argument(
        "--skins",
        action="store_true",
        default=False,
        help="Generate skins team allocation",
    )
    subparsers = parser.add_subparsers(
        dest="generator", description="Choose a letter generator:", required=True
    )
    for generator_name, generator_cls in GENERATORS.items():
        generator_cls.add_subcommand(
            subparsers.add_parser(
                name=generator_name,
                help=generator_cls.__doc__,
            )
        )
    args = dict(parser.parse_args()._get_kwargs())

    logging.basicConfig(level=logging.DEBUG if args.pop("debug") else logging.INFO)

    if args.pop("skins"):
        skins_teams()

    generator_name = args.pop("generator")
    generator = GENERATORS[generator_name](**args)

    print(
        f"Invoked: {parser.prog} {' '.join(sys.argv[sys.argv.index(generator_name):])}"
    )
    for hole, (d, b, e) in enumerate(generator.generate_letters(), 1):
        print(
            f"Hole: {hole: >2}, LOTD: {d.upper()}, Bonus: {b.upper()}{' - ' + e if e else ''}"
        )
        if hole >= 18:
            break


@generator
class NoRepeat:
    """Shuffled once per round"""

    def __init__(self, allow_same: bool) -> None:
        self.allow_same = allow_same

    @classmethod
    def add_subcommand(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--allow-same",
            action="store_true",
            default=False,
            help="Allow daily letters to be the same",
        )

    def generate_letters(self) -> Iterable[Letters]:
        for daily, bonus in zip(random_letters(), random_letters()):
            if not self.allow_same and daily == bonus:
                continue
            yield Letters(daily, bonus, None)


@generator
class FullRandom:
    """Fully random every hole"""

    def __init__(self, allow_same: bool) -> None:
        self.allow_same = allow_same

    @classmethod
    def add_subcommand(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--allow-same",
            action="store_true",
            default=False,
            help="Allow daily letters to be the same",
        )

    def generate_letters(self) -> Iterable[Letters]:
        while True:
            daily = random_letters()[0]
            bonus = random_letters()[0]
            if not self.allow_same and daily == bonus:
                continue
            yield Letters(daily, bonus, None)


@generator
class LastBonusIsLotd:
    """Daily bonus letter is random and becomes the next days LOTD"""

    @classmethod
    def add_subcommand(cls, parser: ArgumentParser) -> None:
        pass

    def generate_letters(self) -> Iterable[Letters]:
        letters = random_letters()
        daily = letters.pop()
        bonus = letters.pop()
        while True:
            yield Letters(daily, bonus, None)
            daily = bonus
            bonus = letters.pop()


@generator
class LotdBonusExist:
    """Ensure words exist for LOTD and Bonus"""

    def __init__(
        self,
        words_source: str,
        allow_same: bool,
        allow_duplicate_letters: bool,
        only_same_letters: bool,
        min_limit: int,
        limit_dropoff: int,
    ) -> None:
        self.words_source = words_source
        self.allow_same = allow_same
        self.allow_duplicate_letters = allow_duplicate_letters
        self.only_same_letters = only_same_letters
        self.min_limit = min_limit
        self.limit_dropoff = limit_dropoff / 100

    @classmethod
    def add_subcommand(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--words-source",
            default=PUBLIC_WORD_LIST,
            help="Location of a word list",
            metavar="FILE|URL",
        )
        parser.add_argument(
            "--allow-same",
            action="store_true",
            default=False,
            help="Allow daily letters to be the same",
        )
        parser.add_argument(
            "--allow-duplicate-letters",
            action="store_true",
            default=False,
            help="Allow LOTD and bonus letter repeats",
        )
        parser.add_argument(
            "--only-same-letters",
            action="store_true",
            default=False,
            help="Everyday is a duplicate letter day",
        )
        parser.add_argument(
            "--min-limit",
            type=int,
            default=5,
            help="Ensure LOTD/Bonus combination that have n occurances are always available",
            metavar="NUM",
        )
        parser.add_argument(
            "--limit-dropoff",
            type=int,
            default=85,
            choices=range(1, 101),
            help="The count limit will drop off by this factor each hole",
            metavar="NUM",
        )

    def generate_letters(self) -> Iterable[Letters]:
        # Grab first and last letter, count occurances
        letter_sets = defaultdict(int)
        for word in self.words():
            if not word or len(word) != WORDLE_WORD_LENGTH:
                continue

            first, last = word[0], word[-1]

            if self.only_same_letters:
                if first != last:
                    continue
            elif not self.allow_same and first == last:
                continue

            # Increase the count for this letter set
            letter_sets[f"{first}{last}"] += 1

        def _select(max_count: int) -> str:
            # expand each letter set based on the count
            # any letter set that has a count above `max_count` will not be included
            _expanded = []
            for _combo, _count in letter_sets.items():
                if _count > max(max_count, self.min_limit):
                    continue
                _expanded.extend([_combo] * _count)
            random.shuffle(_expanded)
            return _expanded[0]

        limit = max(letter_sets.values())
        max_limit_len = len(str(limit))
        firsts = set()
        lasts = set()
        while True:
            letters = _select(limit)
            letter_count = letter_sets[letters]
            first, last = letters

            if first in firsts or last in lasts:
                continue

            yield Letters(
                first,
                last,
                f"Occurances: {letter_count: >{max_limit_len}}, "
                f"occurance limit: {limit: >{max_limit_len}}",
            )

            letter_sets.pop(letters)

            if not self.allow_duplicate_letters:
                firsts.add(first)
                lasts.add(last)

            old_limit = limit
            limit = int(limit * self.limit_dropoff)
            logging.debug(
                f"Reduced limit by a factor of {self.limit_dropoff}: ",
                f"{old_limit: >{max_limit_len}} -> {limit: >{max_limit_len}}",
            )

    def words(self) -> List[str]:
        contents: str
        if self.words_source.startswith("http"):
            contents = urllib.request.urlopen(self.words_source).read().decode()
        elif Path(self.words_source).exists():
            contents = Path(self.words_source).read_text()
        else:
            print(f"{self.words_source} is not a valid url or file")
            exit(1)
        return [word for word in contents.split("\n") if word]


@dataclass
class LetterPack:
    letters: MutableMapping[str, Letter]
    expand_factor: int

    def choose(self, remove: bool) -> Letter:
        expanded_letters: List[str] = []
        for _letter in self.letters.values():
            expanded_letters.extend(
                [_letter.value] * int(_letter.probability * self.expand_factor)
            )
        random.shuffle(expanded_letters)
        chosen = expanded_letters[0]
        if remove:
            return self.letters.pop(chosen)
        return self.letters[chosen]


@dataclass
class Letter:
    value: str
    probability: float
    next_letters: Optional[LetterPack]


class StatsRow(TypedDict):
    letter: str
    count: str  # int
    obsProb: str  # float
    last_a: str  # float
    last_b: str  # float
    last_c: str  # float
    last_d: str  # float
    last_e: str  # float
    last_f: str  # float
    last_g: str  # float
    last_h: str  # float
    last_i: str  # float
    last_j: str  # float
    last_k: str  # float
    last_l: str  # float
    last_m: str  # float
    last_n: str  # float
    last_o: str  # float
    last_p: str  # float
    last_q: str  # float
    last_r: str  # float
    last_s: str  # float
    last_t: str  # float
    last_u: str  # float
    last_v: str  # float
    last_w: str  # float
    last_x: str  # float
    last_y: str  # float
    last_z: str  # float


@generator
class StatsDriven:
    def __init__(
        self,
        data_file: Path,
        allow_same: bool,
        allow_duplicate_letters: bool,
        no_repeat_bonus_for: int,
    ) -> None:
        self.data_file = data_file
        self.expand_factor = 10000
        self.allow_duplicate_letters = allow_duplicate_letters
        self.allow_same = allow_same
        self.no_repeat_bonus_for = no_repeat_bonus_for

    @classmethod
    def add_subcommand(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--data-file",
            default=f"{Path(sys.argv[0]).parent.as_posix()}/data/WordleGolfBetterProbs.csv",
            help="Location of a letter distribution file",
            type=Path,
            metavar="FILE",
        )
        parser.add_argument(
            "--allow-same",
            action="store_true",
            default=False,
            help="Allow daily letters to be the same",
        )
        parser.add_argument(
            "--allow-duplicate-letters",
            action="store_true",
            default=False,
            help="Allow LOTD and bonus letter repeats",
        )
        parser.add_argument(
            "--no-repeat-bonus-for",
            type=int,
            default=5,
            help="Holes between a given bonus letter",
        )

    def generate_letters(self) -> Iterable[Letters]:
        """Actually generate the letter sequence"""
        while True:
            lotd = self.pack.choose(remove=not self.allow_duplicate_letters)
            assert lotd.next_letters is not None
            previous_bonus = []
            while True:
                bonus = lotd.next_letters.choose(
                    remove=not self.allow_duplicate_letters
                )
                if not self.allow_same and lotd.value == bonus.value:
                    continue
                if bonus in previous_bonus[self.no_repeat_bonus_for :]:
                    continue
                previous_bonus.append(bonus)
                break
            yield Letters(
                lotd.value,
                bonus.value,
                f"Probs: lotd: {lotd.probability*100:.2f} , bonus: {bonus.probability*100:.2f}",
            )

    @cached_property
    def pack(self) -> LetterPack:
        with open(self.data_file) as csvfile:
            return to_letter_pack(
                [cast(StatsRow, row) for row in csv.DictReader(csvfile)],
                self.expand_factor,
            )


def to_letter_pack(rows: List[StatsRow], expand_factor: int) -> LetterPack:
    pack = LetterPack(letters={}, expand_factor=expand_factor)
    for row in rows:
        letter = Letter(
            value=row["letter"],
            probability=float(row["obsProb"]),
            next_letters=LetterPack(
                letters={
                    _letter: Letter(
                        value=_letter,
                        probability=float(row[f"last_{_letter}"]),
                        next_letters=None,
                    )
                    for _letter in string.ascii_lowercase
                },
                expand_factor=expand_factor,
            ),
        )
        pack.letters[letter.value] = letter
    return pack


def random_letters() -> List[str]:
    letters = list(map(chr, range(ord("a"), ord("z") + 1)))
    random.shuffle(letters)
    return letters


def skins_teams() -> None:
    adam_heather = ["adam", "heather"]
    random.shuffle(adam_heather)

    players = ["danc", "danz", "ina", "ross"]
    random.shuffle(players)

    team_a = adam_heather[:1] + players[:2]
    team_b = adam_heather[1:] + players[2:]
    print("Skins team allocations:")
    print(f" Team A: {', '.join(sorted(team_a))}")
    print(f" Team B: {', '.join(sorted(team_b))}")
    print("")


if __name__ == "__main__":
    main()
