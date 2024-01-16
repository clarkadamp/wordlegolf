import random

all_letters = list(map(chr, range(ord("a"), ord("z") + 1)))


adam_heather = ["adam", "heather"]
random.shuffle(adam_heather)

players = ["danc", "danz", "ina", "ross"]
random.shuffle(players)

team_a = adam_heather[:1] + players[:2]
team_b = adam_heather[1:] + players[2:]


lotd = all_letters.copy()
bonus = all_letters.copy()
random.shuffle(lotd)
random.shuffle(bonus)


print(f"Team A: {', '.join(sorted(team_a))}")
print(f"Team B: {', '.join(sorted(team_b))}")

hole: int = 0
for d, b in zip(lotd, bonus):
    if d == b:
        continue
    hole += 1
    print(f"Hole: {hole}, LOTD: {d.upper()} Bonus: {b.upper()}")
    if hole >= 18:
        break
