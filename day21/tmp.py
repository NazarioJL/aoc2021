from collections import defaultdict


if __name__ == "__main__":
    start_pos = [7, 2]  # my input

    turn = 0
    universes = defaultdict(int)

    universes[(start_pos[0], start_pos[1], 0, 0)] = 1

    # Count all possible outcomes of rolling 3 3-sided dice
    possible_rolls = defaultdict(int)
    for d1 in range(1, 4):
        for d2 in range(1, 4):
            for d3 in range(1, 4):
                possible_rolls[d1 + d2 + d3] += 1

    # possible_rolls[1] = 1
    # possible_rolls[2] = 1
    # possible_rolls[3] = 1

    # Play games in all universes at once
    in_progress = True
    while in_progress:
        in_progress = False
        next = defaultdict(int)
        for key in universes:
            p1, p2, s1, s2 = key
            if max([s1, s2]) < 21:
                in_progress = True
                for roll in possible_rolls:
                    p1, p2, s1, s2 = key
                    if turn == 0:
                        p1 = (p1 - 1 + roll) % 10 + 1
                        s1 += p1
                    else:
                        p2 = (p2 - 1 + roll) % 10 + 1
                        s2 += p2
                    next[(p1, p2, s1, s2)] += possible_rolls[roll] * universes[key]
            else:
                if universes[key]:
                    next[key] += universes[key]
        print(f"{turn+1}. {len(next)} {sum(next.values())}")
        universes = next
        turn = (turn + 1) % 2

    win1 = sum([count for key, count in universes.items() if key[2] >= 21])
    win2 = sum([count for key, count in universes.items() if key[3] >= 21])
    print(win1, " ", win2)

    print(max([win1, win2]))
