# Wordle Hints
This code exposes several functions to help with your daily [Wordle](https://www.nytimes.com/games/wordle/index.html).

## Functionality
### Represent a Game
```py
# 0 = Gray
# 1 = Yellow
# 2 = Green
game = Game([
    ('guess', '10000'),
    ('words', '00000'),
    ('enjoy', '01002'),
    ('games', '12000'),
])    

print(game)
```
```
GUESS
▄____
WORDS
_____
ENJOY
_▄__█
GAMES
▄█___
```

### Masks
```py
# Get all possible "masks" using known information
print(', '.join(game.generate_masks(pretty=True)))
```
```
•AGNY, NAG•Y, NA•GY, •ANGY
```
### Get Possible Matches Using `words.txt` (Cheating)
```py
print(f'Matches: {", ".join(game.get_matches())}')
```
```
Matches: RANGY, NAGGY, TANGY, MANGY
```
### Heuristic Search
```py
list(game.generate_matches(verbose=True, time_limit=1))
```
```
(1.0s) Skipping SAGNY
(0.0s) Skipping NAGUY
       Found word: NAGGY
(1.0s) Skipping NAGUY
(1.0s) Skipping NAOGY
(0.0s) Skipping GANGY
       Found word: MANGY
(0.0s) Skipping VANGY
       Found word: RANGY
(0.0s) Skipping OANGY
       Found word: TANGY
(1.0s) Skipping NANGY

Checked 60362 words (4 unique valid, 97 unique invalid)
% valid (of total): 3.16%
% valid (of unique): 3.96%
Duplicate valids checked: 1902
Duplicate invalids checked: 58359
---
Matches: RANGY, NAGGY, TANGY, MANGY
```

## To-do list
- Include [`wordle-list`](https://github.com/tabatkins/wordle-list/blob/39ee14e80dc1ef9df55e682e01979a75ed1ee171/words) as a submodule to this repo
- Make sure functionality works as intended for target words & guesses with duplicate letters
- Improve heuristics by analyzing `words.txt` to find actual probability distribution of vowel/consonant for all combos of vowels/consonants before & after
