from typing import Dict, List, Literal, Optional, Set, Tuple, Union
import string
from collections import Counter, defaultdict
from copy import deepcopy
import random
import re
import time


ALPHABET = string.ascii_uppercase
VOWELS = 'AEIOU'
CONSONANTS = list((set(ALPHABET) - set(VOWELS)) | {'Y'})

ALPHABET_SET = set(ALPHABET)
VOWELS_SET = set(VOWELS)
CONSONANTS_SET = set(CONSONANTS)

with open('words.txt', 'r') as f:
    WORDS_TXT = f.read()
WORDS_SET = set(WORDS_TXT.splitlines())


class Result:
    """
    Represent the result of a guess: 
        0 = not in word
        1 = wrong spot
        2 = right spot
    ex:
        spore
        00120
    """

    str_key = {
        '0': '_',
        '1': '▄',
        '2': '█'
    }

    def __init__(self, result: str):
        assert all((n in ('0', '1', '2') for n in result))
        self.result = result

    def __str__(self):
        return ''.join(map(lambda i: Result.str_key[i], self.result))

    def __repr__(self):
        return f'Result({self.result})'

    def __len__(self):
        return len(self.result)

    def __iter__(self):
        yield from self.result

class Guess:
    def __init__(self,
                 word: str,
                 result: Union[str, Result]):
        if isinstance(result, str):
            result = Result(result)
        assert len(word) == len(result), f'word length ({len(word)}) must equal result length ({len(result)})'
        
        self.word = word.upper()
        self.result = result

    def __str__(self):
        return f'{self.word}\n{self.result}'

    def __repr__(self):
        return f'Guess({self.word}, {self.result})'

    def __len__(self):
        return len(self.word)

class Game:
    def __init__(self,
                 guesses: Union[List[Guess], List[Tuple[str, str]]] = None,
                 length: int = 5):
        if guesses is None:
            guesses = []
        elif not isinstance(guesses[0], Guess):
            assert all((len(result) == length for _, result in guesses))
            guesses = [Guess(word, result) for word, result in guesses]
        self.guesses = guesses

        self.remaining_letters = ALPHABET_SET.copy()

        # Hold locations of correct letter placements
        self.correct_mask: List[Optional[str]] = [None] * length
        # Map letters to a set of their possible locations (0-length-1 idxs)
        self.placement_constraints = defaultdict(lambda: set(range(length)))

        for guess in self.guesses:
            self.add_guess(guess)

    def __str__(self):
        return '\n'.join(str(guess) for guess in self.guesses)

    def add_guess(self, guess: Guess):
        # # Make sure duplicate letters have representation
        # included_letters = [ch for ch, res in zip(guess.word, guess.result) if res != 0]        

        for idx, (ch, res) in enumerate(zip(guess.word, guess.result)):
            if res == '0':
                self.remaining_letters -= {ch}
            elif res == '1':
                self.placement_constraints[ch] -= {idx, *(i for i, e in enumerate(self.correct_mask) if e)}
            else:
                assert res == '2'

                self.correct_mask[idx] = ch
                self.placement_constraints.pop(ch, None)

                # No other letters can go there
                for ch, idxs in self.placement_constraints.items():
                    idxs -= {idx}

        # self.remaining_letters -= {ch for ch, res in zip(guess.word, guess.result) if res == 0}

    def generate_masks(self,
                       mask=None,
                       constrs=None,
                       last_idx=-1,
                       pretty=False):
        if mask is None:
            mask = deepcopy(self.correct_mask)
        if constrs is None:
            constrs = deepcopy(self.placement_constraints)

        for ch, idxs in constrs.items():
            # print(ch, idxs)
            for idx in idxs:
                # Prevents duplicate outputs
                if last_idx > idx:
                    continue

                # Place letter here
                mask[idx] = ch
                
                # Remove this constraint
                constrs_copy = deepcopy(constrs)
                constrs_copy.pop(ch, None)

                # See if there are more constraints to satisfy: if so, recurse
                if not constrs_copy:
                    if pretty:
                        yield ''.join([e if e is not None else '•' for e in mask])
                    else:
                        yield mask
                else:
                    yield from self.generate_masks(mask=deepcopy(mask),
                                                   constrs={ch: idxs-{idx} for ch, idxs in constrs_copy.items()},
                                                   last_idx=idx,
                                                   pretty=pretty)

                # Reset mask
                mask[idx] = None

    def generate_matches(self, time_limit=15, verbose=False, pretty=True):
        invalid_guessed_words = set()
        duplicate_invalid_words = 0
        total_invalid_words = 0
        valid_guessed_words = set()
        duplicate_valid_words = 0
        total_valid_words = 0


        def fill_mask(mask):
            filled = mask.copy()

            empty_idxs = [i for i, e in enumerate(mask) if e is None]
            for idx in empty_idxs:
                # Get consonant/vowel before and after
                if idx-1 < 0:
                    before = None
                elif filled[idx-1] is None:
                    before = None
                else:
                    before = filled[idx-1] in CONSONANTS_SET

                if idx+1 >= len(mask):
                    after = None
                elif filled[idx+1] is None:
                    after = None
                else:
                    after = filled[idx+1] in CONSONANTS_SET

                # before = (mask[idx-1] in CONSONANTS_SET, None)[idx-1 < 0]
                # after = (mask[idx+1] in CONSONANTS_SET, None)[idx+1 >= len(mask)], print(mask, idx)

                r = random.random()
                if before is None and after:
                    # _ / consonant
                    # look equally for vowels/consonants
                    if r > 0.5:
                        ch = random.choice(VOWELS)
                    else:
                        ch = random.choice(CONSONANTS)
                elif before is None and not after:
                    # _ / vowel
                    # look mostly for consonsants
                    if r > 0.65:
                        ch = random.choice(VOWELS)
                    else:
                        ch = random.choice(CONSONANTS)
                elif before and after is None:
                    # consonant / _
                    # look equally for vowels/consonants
                    if r > 0.5:
                        ch = random.choice(VOWELS)
                    else:
                        ch = random.choice(CONSONANTS)
                elif not before and after is None:
                    # vowel / _
                    # look mostly for consonants
                    if r > 0.6:
                        ch = random.choice(VOWELS)
                    else:
                        ch = random.choice(CONSONANTS)
                elif before and after:
                    # consonant / consonant
                    # mostly look for vowels
                    if random.random() > 0.05:
                        ch = random.choice(VOWELS)
                    else:
                        ch = random.choice(CONSONANTS)
                elif not before and after:
                    # vowel / consonant
                    # look for vowel a little more
                    if random.random() > 0.4:
                        ch = random.choice(VOWELS)
                    else:
                        ch = random.choice(CONSONANTS)
                elif before and not after:
                    # consonant / vowel
                    # look for vowel a little more
                    if random.random() > 0.4:
                        ch = random.choice(VOWELS)
                    else:
                        ch = random.choice(CONSONANTS)
                else:
                    # vowel / vowel
                    # look for consonant a little more
                    if random.random() > 0.65:
                        ch = random.choice(VOWELS)
                    else:
                        ch = random.choice(CONSONANTS)
                filled[idx] = ch
            return filled

        for mask in self.generate_masks():
            has_skipped_before = False

            start_time = time.time()
            while True:
                filled = fill_mask(mask)
                filled_pretty = ''.join(filled)
                
                if filled_pretty in WORDS_SET:
                    total_valid_words += 1
                    already_guessed = filled_pretty in valid_guessed_words
                    duplicate_valid_words += int(already_guessed)  # 0 or 1
                    
                    if already_guessed:
                        continue

                    valid_guessed_words |= {filled_pretty}
                    
                    if verbose:
                        print(f'\n       Found word: {filled_pretty}')

                    if pretty:
                        yield filled_pretty
                    else:
                        yield filled
                else:
                    total_invalid_words += 1
                    duplicate_invalid_words += bool(filled_pretty in invalid_guessed_words)
                    invalid_guessed_words |= {filled_pretty}

                    if verbose:
                        if has_skipped_before:
                            print('\r', end='')

                        print(f'({time.time() - start_time:3.1f}s) Skipping {filled_pretty}', end='')
                        has_skipped_before = True

                end_time = time.time()
                if end_time - start_time >= time_limit:
                    if verbose:
                        print()
                    break

        if verbose:
            print()
            print(f'Checked {total_valid_words + total_invalid_words:,} words ({len(valid_guessed_words):,} unique valid, {len(invalid_guessed_words):,} unique invalid)')
            print(f'% valid (of total): {total_valid_words / (total_valid_words + total_invalid_words):.2%}')
            print(f'% valid (of unique): {len(valid_guessed_words) / (len(valid_guessed_words) + len(invalid_guessed_words)):.2%}')
            print(f'Duplicate valids checked: {duplicate_valid_words:,}')
            print(f'Duplicate invalids checked: {duplicate_invalid_words:,}')
            print('---')
            print(f'Matches: {", ".join(valid_guessed_words)}')

    def get_matches(self):
        word_matches = set()
        for mask in self.generate_masks():
            mask_pat = ''.join((e if e is not None else '.' for e in mask))
            print(mask_pat)
            matches = re.findall(mask_pat, WORDS_TXT, re.MULTILINE)
            word_matches |= set(matches)
        return word_matches

def main():
    # Enter words and results here:
    #   0 = gray   = not in word
    #   1 = yellow = wrong spot
    #   2 = green  = right spot
    game = Game([
        ('jacob', '00010'),
        ('thing', '00000'),
        ('lunes', '00120'),
        # ('games', '12000'),
    ])
    
    # Show all guesses & results
    print()
    print('Game')
    print('====')
    print(game)
    
    # Show possible "masks" given existing constraints
    print()
    print('Masks')
    print('=====')
    print(', '.join(game.generate_masks(pretty=True)))

    # Get actual matches using words.txt (cheating)
    print()
    print('Cheating')
    print('========')
    print(f'Matches: {", ".join(game.get_matches())}')

    # Use basic heuristics to fill masks based on surrounding consonants/vowels
    print()
    print('Heuristics')
    print('==========')
    list(game.generate_matches(verbose=True, time_limit=1))


if __name__ == '__main__':
    main()
