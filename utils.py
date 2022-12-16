def pluralize(n, word, plural_form=None):
    if plural_form is None:
        plural_form = f'{word}s'
    if n == 1:
        return f'{n} {word}'
    else:
        return f'{n} {plural_form}'
