
def safe_parse(x):
    import ast
    try:
        return ast.literal_eval(x)
    except:
        return x


def isNumeric(x):
    return isinstance(x, (int, float, complex)) and not isinstance(x, bool)