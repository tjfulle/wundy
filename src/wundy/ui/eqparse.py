import io
import tokenize


def token_matches(token: tokenize.TokenInfo, type: int, string: str) -> bool:
    return token.type == type and token.string == string


def parse_equation_expression(expr: str) -> list[tuple[int, int, float]]:
    """Equation must be of form"""
    equation: list[tuple[int, int, float]] = []

    fp = io.StringIO(expr.strip()).readline
    tokens = tokenize.generate_tokens(fp)

    while True:
        try:
            token = next(tokens)
        except StopIteration:
            break

        if token.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
            break

        coeff: float = 1.0
        if token.type == tokenize.OP:
            if token.string == "-":
                coeff *= -1.0
            elif token.string != "+":
                raise EquationSyntaxError(token, "+")
            token = next(tokens)
        elif equation:
            raise EquationSyntaxError(token, "[+-]")

        if token.type == tokenize.NUMBER:
            coeff *= float(token.string)
            token = next(tokens)
            if not token_matches(token, tokenize.OP, "*"):
                raise EquationSyntaxError(token, "*")
            token = next(tokens)

        if not token_matches(token, tokenize.NAME, "u"):
            raise EquationSyntaxError(token, "u")

        token = next(tokens)
        if not token_matches(token, tokenize.OP, "["):
            raise EquationSyntaxError(token, "[")

        token = next(tokens)
        if token.type != tokenize.NUMBER:
            raise EquationSyntaxError(token, "\d")
        node: int = int(token.string)

        token = next(tokens)
        if not token_matches(token, tokenize.OP, ","):
            raise EquationSyntaxError(token, ",")

        token = next(tokens)
        if token.type != tokenize.NUMBER:
            raise EquationSyntaxError(token, "\d")
        dof: int = int(token.string)

        token = next(tokens)
        if not token_matches(token, tokenize.OP, "]"):
            raise EquationSyntaxError(token, "]")

        equation.append((node, dof, coeff))

    return equation


class EquationSyntaxError(SyntaxError):
    def __init__(self, token: tokenize.TokenInfo, expected: str) -> None:
        msg = f"Error in equation:\n{token.line}\n"
        msg += " " * (token.start[0] - 1) + "^\n"
        msg += f"expected {expected}, got {token.string}\n"
        super().__init__(msg)
