import pytest

from wundy.ui.eqparse import EquationSyntaxError
from wundy.ui.eqparse import parse_equation_expression


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("u[1,0]", [(1, 0, 1.0)]),
        ("-u[2,1]", [(2, 1, -1.0)]),
        ("2*u[1,0]-2*u[4,1]+u[10,0]", [(1, 0, 2.0), (4, 1, -2.0), (10, 0, 1.0)]),
        ("5 * u[4, 1] + 6*u[  56, 2]-u[65  , 1]", [(4, 1, 5.0), (56, 2, 6.0), (65, 1, -1.0)]),
    ],
)
def test_equations(expr, expected):
    equation = parse_equation_expression(expr)
    assert equation == expected


@pytest.mark.parametrize("expr", ["u[1,1]*u[2,1]", "5+u[2,3]", "u[1,2] 2 u[3,2]"])
def test_bad(expr):
    with pytest.raises(EquationSyntaxError):
        parse_equation_expression(expr)
