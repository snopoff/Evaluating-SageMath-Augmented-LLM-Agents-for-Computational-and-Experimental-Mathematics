"""
Test SymPy comparison: different representations of the same mathematical answer.
"""

from sympy import (
    symbols,
    simplify,
    expand,
    Rational,
    pi,
    sqrt,
    floor,
    ceiling,
    factorial,
    binomial,
    fibonacci,
    Mod,
    Piecewise,
    Eq,
    Abs,
    I,
    E,
    oo,
    sin,
    cos,
    tan,
    log,
    exp,
    Sum,
    Product,
    Matrix,
    Tuple,
    FiniteSet,
    Gt,
    Lt,
    Interval,
)
from sympy.parsing.sympy_parser import parse_expr

n, m, k, x, y, z, alpha, beta, delta = symbols("n m k x y z alpha beta delta", real=True, positive=True)


def compare_example(label: str, expr1, expr2, vars_dict=None):
    """
    Compare two expressions using a smart multi-stage strategy.

    Args:
        label: Description of the test
        expr1: First expression (can be SymPy object or string)
        expr2: Second expression (can be SymPy object or string)
        vars_dict: Optional dict of variable substitutions for numerical check
    """
    # Local dict for parse_expr
    local_dict = {
        "n": n,
        "m": m,
        "k": k,
        "x": x,
        "y": y,
        "z": z,
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "pi": pi,
        "sqrt": sqrt,
        "floor": floor,
        "ceiling": ceiling,
        "factorial": factorial,
        "binomial": binomial,
        "fibonacci": fibonacci,
        "Rational": Rational,
        "Mod": Mod,
        "Piecewise": Piecewise,
        "Eq": Eq,
        "sin": sin,
        "cos": cos,
        "tan": tan,
        "log": log,
        "exp": exp,
        "Abs": Abs,
        "I": I,
        "E": E,
        "oo": oo,
        "Sum": Sum,
        "Product": Product,
        "Matrix": Matrix,
        "Tuple": Tuple,
        "FiniteSet": FiniteSet,
        "Gt": Gt,
        "Lt": Lt,
        "Interval": Interval,
    }

    # Parse if strings
    if isinstance(expr1, str):
        expr1 = parse_expr(expr1, local_dict=local_dict)
    if isinstance(expr2, str):
        expr2 = parse_expr(expr2, local_dict=local_dict)

    print(f"\n{'=' * 80}")
    print(f"TEST: {label}")
    print(f"{'=' * 80}")
    print(f"  expr1: {expr1}")
    print(f"  expr2: {expr2}")

    # STAGE 1: Simplify difference (primary method)
    try:
        diff = simplify(expr1 - expr2)
        stage1_match = diff == 0
        stage1_success = True
        stage1_result = f"simplify(expr1 - expr2) = {diff}"
    except Exception as e:
        stage1_match = False
        stage1_success = False
        stage1_result = f"ERROR: {type(e).__name__}: {e}"

    print("\n  [Stage 1] Simplify difference (PRIMARY):")
    print(f"    {stage1_result}")
    print(f"    MATCH: {stage1_match} {'✓' if stage1_match else '✗'}")

    # Early exit if stage 1 succeeds
    if stage1_match:
        print("\n  ✓ PASS (Stage 1)")
        return

    # STAGE 2: Expand canonical form (for algebraic expressions)
    try:
        expanded1 = expand(expr1)
        expanded2 = expand(expr2)
        stage2_match = expanded1 == expanded2
        stage2_success = True
        if not stage2_match:
            stage2_result = "expand(expr1) ≠ expand(expr2)"
        else:
            stage2_result = f"expand(expr1) = expand(expr2) = {expanded1}"
    except Exception as e:
        stage2_match = False
        stage2_success = False
        stage2_result = f"ERROR: {type(e).__name__}"

    print("\n  [Stage 2] Expand canonical form (ALGEBRAIC):")
    print(f"    {stage2_result}")
    print(f"    MATCH: {stage2_match} {'✓' if stage2_match else '✗'}")

    # Early exit if stage 2 succeeds
    if stage2_match:
        print("\n  ✓ PASS (Stage 2)")
        return

    # STAGE 3: Numerical substitution (if vars_dict provided)
    stage3_match = False
    stage3_success = False
    if vars_dict and vars_dict:
        try:
            val1 = complex(expr1.subs(vars_dict))
            val2 = complex(expr2.subs(vars_dict))
            # Use relative tolerance for floating point
            tolerance = max(1e-9 * max(abs(val1), abs(val2), 1), 1e-12)
            stage3_match = abs(val1 - val2) < tolerance
            stage3_success = True
            stage3_result = f"expr1 = {val1}, expr2 = {val2}, diff = {abs(val1 - val2):.2e}"
        except (TypeError, ValueError, AttributeError) as e:
            stage3_match = False
            stage3_success = False
            stage3_result = f"SKIP: {type(e).__name__}"
    else:
        stage3_success = False
        stage3_result = "SKIP: no substitution dict"

    print("\n  [Stage 3] Numerical substitution (FALLBACK):")
    print(f"    subs {vars_dict}")
    print(f"    {stage3_result}")
    print(f"    MATCH: {stage3_match} {'✓' if stage3_match else '✗'}")

    # Final result
    overall = stage1_match or stage2_match or stage3_match
    print(f"\n  {'✓ PASS' if overall else '✗ FAIL'}")


# ============================================================================
# TEST CASES
# ============================================================================

# 1. Simple arithmetic: different forms of same expression
compare_example("Simple: 2*n + 2*n vs 4*n", "2*n + 2*n", "4*n", {n: 5})

# 2. Rational: fraction vs decimal representation
compare_example("Rational: Rational(1, 2) vs 0.5", Rational(1, 2), 0.5, {})

# 3. Number vs expression without variables
compare_example("Number: 5 vs 2 + 3", 5, "2 + 3", {})

# 4. Expanded vs factored form
compare_example("Algebra: (n + 1)**2 vs n**2 + 2*n + 1", "(n + 1)**2", "n**2 + 2*n + 1", {n: 10})

# 5. Rational expressions
compare_example("Rational expr: (2*n + 4) / 2 vs n + 2", "(2*n + 4) / 2", "n + 2", {n: 7})

# 6. With pi constant
compare_example(
    "Constant: pi * (beta - alpha)**2 vs pi*(beta - alpha)**2", "pi * (beta - alpha)**2", "pi*(beta - alpha)**2", {alpha: 1, beta: 3}
)

# 7. Square root simplification
compare_example("Sqrt: sqrt(4*n) vs 2*sqrt(n)", "sqrt(4*n)", "2*sqrt(n)", {n: 9})

# 8. Floor function
compare_example("Floor: floor(2**(n - 1) * 5 / 3) vs floor(2**(n-1)*5/3)", "floor(2**(n - 1) * 5 / 3)", "floor(2**(n-1)*5/3)", {n: 5})

# 9. Fibonacci sequence
compare_example("Fibonacci: fibonacci(n + 1) vs fibonacci(n+1)", "fibonacci(n + 1)", "fibonacci(n+1)", {n: 6})

# 10. Binomial coefficient
compare_example("Binomial: binomial(n, k) vs binomial(n,k)", "binomial(n, k)", "binomial(n,k)", {n: 10, k: 3})

# 11. Factorial
compare_example("Factorial: factorial(k) / k**k vs factorial(k)/k**k", "factorial(k) / k**k", "factorial(k)/k**k", {k: 5})

# 12. Ceiling function
compare_example("Ceiling: ceiling((n + 1) / 2) vs -floor((-n - 1) / 2)", "ceiling((n + 1) / 2)", "-floor((-n - 1) / 2)", {n: 10})

# 13. Complex expression
compare_example(
    "Complex: 4 * (pi**2 - 3) / (5 * pi**2) vs 4*(pi**2 - 3)/(5*pi**2)", "4 * (pi**2 - 3) / (5 * pi**2)", "4*(pi**2 - 3)/(5*pi**2)", {}
)

# 14. Expression with multiple operations
compare_example("Multi-op: (2*m + 1)*k - m vs 2*m*k + k - m", "(2*m + 1)*k - m", "2*m*k + k - m", {m: 3, k: 4})

# 15. Difference of powers
compare_example("Powers: (n + 1)**2 - n**2 vs 2*n + 1", "(n + 1)**2 - n**2", "2*n + 1", {n: 100})

# 16. Fractional exponent
compare_example("Frac exp: (n)**0.5 vs sqrt(n)", "n**0.5", "sqrt(n)", {n: 16})

# 17. Named sequence difference
compare_example("Sequence: Depends on context (symbolic only)", "M_n - M_n_minus_2", "M_n - M_n_minus_2", {})

# 18. Negative number
compare_example("Negative: -7 vs -(7)", "-7", "-(7)", {})

# 19. Modular arithmetic (symbolic)
compare_example("Mod expr: Mod(n, 2) vs Mod(n,2)", "Mod(n, 2)", "Mod(n,2)", {n: 5})

# 20. Zero in different forms
compare_example("Zero: 0 vs n - n", "0", "n - n", {n: 42})

# 21. Piecewise: same conditions, same expressions
compare_example(
    "Piecewise 1: Same structure",
    "Piecewise((4, Eq(Mod(m, 2), 1)), (6, Eq(Mod(m, 2), 0)))",
    "Piecewise((4, Eq(Mod(m, 2), 1)), (6, Eq(Mod(m, 2), 0)))",
    {m: 5},
)

# 22. Piecewise: different order (should still match with simplify)
compare_example(
    "Piecewise 2: Same expressions, different order",
    "Piecewise((6, Eq(Mod(m, 2), 0)), (4, Eq(Mod(m, 2), 1)))",
    "Piecewise((4, Eq(Mod(m, 2), 1)), (6, Eq(Mod(m, 2), 0)))",
    {m: 3},
)

# 23. Piecewise: expressions with variables
compare_example(
    "Piecewise 3: Variable expressions",
    "Piecewise((4*n, Eq(Mod(n, 2), 1)), (2*n, Eq(Mod(n, 2), 0)))",
    "Piecewise((4*n, Eq(Mod(n, 2), 1)), (2*n, Eq(Mod(n, 2), 0)))",
    {n: 7},
)

# 24. Piecewise vs simple expression (when condition always true)
compare_example("Piecewise 4: Single condition (always true)", "Piecewise((2*n + 1, True))", "2*n + 1", {n: 10})

# 25. Multiple Piecewise (3 cases)
compare_example(
    "Piecewise 5: Three cases",
    "Piecewise((4, Eq(n, 1)), (6, Eq(n, 2)), (10, True))",
    "Piecewise((4, Eq(n, 1)), (6, Eq(n, 2)), (10, True))",
    {n: 3},
)

# 26. Tuple comparison
compare_example("Tuple: Tuple(1, 2, 3) vs Tuple(1,2,3)", "Tuple(1, 2, 3)", "Tuple(1,2,3)", {})

# 27. Tuple with expressions
compare_example("Tuple with expr: Tuple(2*n, 3*n) vs Tuple(2*n, 3*n)", "Tuple(2*n, 3*n)", "Tuple(2*n, 3*n)", {n: 5})

# 28. FiniteSet
compare_example("FiniteSet: FiniteSet(1, 2, 3) vs FiniteSet(1,2,3)", "FiniteSet(1, 2, 3)", "FiniteSet(1,2,3)", {})

# 29. FiniteSet (order independent)
compare_example("FiniteSet order: FiniteSet(3, 1, 2) vs FiniteSet(1,2,3)", "FiniteSet(3, 1, 2)", "FiniteSet(1,2,3)", {})

# 30. Trigonometric: sin(pi/2) vs 1
compare_example("Trig 1: sin(pi/2) vs 1", "sin(pi/2)", "1", {})

# 31. Trigonometric: cos(0) vs 1
compare_example("Trig 2: cos(0) vs 1", "cos(0)", "1", {})

# 32. Pythagorean identity
compare_example("Trig 3: sin(x)**2 + cos(x)**2 vs 1", "sin(x)**2 + cos(x)**2", "1", {x: 0.7})

# 33. Logarithm expansion
compare_example("Log 1: log(2*x) vs log(2) + log(x)", "log(2*x)", "log(2) + log(x)", {x: 3})

# 34. Exponential rule
compare_example("Exp 1: exp(x + y) vs exp(x)*exp(y)", "exp(x + y)", "exp(x)*exp(y)", {x: 0.5, y: 0.3})

# 35. Absolute value equality
compare_example("Abs 1: Abs(n) vs Abs(-n)", "Abs(n)", "Abs(-n)", {n: 5})

# 36. Absolute vs sqrt (domain-dependent!)
compare_example("Abs 2: Abs(n) vs sqrt(n**2) (when n > 0)", "Abs(n)", "sqrt(n**2)", {n: 4})

# 37. Commutative: addition
compare_example("Commutative 1: m + n vs n + m", "m + n", "n + m", {m: 3, n: 7})

# 38. Commutative: multiplication
compare_example("Commutative 2: k*m vs m*k", "k*m", "m*k", {k: 4, m: 5})

# 39. Nested functions: sqrt(n**2)
compare_example("Nested 1: sqrt(n**2) vs Abs(n)", "sqrt(n**2)", "Abs(n)", {n: 6})

# 40. Negative exponent
compare_example("Exponent 1: n**(-1) vs 1/n", "n**(-1)", "1/n", {n: 5})

# 41. Fractional exponent
compare_example("Exponent 2: m**(1/2) vs sqrt(m)", "m**(1/2)", "sqrt(m)", {m: 9})

# 42. Negative fractional exponent
compare_example("Exponent 3: n**(-1/2) vs 1/sqrt(n)", "n**(-1/2)", "1/sqrt(n)", {n: 4})

# 43. Symbolic constant: E vs exp(1)
compare_example("Constant 1: E vs exp(1)", "E", "exp(1)", {})

# 44. Complex number
compare_example("Complex 1: I**2 vs -1", "I**2", "-1", {})

# 45. Matrix comparison
compare_example("Matrix 1: Matrix([[1, 2], [3, 4]]) vs Matrix([[1,2],[3,4]])", "Matrix([[1, 2], [3, 4]])", "Matrix([[1,2],[3,4]])", {})

print("\n" + "=" * 80)
print("ALL TESTS COMPLETE (45 tests total)")
print("=" * 80)
