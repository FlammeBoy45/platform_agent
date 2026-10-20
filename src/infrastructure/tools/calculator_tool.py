import ast
import operator
import re
from collections.abc import Callable
from typing import Any


class CalculatorTool:
    name = "calculator"
    description = (
        "Evalue une expression mathematique simple et sure. "
        "Operations supportees: +, -, *, /, //, %, ** et parentheses. "
        "Comprend aussi: '15% de 340', 'moitié de 100', 'racine carrée de 16'."
    )
    args_schema = {
        "expression": "Expression mathematique, exemple: 2 + 2 * 3 ou 15% de 340.",
    }
    return_direct = True
    trigger_words: tuple[str, ...] = (
        "calcule",
        "calcul",
        "combien font",
        "combien fait",
        "+",
        "-",
        "*",
        "/",
        "=",
    )

    _binary_operators: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary_operators: dict[type[ast.unaryop], Callable[[float], float]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def infer_args(self, user_input: str, args: dict[str, Any]) -> dict[str, Any]:
        """Tente d'extraire l'expression mathématique depuis l'entrée utilisateur."""
        if args.get("expression"):
            return args

        expression = user_input.lower()
        prefixes = (
            "calcule",
            "calcul",
            "combien font",
            "combien fait",
        )
        for prefix in prefixes:
            if expression.startswith(prefix):
                expression = expression[len(prefix) :]
                break

        expression = expression.replace("?", "").replace("=", "")
        return {**args, "expression": expression.strip()}

    def _preprocess_expression(self, expression: str) -> str:
        """Convertit le langage naturel en expression mathématique."""
        expression = expression.replace(",", ".")

        # Nettoyer les mots parasites
        for word in ["calcule", "calcul", "combien fait", "combien font", "?", "="]:
            expression = expression.lower().replace(word, "").strip()

        # "15% de 340" → "340 * 0.15"
        percent_match = re.search(
            r"(\d+(?:\.\d+)?)\s*%\s*(?:de|d\'|du|sur)\s*(\d+(?:\.\d+)?)",
            expression.lower(),
        )
        if percent_match:
            value = float(percent_match.group(1))
            total = float(percent_match.group(2))
            return f"{total} * {value / 100}"

        # "15 pourcent de 340" → "340 * 0.15"
        pourcent_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:pourcent|pour cent)\s*"
            r"(?:de|d\'|du|sur)\s*(\d+(?:\.\d+)?)",
            expression.lower(),
        )
        if pourcent_match:
            value = float(pourcent_match.group(1))
            total = float(pourcent_match.group(2))
            return f"{total} * {value / 100}"

        # "moitié de 100" → "100 * 0.5"
        if "moitié" in expression.lower() or "moitie" in expression.lower():
            total_match = re.search(r"(\d+(?:\.\d+)?)", expression)
            if total_match:
                return f"{total_match.group(1)} * 0.5"

        # "racine carrée de 16" → "16 ** 0.5"
        if "racine" in expression.lower():
            num_match = re.search(r"(\d+(?:\.\d+)?)", expression)
            if num_match:
                return f"{num_match.group(1)} ** 0.5"

        return expression

    async def execute(self, **kwargs: Any) -> str:
        expression = str(kwargs.get("expression", "")).strip()
        if not expression:
            return "Expression manquante."

        # Remplacer virgule française par point
        expression = expression.replace(",", ".")

        # Pré-traiter le langage naturel
        expression = self._preprocess_expression(expression)

        try:
            tree = ast.parse(expression, mode="eval")
            result = self._evaluate(tree.body)
        except Exception as exc:
            return f"Calcul impossible: {exc}"

        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)

    def _evaluate(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)

        if isinstance(node, ast.BinOp):
            binary_operator_fn = self._binary_operators.get(type(node.op))
            if not binary_operator_fn:
                raise ValueError("operateur non supporte")

            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("puissance trop grande")
            return float(binary_operator_fn(left, right))

        if isinstance(node, ast.UnaryOp):
            unary_operator_fn = self._unary_operators.get(type(node.op))
            if not unary_operator_fn:
                raise ValueError("operateur unaire non supporte")
            return float(unary_operator_fn(self._evaluate(node.operand)))

        raise ValueError("expression non supportee")
