"""
Paso 1 - Definición de herramientas (tools) para el cliente de Claude.

Cada tool tiene dos partes:
  1) El SCHEMA (lo que Claude ve): name, description e input_schema en JSON Schema.
     Claude usa esto para decidir CUÁNDO y CON QUÉ argumentos llamar la tool.
  2) La FUNCIÓN (lo que tu código ejecuta): recibe los argumentos y devuelve un resultado.
"""

# --------------------------------------------------------------------------- #
# 1) Definiciones que se registran con Claude (input_schema en JSON Schema)
# --------------------------------------------------------------------------- #

CALCULATOR_TOOL = {
    "name": "calculator",
    "description": (
        "Evalúa una expresión aritmética y devuelve el resultado numérico. "
        "Úsala siempre que necesites hacer cálculos exactos en lugar de estimarlos."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Expresión matemática a evaluar, por ejemplo '2 * (3 + 4)'.",
            }
        },
        "required": ["expression"],
    },
}

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Busca información en la web y devuelve resultados relevantes. "
        "Úsala cuando necesites datos que no conoces o que podrían haber cambiado."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Términos de búsqueda, por ejemplo 'población de Argentina 2024'.",
            }
        },
        "required": ["query"],
    },
}

# Lista que le pasaremos a Claude en el parámetro `tools`.
TOOLS = [CALCULATOR_TOOL, WEB_SEARCH_TOOL]


# --------------------------------------------------------------------------- #
# 2) Implementaciones reales que ejecuta nuestro código
# --------------------------------------------------------------------------- #

def calculator(expression: str) -> str:
    """Evalúa una expresión aritmética simple de forma segura."""
    # Solo permitimos caracteres aritméticos para evitar ejecución de código arbitrario.
    allowed = set("0123456789+-*/(). %")
    if not set(expression) <= allowed:
        return "Error: la expresión contiene caracteres no permitidos."
    try:
        # eval con namespace vacío: sin builtins ni variables disponibles.
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as exc:  # noqa: BLE001 - devolvemos el error como texto para Claude
        return f"Error al evaluar la expresión: {exc}"
    return str(result)


def web_search(query: str) -> str:
    """Stub de búsqueda web: devuelve resultados simulados (mock)."""
    mock_db = {
        "torre eiffel altura": "La Torre Eiffel mide 330 metros de altura.",
        "velocidad de la luz": "La velocidad de la luz es 299792458 metros por segundo.",
        "poblacion argentina": "La población de Argentina es aproximadamente 46000000 habitantes.",
    }
    key = query.lower().strip()
    for known_key, value in mock_db.items():
        if known_key in key or key in known_key:
            return value
    return f"Resultado simulado para '{query}': no se encontró un dato específico en el mock."


# Mapa nombre-de-tool -> función. Lo usaremos en el loop para despachar la llamada.
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "web_search": web_search,
}
