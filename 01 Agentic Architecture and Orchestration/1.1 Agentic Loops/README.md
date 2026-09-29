# Agentic Loop con Claude

Implementación de un **loop agéntico** usando la API de Anthropic (Claude). El agente
decide de forma autónoma qué herramientas usar, ejecuta las llamadas y razona con los
resultados hasta completar la tarea.

La idea central: el control del flujo se basa en el campo **`stop_reason`** que devuelve
la API (determinístico), no en parsear el texto del modelo (frágil).

## Estructura

| Archivo | Descripción |
|---------|-------------|
| `tools.py` | Definiciones de las herramientas (`input_schema` en JSON Schema) y sus implementaciones. |
| `agent.py` | El loop agéntico: cliente, manejo de `stop_reason` y tope de seguridad. |
| `requirements.txt` | Dependencias. |

## Herramientas disponibles

- **`calculator`**: evalúa una expresión aritmética y devuelve el resultado.
- **`web_search`**: stub de búsqueda web que devuelve resultados simulados (mock).

## Cómo funciona el loop

```
messages.create ──> stop_reason?
                       │
       tool_use ───────┤──> ejecutar tool + append tool_result ──> (vuelve al inicio)
                       │
       end_turn ───────┴──> extraer texto y return
```

- **`tool_use`**: Claude pidió una o más herramientas. Se ejecutan, se agrega el
  `tool_result` al historial (con su `tool_use_id`) y el loop continúa.
- **`end_turn`**: Claude terminó. Se extrae el texto final y se devuelve.
- **`MAX_ITERATIONS`** (20): tope de seguridad como *fallback*, **no** es el mecanismo
  de parada. En operación normal nunca debería dispararse.

## Ejecución

Requiere Python 3.9+ y una API key de Anthropic.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="tu-key-aca"
python agent.py
```

### Salida esperada

El prompt de ejemplo (`"Busca la altura de la Torre Eiffel y multiplícala por 3."`)
fuerza dos llamadas a herramientas encadenadas:

```
[iteracion 1] stop_reason = tool_use
    -> tool web_search({'query': 'altura torre eiffel'}) = La Torre Eiffel mide 330 metros de altura.
[iteracion 2] stop_reason = tool_use
    -> tool calculator({'expression': '330 * 3'}) = 990
[iteracion 3] stop_reason = end_turn
La Torre Eiffel mide 330 metros, multiplicada por 3 son 990 metros.
```

Nótese el patrón: dos iteraciones `tool_use` antes de `end_turn`, demostrando que el
agente mantiene el estado y razona con el resultado anterior.
