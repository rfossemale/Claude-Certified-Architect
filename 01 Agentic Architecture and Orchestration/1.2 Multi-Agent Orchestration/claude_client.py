"""
Llamada a Claude compartida por el coordinador y los subagentes.

Vive en su propio módulo para que `coordinator.py` pueda importar
`subagents.py` sin que los subagentes dependan del coordinador: en
hub-and-spoke los spokes no conocen al hub, solo reciben su tarea.
"""

import anthropic

MODEL = "claude-opus-5-5"

# Tope de reanudaciones cuando una tool del servidor pausa el turno.
MAX_CONTINUATIONS = 5


def call_claude(
    client: anthropic.Anthropic,
    system: str,
    prompt: str,
    schema: dict | None = None,
    tools: list[dict] | None = None,
) -> str:
    """Una llamada a Claude sin historial previo: devuelve el texto final.

    Cada llamada arranca de cero (solo `system` + `prompt`): un subagente solo
    sabe lo que va en su prompt.

    - `schema`: la respuesta es JSON válido según ese esquema.
    - `tools`: herramientas disponibles (p. ej. la búsqueda web del servidor).
    """
    extra = {}
    if schema is not None:
        extra["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    if tools is not None:
        extra["tools"] = tools

    messages = [{"role": "user", "content": prompt}]

    for _ in range(MAX_CONTINUATIONS + 1):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=system,
            messages=messages,
            # Si Claude rechaza la solicitud por política, la API reintenta sola con
            # un modelo de respaldo dentro de la misma llamada.
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
            **extra,
        )

        # Igual que en el loop agéntico: decidimos por stop_reason, no por el texto.
        if response.stop_reason == "end_turn":
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        if response.stop_reason == "pause_turn":
            # Las tools del servidor (web search) corren su propio loop en la API.
            # Si pausa, reenviamos la respuesta tal cual y la API continúa sola.
            messages = [messages[0], {"role": "assistant", "content": response.content}]
            continue

        raise RuntimeError(f"stop_reason inesperado: {response.stop_reason}")

    raise RuntimeError(f"El turno siguió pausado tras {MAX_CONTINUATIONS} reanudaciones.")
