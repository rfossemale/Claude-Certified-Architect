"""
Paso 3 - Subagentes (spokes) con paso EXPLÍCITO de contexto.

Aislamiento de subagentes: cada subagente es una llamada nueva a Claude, sin
memoria compartida y sin heredar la conversación del coordinador. Lo ÚNICO que
sabe es lo que viene en su prompt. Por eso cada función recibe todo por
parámetro y lo escribe en el prompt:
  - el tema general y el objetivo de la investigación,
  - el subtema asignado y su foco,
  - el contexto de agentes anteriores (si lo hay).

Si un subagente devuelve algo pobre, el diagnóstico correcto es revisar QUÉ
contexto le pasó el coordinador, no el subagente en sí.

Los dos spokes:
  1) web_search_agent: busca en la web (tool de servidor de Anthropic).
  2) document_analysis_agent: analiza los hallazgos del primero y redacta la
     sección. No ve al web agent: el coordinador le copia los hallazgos en el prompt.
"""

import anthropic

from claude_client import call_claude

# Búsqueda web real ejecutada por la API de Anthropic (tool de servidor): no
# hay que implementarla ni ejecutarla nosotros, a diferencia del stub del 1.1.
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}

WEB_SEARCH_SYSTEM_PROMPT = """\
Eres un subagente de BÚSQUEDA WEB dentro de un sistema de investigación.
Recibes una tarea de un coordinador. Busca información actual y verificable
sobre el subtema asignado y devuelve hallazgos concretos: datos, cifras,
estado de madurez y fuentes. No escribas introducciones ni conclusiones.
"""

DOCUMENT_ANALYSIS_SYSTEM_PROMPT = """\
Eres un subagente de ANÁLISIS DE DOCUMENTOS dentro de un sistema de investigación.
Recibes de un coordinador un subtema y un conjunto de hallazgos ya recopilados.
Analiza SOLO ese material y redacta una sección de reporte clara y sustantiva.
Si el material no alcanza para algún aspecto pedido, dilo explícitamente.
"""


def _task_context(topic: str, goal: str, subtopic: dict) -> str:
    """Bloque de contexto común que el coordinador pasa a TODO subagente."""
    return (
        f"Tema general de la investigación: {topic}\n"
        f"Objetivo de la investigación: {goal}\n"
        f"Subtema asignado: {subtopic['name']}\n"
        f"Foco del subtema: {subtopic['focus']}\n"
    )


def web_search_agent(
    client: anthropic.Anthropic,
    topic: str,
    goal: str,
    subtopic: dict,
    prior_context: str = "",
) -> str:
    """Spoke 1: devuelve hallazgos de la web sobre el subtema."""
    prompt = _task_context(topic, goal, subtopic)
    if prior_context:
        prompt += f"\nContexto de investigaciones anteriores:\n{prior_context}\n"
    prompt += (
        "\nTarea: busca en la web y devuelve una lista de hallazgos concretos "
        "sobre el subtema asignado (principio de funcionamiento, madurez, "
        "capacidad o adopción actual, ventajas, limitaciones), cada uno con su fuente."
    )
    return call_claude(client, WEB_SEARCH_SYSTEM_PROMPT, prompt, tools=[WEB_SEARCH_TOOL])


def document_analysis_agent(
    client: anthropic.Anthropic,
    topic: str,
    goal: str,
    subtopic: dict,
    findings: str,
) -> str:
    """Spoke 2: analiza los hallazgos del web agent y redacta la sección."""
    prompt = (
        _task_context(topic, goal, subtopic)
        # Contexto de un agente anterior, copiado EXPLÍCITAMENTE por el coordinador.
        + f"\nHallazgos recopilados por el agente de búsqueda web:\n{findings}\n"
        + "\nTarea: redacta la sección del reporte para este subtema (máximo 250 "
        "palabras) cubriendo: cómo funciona, madurez actual, ventajas y "
        "limitaciones. Básate solo en los hallazgos."
    )
    return call_claude(client, DOCUMENT_ANALYSIS_SYSTEM_PROMPT, prompt)
