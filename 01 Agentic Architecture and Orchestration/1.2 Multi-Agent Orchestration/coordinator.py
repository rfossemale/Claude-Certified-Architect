"""
Paso 1 - El coordinador (hub).

Idea central de hub-and-spoke: TODA la comunicación pasa por el coordinador.
Los subagentes (spokes) nunca hablan entre sí ni con el usuario; solo reciben
una tarea del hub y le devuelven un resultado.

Responsabilidades que son SOLO del coordinador (no de los subagentes):
  - Descomponer el tema en subtemas.                      (Paso 2)
  - Elegir qué subagente hace cada cosa y con qué contexto. (Paso 3)
  - Agregar los resultados y evaluar la cobertura.          (Paso 4)
  - Re-delegar si hay huecos (refinamiento iterativo).      (Paso 5)

En este paso armamos el ESQUELETO: el system prompt que define el rol del hub,
la estructura del reporte y la función `run_coordinator(topic)`. Por ahora el
coordinador redacta el reporte directamente, en una sola llamada. Es una
línea de base que los pasos siguientes reemplazan por la delegación real.

Paso 2 - Descomposición AMPLIA del tema.

El error típico del examen es la "descomposición estrecha": para "energías
renovables" el coordinador asigna solo solar y eólica, y el reporte final no
menciona geotérmica, mareomotriz, biomasa ni fusión. Los subagentes hicieron
bien su trabajo; el hueco nació en el coordinador, que nunca les pidió eso.

Por eso `decompose()`:
  - Pide explícitamente cubrir TODO el alcance (establecido, emergente, experimental).
  - Usa structured outputs (JSON Schema) para recibir una lista parseable, no texto libre.
  - Valida en código que haya al menos MIN_SUBTOPICS; si no, re-pide diciendo por qué.
"""

import json
from dataclasses import dataclass, field

import anthropic

MODEL = "claude-opus-5-5"

# Mínimo de subtemas que exige el enunciado. Menos que esto = descomposición estrecha.
MIN_SUBTOPICS = 5

# Esquema de la respuesta de descomposición. Cada subtema trae un `focus`:
# ese texto viaja después al subagente como parte de su contexto (Paso 3).
DECOMPOSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "subtopics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "focus": {"type": "string"},
                },
                "required": ["name", "focus"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subtopics"],
    "additionalProperties": False,
}

# El system prompt define el ROL del hub. Nótese que describe orquestación,
# no investigación: el coordinador planifica, delega, integra y evalúa.
COORDINATOR_SYSTEM_PROMPT = """\
Eres el COORDINADOR central de un sistema de investigación multi-agente con
arquitectura hub-and-spoke.

Tu rol:
- Recibes un tema de investigación amplio.
- Lo descompones en subtemas que cubran TODO el alcance del tema, no solo los
  aspectos más obvios o populares.
- Delegas cada subtema a subagentes especializados (búsqueda web y análisis de
  documentos). Los subagentes NO comparten memoria contigo ni entre ellos: todo
  el contexto que necesiten debe ir explícito en su tarea.
- Integras los resultados en un reporte coherente y evalúas si la cobertura es
  completa. Si detectas huecos, re-delegas consultas específicas para cubrirlos.

Eres responsable de la calidad final: si el reporte queda incompleto, la causa
más probable es tu descomposición o el contexto que pasaste, no los subagentes.
"""


@dataclass
class ResearchReport:
    """Reporte estructurado que devuelve el coordinador."""

    topic: str
    # Subtemas en los que el coordinador descompuso el tema. (Paso 2)
    subtopics: list[str] = field(default_factory=list)
    # Una sección de contenido por subtema: {subtema: texto}.
    sections: dict[str, str] = field(default_factory=dict)
    # Evaluación de cobertura por subtema: {subtema: "completo"|"parcial"|"faltante"}. (Paso 4)
    coverage: dict[str, str] = field(default_factory=dict)
    # Cuántas rondas de delegación hicieron falta. (Paso 5)
    iterations: int = 0

    def to_markdown(self) -> str:
        lines = [f"# Reporte de investigación: {self.topic}", ""]
        if self.subtopics:
            lines += ["**Subtemas:** " + ", ".join(self.subtopics), ""]
        for title, body in self.sections.items():
            lines += [f"## {title}", "", body.strip(), ""]
        if self.coverage:
            lines += ["## Cobertura", ""]
            lines += [f"- {sub}: {status}" for sub, status in self.coverage.items()]
            lines.append("")
        lines.append(f"_Rondas de delegación: {self.iterations}_")
        return "\n".join(lines)


def call_claude(
    client: anthropic.Anthropic, system: str, prompt: str, schema: dict | None = None
) -> str:
    """Una llamada a Claude sin historial previo: devuelve el texto final.

    Cada llamada arranca de cero (solo `system` + `prompt`). Esto va a ser
    clave en el Paso 3: un subagente solo sabe lo que va en su prompt.

    Si se pasa `schema`, la respuesta es JSON válido según ese esquema.
    """
    extra = {}
    if schema is not None:
        extra["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        # Si Claude rechaza la solicitud por política, la API reintenta sola con
        # un modelo de respaldo dentro de la misma llamada.
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},
        **extra,
    )

    # Igual que en el loop agéntico: decidimos por stop_reason, no por el texto.
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason inesperado: {response.stop_reason}")

    return "".join(block.text for block in response.content if block.type == "text")


def decompose(client: anthropic.Anthropic, topic: str) -> list[dict]:
    """Descompone el tema en >= MIN_SUBTOPICS subtemas: [{"name", "focus"}, ...]."""
    prompt = (
        f"Tema de investigación: {topic}\n\n"
        f"Descompón el tema en al menos {MIN_SUBTOPICS} subtemas que, juntos, cubran "
        "TODO su alcance. Antes de responder, piensa qué categorías existen en el "
        "tema completo, no solo las más conocidas o populares.\n"
        "- Incluye lo establecido, lo emergente y lo experimental.\n"
        "- Los subtemas no deben solaparse: cada uno es una categoría distinta.\n"
        "- `name`: nombre corto del subtema.\n"
        "- `focus`: 1-2 oraciones sobre qué investigar en ese subtema "
        "(principio de funcionamiento, madurez, ventajas, limitaciones)."
    )

    # Un reintento como máximo: si la lista es corta, le decimos POR QUÉ la
    # rechazamos para que el segundo intento sea más amplio, no igual.
    for attempt in range(2):
        text = call_claude(client, COORDINATOR_SYSTEM_PROMPT, prompt, DECOMPOSITION_SCHEMA)
        subtopics = json.loads(text)["subtopics"]
        print(f"[descomposicion intento {attempt + 1}] {len(subtopics)} subtemas")
        if len(subtopics) >= MIN_SUBTOPICS:
            return subtopics

        names = ", ".join(s["name"] for s in subtopics)
        prompt += (
            f"\n\nTu propuesta anterior ({names}) es demasiado estrecha: tiene "
            f"{len(subtopics)} subtemas y se necesitan al menos {MIN_SUBTOPICS}. "
            "Identifica las categorías que faltan."
        )

    raise RuntimeError(f"La descomposición no alcanzó {MIN_SUBTOPICS} subtemas.")


def run_coordinator(topic: str) -> ResearchReport:
    """Punto de entrada: recibe un tema amplio y devuelve un reporte estructurado."""
    # Sin api_key explícita, el SDK la toma de ANTHROPIC_API_KEY.
    client = anthropic.Anthropic()

    # Paso 2: el hub decide el alcance de la investigación.
    subtopics = decompose(client, topic)
    for sub in subtopics:
        print(f"    - {sub['name']}: {sub['focus']}")

    # Todavía sin subagentes: el hub escribe un panorama general él mismo,
    # ahora guiado por los subtemas. En los pasos siguientes esto se reemplaza por:
    #   results   = delegate(subtopics, ...)    -> Paso 3
    #   coverage  = evaluate(results, ...)      -> Paso 4
    #   loop de refinamiento sobre los huecos   -> Paso 5
    subtopic_names = [sub["name"] for sub in subtopics]
    overview = call_claude(
        client,
        COORDINATOR_SYSTEM_PROMPT,
        f"Tema de investigación: {topic}\n"
        f"Subtemas: {', '.join(subtopic_names)}\n\n"
        "Por ahora no tienes subagentes disponibles. Escribe un panorama general "
        "breve del tema que mencione cada subtema (máximo 300 palabras).",
    )

    return ResearchReport(
        topic=topic,
        subtopics=subtopic_names,
        sections={"Panorama general": overview},
        iterations=1,
    )


if __name__ == "__main__":
    report = run_coordinator("renewable energy technologies")
    print(report.to_markdown())
