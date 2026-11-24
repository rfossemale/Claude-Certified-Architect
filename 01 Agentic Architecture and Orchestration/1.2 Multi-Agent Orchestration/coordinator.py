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

Paso 3 - Delegación a los subagentes (ver subagents.py).

`delegate()` es el hub en acción: por cada subtema llama al web agent, toma
sus hallazgos y se los PASA al agente de análisis. Los spokes nunca se hablan
entre sí; todo pasa por acá. Los subtemas son independientes, así que se
delegan en paralelo.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import anthropic

from claude_client import call_claude
from subagents import document_analysis_agent, web_search_agent

# Objetivo que el coordinador transmite a cada subagente. Sin esto, un subagente
# aislado no sabe para qué sirve su parte ni con qué profundidad investigar.
DEFAULT_GOAL = (
    "Producir un reporte comparativo que permita entender el panorama completo "
    "del tema: cada alternativa, su funcionamiento, madurez, ventajas y limitaciones."
)

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


def research_subtopic(
    client: anthropic.Anthropic, topic: str, goal: str, subtopic: dict
) -> dict:
    """Cadena web -> análisis para UN subtema. El hub hace de puente entre ambos."""
    print(f"[delegar] web_search_agent -> {subtopic['name']}")
    findings = web_search_agent(client, topic, goal, subtopic)

    # Los spokes no se comunican: el coordinador toma la salida del primero y
    # la incluye explícitamente en el prompt del segundo.
    print(f"[delegar] document_analysis_agent -> {subtopic['name']}")
    section = document_analysis_agent(client, topic, goal, subtopic, findings)

    return {"findings": findings, "section": section}


def delegate(
    client: anthropic.Anthropic, topic: str, goal: str, subtopics: list[dict]
) -> dict[str, dict]:
    """Delega todos los subtemas en paralelo: {nombre: {"findings", "section"}}."""
    with ThreadPoolExecutor(max_workers=len(subtopics)) as pool:
        results = pool.map(
            lambda sub: research_subtopic(client, topic, goal, sub), subtopics
        )
        return {sub["name"]: result for sub, result in zip(subtopics, results)}


def run_coordinator(topic: str, goal: str = DEFAULT_GOAL) -> ResearchReport:
    """Punto de entrada: recibe un tema amplio y devuelve un reporte estructurado."""
    # Sin api_key explícita, el SDK la toma de ANTHROPIC_API_KEY.
    client = anthropic.Anthropic()

    # Paso 2: el hub decide el alcance de la investigación.
    subtopics = decompose(client, topic)
    for sub in subtopics:
        print(f"    - {sub['name']}: {sub['focus']}")

    # Paso 3: el hub delega cada subtema a los spokes con contexto explícito.
    # En los pasos siguientes se agrega:
    #   coverage  = evaluate(results, ...)      -> Paso 4
    #   loop de refinamiento sobre los huecos   -> Paso 5
    results = delegate(client, topic, goal, subtopics)

    return ResearchReport(
        topic=topic,
        subtopics=[sub["name"] for sub in subtopics],
        sections={name: result["section"] for name, result in results.items()},
        iterations=1,
    )


if __name__ == "__main__":
    report = run_coordinator("renewable energy technologies")
    print(report.to_markdown())
