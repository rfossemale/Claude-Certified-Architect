"""
Paso 2 - El loop agéntico.

Idea central: NO adivinamos qué quiere hacer Claude leyendo su texto.
Miramos `response.stop_reason`, que es un campo DETERMINÍSTICO que la API
devuelve en cada respuesta. Sus valores relevantes para nosotros:

  - "tool_use" : Claude pidió ejecutar una o más herramientas. (Paso 3)
  - "end_turn" : Claude terminó y tiene una respuesta final. (Paso 4)

En este paso armamos el ESQUELETO del loop: crear el cliente, mandar la
conversación y ramificar según stop_reason. Las ramas se completan en los
pasos siguientes.
"""

import os

import logging

import anthropic

from tools import TOOLS, TOOL_FUNCTIONS

MODEL = "claude-sonnet-4-20250514"

# Tope de seguridad: NO es el mecanismo de parada (eso lo hace end_turn).
# Es solo un fallback para evitar un loop infinito ante un comportamiento anómalo.
MAX_ITERATIONS = 20


def run_agent(user_prompt: str) -> str:
    # La API key se lee de la variable de entorno ANTHROPIC_API_KEY.
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # `messages` es el historial completo de la conversación. Va creciendo
    # a medida que agregamos respuestas del asistente y resultados de tools.
    messages = [
        {"role": "user", "content": user_prompt},
    ]

    iteration = 0
    while iteration < MAX_ITERATIONS:
        iteration += 1
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        print(f"[iteracion {iteration}] stop_reason = {response.stop_reason}")

        # El punto clave del examen: decidimos por stop_reason, no por el texto.
        if response.stop_reason == "tool_use":
            # 1) Guardamos la respuesta del asistente TAL CUAL en el historial.
            #    Contiene los bloques tool_use que Claude quiere ejecutar.
            messages.append({"role": "assistant", "content": response.content})

            # 2) Recorremos los bloques de la respuesta y ejecutamos cada tool_use.
            #    Una misma respuesta puede pedir varias tools a la vez.
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_fn = TOOL_FUNCTIONS[block.name]
                # los signos de ** se usan para desempaquetar el diccionario de entrada como argumentos de la función.
                output = tool_fn(**block.input)
                print(f"    -> tool {block.name}({block.input}) = {output}")

                # 3) Cada resultado se referencia con el tool_use_id correspondiente.
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    }
                )

            # 4) Los resultados vuelven a Claude como un mensaje de rol "user".
            messages.append({"role": "user", "content": tool_results})

            # El loop vuelve a empezar: Claude verá el resultado y seguirá razonando.
            continue

        elif response.stop_reason == "end_turn":
            # Claude terminó: juntamos el texto de los bloques de tipo "text"
            # y salimos del loop devolviendo la respuesta final.
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return final_text

        else:
            # Otros stop_reason posibles: "max_tokens", "stop_sequence", etc.
            raise RuntimeError(f"stop_reason inesperado: {response.stop_reason}")

    # Si llegamos aquí, el fallback se disparó: en operación normal no debería pasar.
    logging.warning(
        "Se alcanzó el tope de seguridad de %d iteraciones sin end_turn.",
        MAX_ITERATIONS,
    )
    return "El agente alcanzó el límite máximo de iteraciones sin completar la tarea."


if __name__ == "__main__":
    # Prompt secuencial: primero BUSCAR la altura, luego CALCULAR con ese valor.
    prompt = "Busca la altura de la Torre Eiffel y multiplícala por 3."
    print(run_agent(prompt))
