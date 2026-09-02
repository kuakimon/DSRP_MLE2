"""
App de Streamlit con un agente (Claude + tool use) que responde preguntas
apoyándose en las tools expuestas por el servidor MCP (mcp_server/server.py).

Requiere:
    pip install streamlit anthropic "mcp[cli]"
    export ANTHROPIC_API_KEY=sk-...

Ejecutar:
    streamlit run streamlit_app/app.py
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path

import streamlit as st
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.append(str(Path(__file__).resolve().parent))
import metrics  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = ROOT_DIR / "mcp_server" / "server.py"

MODEL_NAME = "claude-sonnet-4-6"
SYSTEM_PROMPT = (
    "Eres un asistente experto en el dataset y modelo de clasificación de "
    "fake news del proyecto. Usa las tools disponibles para responder con "
    "datos reales del dataset o del modelo en vez de inventar información. "
    "Responde en español, de forma clara y concisa."
)


def _unwrap_exception(exc: BaseException) -> BaseException:
    """anyio/asyncio envuelven errores dentro de ExceptionGroup ('unhandled
    errors in a TaskGroup'). Esta función baja hasta la excepción real para
    poder mostrar un mensaje útil en vez de un envoltorio genérico.
    """
    seen = exc
    while hasattr(seen, "exceptions") and seen.exceptions:  # ExceptionGroup / BaseExceptionGroup
        seen = seen.exceptions[0]
    return seen


def mcp_tools_to_anthropic_format(mcp_tools) -> list:
    """Convierte la lista de tools del MCP al formato que espera la API de Anthropic."""
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema,
        }
        for tool in mcp_tools
    ]


async def run_agent_turn(user_message: str, history: list) -> dict:
    """Abre una sesión MCP, corre el loop de tool-use con Claude y devuelve
    la respuesta final junto con datos para las métricas online (tools
    usadas y errores de tool-calls).
    """
    client = Anthropic()  # usa ANTHROPIC_API_KEY del entorno
    server_params = StdioServerParameters(command=sys.executable, args=[str(SERVER_SCRIPT)])

    tools_used = []
    tool_errors = 0

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            anthropic_tools = mcp_tools_to_anthropic_format(mcp_tools)

            messages = history + [{"role": "user", "content": user_message}]

            while True:
                response = client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=anthropic_tools,
                    messages=messages,
                )

                # Si Claude no pide usar ninguna tool, esa es la respuesta final.
                if response.stop_reason != "tool_use":
                    final_text = "".join(
                        block.text for block in response.content if block.type == "text"
                    )
                    return {
                        "answer": final_text,
                        "tools_used": tools_used,
                        "tool_errors": tool_errors,
                    }

                # Si pide tools, las ejecutamos vía MCP y devolvemos los resultados.
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tools_used.append(block.name)
                        try:
                            result = await session.call_tool(block.name, block.input)
                            result_text = "".join(
                                c.text for c in result.content if hasattr(c, "text")
                            )
                        except Exception as exc:  # noqa: BLE001
                            tool_errors += 1
                            result_text = f"Error al ejecutar la tool: {exc}"
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_text,
                            }
                        )
                messages.append({"role": "user", "content": tool_results})


def render_sidebar_metrics():
    summary = metrics.load_summary()
    with st.sidebar:
        st.header("📊 Métricas online")
        if not summary:
            st.caption("Aún no hay interacciones registradas.")
            return
        st.metric("Interacciones", summary["total_interactions"])
        st.metric("Latencia promedio (s)", summary["avg_latency_seconds"])
        st.metric("Tool-calls promedio / turno", summary["avg_tool_calls_per_turn"])
        st.metric("Tasa de éxito", f"{summary['success_rate'] * 100:.1f}%")
        st.metric("Tasa de error en tools", f"{summary['tool_error_rate'] * 100:.1f}%")
        st.metric("Feedback 👍 / 👎", f"{summary['feedback_up']} / {summary['feedback_down']}")


def main():
    st.set_page_config(page_title="Agente Fake News (MCP)", page_icon="📰")
    st.title("📰 Agente de consultas — Fake News Dataset")
    st.caption(
        "Agente con Claude + servidor MCP propio, apoyado en el dataset y "
        "modelo de clasificación de noticias falsas/reales."
    )

    render_sidebar_metrics()

    if "messages" not in st.session_state:
        st.session_state.messages = []  # historial mostrado en pantalla
        st.session_state.agent_history = []  # historial para la API (bloques crudos)
        st.session_state.feedback_given = set()  # índices de mensajes ya calificados

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(
        "Ej: ¿Cuántos artículos fake hay sobre política? / Clasifica este titular..."
    )

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Consultando al agente y al servidor MCP..."):
                start = time.time()
                try:
                    result = asyncio.run(
                        run_agent_turn(user_input, st.session_state.agent_history)
                    )
                    success = True
                except Exception as exc:  # noqa: BLE001
                    # Imprime el traceback completo en la terminal donde corre
                    # streamlit (ahí sí verás la causa real del error).
                    traceback.print_exc()
                    real_cause = _unwrap_exception(exc)
                    result = {
                        "answer": (
                            f"Ocurrió un error: **{type(real_cause).__name__}**: {real_cause}\n\n"
                            "Revisa la terminal donde corriste `streamlit run` para ver el "
                            "traceback completo."
                        ),
                        "tools_used": [],
                        "tool_errors": 1,
                    }
                    success = False
                latency = time.time() - start
                st.markdown(result["answer"])

                metrics.log_interaction(
                    latency_seconds=latency,
                    tools_used=result["tools_used"],
                    tool_errors=result["tool_errors"],
                    success=success,
                )

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        st.session_state.agent_history.append({"role": "user", "content": user_input})
        st.session_state.agent_history.append({"role": "assistant", "content": result["answer"]})

    # Feedback sobre la última respuesta del agente
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_idx = len(st.session_state.messages) - 1
        if last_idx not in st.session_state.feedback_given:
            col1, col2, _ = st.columns([1, 1, 8])
            if col1.button("👍", key=f"up_{last_idx}"):
                metrics.update_last_feedback("up")
                st.session_state.feedback_given.add(last_idx)
                st.rerun()
            if col2.button("👎", key=f"down_{last_idx}"):
                metrics.update_last_feedback("down")
                st.session_state.feedback_given.add(last_idx)
                st.rerun()


if __name__ == "__main__":
    main()
