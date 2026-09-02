"""
Registro y agregación de métricas "online" del agente:

  - latencia total de respuesta por turno
  - número de tool-calls por turno y cuáles se usaron
  - tasa de éxito de tool-calls (sin excepción)
  - feedback explícito del usuario (👍 / 👎) por respuesta
  - tasa de fallback (el agente no pudo responder / hubo error)

Se guarda en un .jsonl simple (una línea = un turno), fácil de leer con
pandas para armar la sección "resultados online" del README/Model Card.
"""

import json
import time
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "logs" / "interactions.jsonl"


def log_interaction(
    latency_seconds: float,
    tools_used: list,
    tool_errors: int,
    success: bool,
    feedback: str | None = None,
) -> None:
    """Agrega una línea al log de interacciones."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "latency_seconds": round(latency_seconds, 3),
        "tools_used": tools_used,
        "num_tool_calls": len(tools_used),
        "tool_errors": tool_errors,
        "success": success,
        "feedback": feedback,  # "up", "down" o None
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_last_feedback(feedback: str) -> None:
    """Actualiza el feedback del último registro (llamado al presionar 👍/👎)."""
    if not LOG_PATH.exists():
        return
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    last = json.loads(lines[-1])
    last["feedback"] = feedback
    lines[-1] = json.dumps(last, ensure_ascii=False)
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_summary() -> dict:
    """Calcula métricas agregadas para mostrar en la sidebar o en el README."""
    if not LOG_PATH.exists():
        return {}
    records = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines()]
    if not records:
        return {}

    n = len(records)
    avg_latency = sum(r["latency_seconds"] for r in records) / n
    avg_tool_calls = sum(r["num_tool_calls"] for r in records) / n
    success_rate = sum(1 for r in records if r["success"]) / n
    tool_error_rate = sum(r["tool_errors"] for r in records) / max(
        1, sum(r["num_tool_calls"] for r in records)
    )
    feedback_up = sum(1 for r in records if r.get("feedback") == "up")
    feedback_down = sum(1 for r in records if r.get("feedback") == "down")

    return {
        "total_interactions": n,
        "avg_latency_seconds": round(avg_latency, 2),
        "avg_tool_calls_per_turn": round(avg_tool_calls, 2),
        "success_rate": round(success_rate, 3),
        "tool_error_rate": round(tool_error_rate, 3),
        "feedback_up": feedback_up,
        "feedback_down": feedback_down,
    }
