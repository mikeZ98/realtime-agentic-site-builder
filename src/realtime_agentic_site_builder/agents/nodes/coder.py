"""Writes each wireframed component's TSX, holding that component's lock while it does."""

from typing import Any, Final

from realtime_agentic_site_builder.agents.runtime import NodeContext, is_wireframe
from realtime_agentic_site_builder.agents.schemas import CoderOutput
from realtime_agentic_site_builder.agents.state import AgentWorkflowState
from realtime_agentic_site_builder.domain.swarm import AgentAction

SYSTEM_PROMPT: Final = (
    "You write one React component in TSX with Tailwind classes. Return a complete module with a "
    "default export. Every tag and bracket must be closed."
)


def _prompt(state: AgentWorkflowState, context: NodeContext, component_id: str, code: str) -> str:
    spec = next((item for item in state["plan"].components if item.id == component_id), None)
    lines = [f"Component id: {component_id}"]
    if spec is not None:
        lines.append(f"Title: {spec.title}\nBrief: {spec.brief}")
        if spec.binds_path is not None:
            lines.append(f"Submit with {spec.binds_method.value} to {spec.binds_path}.")
    feedback = context.feedback_for(component_id)
    if feedback:
        lines.append(f"Your previous attempt was rejected:\n{feedback}\nPrevious code:\n{code}")
    return "\n".join(lines)


async def coder_node(state: AgentWorkflowState, context: NodeContext) -> dict[str, Any]:
    components = context.bundle().frontend.components
    if state["iteration"] == 0:
        targets = [component.id for component in components if is_wireframe(component)]
    else:
        targets = list(state["retry_frontend"])
    current = {component.id: component.code for component in components}
    for component_id in targets:
        with context.recoverable(f"component {component_id}"):
            async with context.locks() as locks:
                await context.emit(
                    AgentAction.ACQUIRING_LOCK,
                    f"Locking {component_id}.",
                    tool_name="lock.acquire",
                    tool_args={"target_id": component_id},
                    target_id=component_id,
                )
                await locks.acquire(component_id)
                await context.emit(
                    AgentAction.EDITING,
                    f"Writing {component_id}.",
                    target_id=component_id,
                    lock=locks.lock_on(component_id),
                )
                prompt = _prompt(state, context, component_id, current.get(component_id, ""))
                result = await context.generate(
                    SYSTEM_PROMPT, prompt, CoderOutput, target_id=component_id
                )
                index = context.component_index(component_id)
                await context.commit(
                    "frontend",
                    component_id,
                    [
                        # Guards the position: the pointer must still name this component.
                        {
                            "op": "test",
                            "path": f"/frontend/components/{index}/id",
                            "value": component_id,
                        },
                        {
                            "op": "replace",
                            "path": f"/frontend/components/{index}/code",
                            "value": result.output.code,
                        },
                    ],
                    lock=locks.lock_on(component_id),
                )
    return {}
