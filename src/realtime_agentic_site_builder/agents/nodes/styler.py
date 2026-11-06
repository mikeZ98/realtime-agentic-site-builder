"""Applies the theme, plus targeted class fixes to components that have real code."""

from typing import Any, Final

from realtime_agentic_site_builder.agents.runtime import NodeContext, is_wireframe
from realtime_agentic_site_builder.agents.schemas import StyleOutput
from realtime_agentic_site_builder.agents.state import AgentWorkflowState
from realtime_agentic_site_builder.core.patch_engine import FRONTEND_DOCUMENT_LOCK
from realtime_agentic_site_builder.domain.blueprint import ComponentNode
from realtime_agentic_site_builder.domain.swarm import AgentAction

SYSTEM_PROMPT: Final = (
    "You refine a site's visual design: palette, contrast and responsive Tailwind classes. Propose "
    "a theme if it should change, and exact text replacements for class names. Each `find` must "
    "occur verbatim in that component's code, and replacements must not change its structure."
)


def _targets(state: AgentWorkflowState, context: NodeContext) -> list[ComponentNode]:
    components = [c for c in context.bundle().frontend.components if not is_wireframe(c)]
    if state["iteration"] == 0:
        return components
    retry = set(state["retry_frontend"])
    return [component for component in components if component.id in retry]


def _style_patches(
    output: StyleOutput, context: NodeContext, target_ids: set[str]
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    if output.theme is not None:
        patches.append({"op": "replace", "path": "/frontend/theme", "value": output.theme})
    code = {c.id: c.code for c in context.bundle().frontend.components}
    for fix in output.fixes:
        # Anything else the model names was not locked for this run, or has no such text.
        if fix.component_id in target_ids and fix.find in code[fix.component_id]:
            code[fix.component_id] = code[fix.component_id].replace(fix.find, fix.replace)
            index = context.component_index(fix.component_id)
            patches.append(
                {
                    "op": "replace",
                    "path": f"/frontend/components/{index}/code",
                    "value": code[fix.component_id],
                }
            )
    return patches


async def styler_node(state: AgentWorkflowState, context: NodeContext) -> dict[str, Any]:
    targets = _targets(state, context)
    if not targets:
        return {}
    target_ids = {component.id for component in targets}
    with context.recoverable("styling"):
        async with context.locks() as locks:
            await locks.acquire(FRONTEND_DOCUMENT_LOCK, *target_ids)
            sections = "\n\n".join(f"Component {c.id}:\n{c.code}" for c in targets)
            prompt = f"Theme: {context.bundle().frontend.theme}\n\n{sections}"
            result = await context.generate(SYSTEM_PROMPT, prompt, StyleOutput)
            patches = _style_patches(result.output, context, target_ids)
            if patches:
                await context.commit("frontend", FRONTEND_DOCUMENT_LOCK, patches)
            else:
                await context.emit(AgentAction.IDLE, "No style changes were needed.")
    return {}
