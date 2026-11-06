"""Decides whether the site is finished. Its verdict is what routes the retry loop."""

from typing import Any, Final

from realtime_agentic_site_builder.agents.runtime import NodeContext, is_wireframe
from realtime_agentic_site_builder.agents.schemas import CriticVerdict, SwarmPlan
from realtime_agentic_site_builder.agents.state import AgentWorkflowState
from realtime_agentic_site_builder.agents.tsx import find_tsx_issue
from realtime_agentic_site_builder.domain.backend import FullstackProjectBundle
from realtime_agentic_site_builder.domain.swarm import AgentAction

SYSTEM_PROMPT: Final = (
    "You review generated site code that already passed structural checks. Approve it unless a "
    "component or endpoint has a concrete defect. Name each defect by its exact component id or "
    "endpoint id."
)


def _structural_issues(
    plan: SwarmPlan, bundle: FullstackProjectBundle
) -> tuple[dict[str, str], dict[str, str]]:
    """Run the checks no model can overrule. Handlers are absent: the schema parses them."""
    frontend: dict[str, str] = {}
    backend: dict[str, str] = {}
    for component in bundle.frontend.components:
        if is_wireframe(component):
            frontend[component.id] = "still a wireframe: no code was committed"
        else:
            issue = find_tsx_issue(component.code)
            if issue is not None:
                frontend[component.id] = f"invalid TSX at {issue}"
    endpoints = {endpoint.id for endpoint in bundle.backend.endpoints}
    for spec in plan.endpoints:
        if spec.endpoint_id not in endpoints:
            backend[spec.endpoint_id] = "planned endpoint was never committed"
    nodes = {component.id: component for component in bundle.frontend.components}
    for spec_component in plan.components:
        node = nodes.get(spec_component.id)
        if (
            node is not None
            and spec_component.endpoint_id is not None
            and (
                node.connected_endpoint_id != spec_component.endpoint_id
                or node.component_type is not spec_component.component_type
            )
        ):
            backend[spec_component.id] = f"not bound to {spec_component.endpoint_id}"
    return frontend, backend


async def _model_review(
    context: NodeContext, bundle: FullstackProjectBundle
) -> tuple[dict[str, str], dict[str, str]]:
    sections = [f"Component {c.id}:\n{c.code}" for c in bundle.frontend.components]
    sections += [f"Endpoint {e.id}:\n{e.handler_code}" for e in bundle.backend.endpoints]
    result = await context.generate(SYSTEM_PROMPT, "\n\n".join(sections), CriticVerdict)
    verdict = result.output
    frontend: dict[str, str] = {}
    backend: dict[str, str] = {}
    if verdict.approved:
        return frontend, backend
    components = {component.id for component in bundle.frontend.components}
    endpoints = {endpoint.id for endpoint in bundle.backend.endpoints}
    for issue in verdict.issues:
        if issue.target_id in components:
            frontend[issue.target_id] = issue.message
        elif issue.target_id in endpoints:
            backend[issue.target_id] = issue.message
        else:
            # A retry needs a target, so this is advice, and it must not burn an iteration.
            context.fail(issue.target_id, f"advisory: {issue.message}")
    return frontend, backend


async def critic_node(state: AgentWorkflowState, context: NodeContext) -> dict[str, Any]:
    bundle = context.bundle()
    frontend, backend = _structural_issues(state["plan"], bundle)
    if not frontend and not backend:
        frontend, backend = await _model_review(context, bundle)
    for target_id, problem in (*frontend.items(), *backend.items()):
        context.fail(target_id, problem)

    iteration = state["iteration"] + 1
    failed = bool(frontend or backend)
    exhausted = failed and iteration >= context.runtime.hub.settings.agent_max_iterations
    if exhausted:
        context.fail("swarm", f"gave up after {iteration} iterations")
    await context.emit(
        AgentAction.RESOLVING_CONFLICT if failed else AgentAction.IDLE,
        f"Rejected {len(frontend) + len(backend)} elements." if failed else "Approved the site.",
        tool_name="critic.verdict",
        tool_args={"approved": not failed, "iteration": iteration},
    )
    return {
        "iteration": iteration,
        "completed": not failed,
        "retry_frontend": () if exhausted else tuple(sorted(frontend)),
        "retry_backend": () if exhausted else tuple(sorted(backend)),
    }
