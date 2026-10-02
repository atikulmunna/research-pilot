"""Render compact JSON shapes of pydantic models for prompts."""

import inspect
import json
import types
from typing import Annotated, Any, Dict, List, Literal, Union, get_args, get_origin

from pydantic import BaseModel


def _llm_fields(model: type[BaseModel]):
    for name, field in model.model_fields.items():
        extra = field.json_schema_extra
        if isinstance(extra, dict) and extra.get("llm") is False:
            continue
        yield name, field


def skeleton(model: type[BaseModel]) -> Dict[str, Any]:
    return {name: _example(field.annotation, field.description) for name, field in _llm_fields(model)}


def _example(annotation: Any, description: str | None = None) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _example(args[0], description)
    if origin is Literal:
        return "|".join(str(a) for a in args)
    if origin in (Union, types.UnionType):
        non_null = [a for a in args if a is not type(None)]
        return _example(non_null[0], description) if non_null else None
    if origin in (list, List):
        return [_example(args[0])] if args else []
    if origin in (dict, Dict):
        return {}
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return skeleton(annotation)
    if annotation is bool:
        return False
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    return f"<{description}>" if description else "<text>"


def field_notes(model: type[BaseModel], prefix: str = "", seen: set | None = None) -> List[str]:
    seen = seen if seen is not None else set()
    if model in seen:
        return []
    seen.add(model)
    notes: List[str] = []
    for name, field in _llm_fields(model):
        if field.description:
            notes.append(f"{prefix}{name}: {field.description}")
        nested = _nested_model(field.annotation)
        if nested is not None:
            notes.extend(field_notes(nested, prefix=f"{prefix}{name}.", seen=seen))
    return notes


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    origin = get_origin(annotation)
    if origin in (Annotated, list, List, Union, types.UnionType):
        for arg in get_args(annotation):
            found = _nested_model(arg)
            if found is not None:
                return found
        return None
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation
    return None


def render_schema(model: type[BaseModel]) -> str:
    lines = [
        "Respond with ONE JSON object and nothing else. Use exactly this shape "
        "(placeholder values show the expected type):",
        json.dumps(skeleton(model), indent=1, ensure_ascii=False),
    ]
    notes = field_notes(model)
    if notes:
        lines.append("Field notes:")
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines)
