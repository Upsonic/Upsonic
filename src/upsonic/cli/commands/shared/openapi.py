from typing import Any, Dict, List


def map_inputs_props(inputs_schema: List[Dict[str, Any]]) -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Map input schema to JSON and multipart properties for OpenAPI schema.
    
    Args:
        inputs_schema: List of input schema items
        
    Returns:
        Tuple of (json_props, multipart_props, required_fields)
    """
    json_props = {}
    multipart_props = {}
    required = []
    
    for item in inputs_schema:
        name = item["name"]
        itype = item.get("type", "string")
        default = item.get("default")
        
        if itype in ("files",):
            json_props[name] = {"type": "array", "items": {"type": "string"}}
            multipart_props[name] = {"type": "array", "items": {"type": "string", "format": "binary"}}
        elif itype in ("file", "binary", "string($binary)"):
            json_props[name] = {"type": "string"}
            multipart_props[name] = {"type": "string", "format": "binary"}
        elif itype == "number":
            json_props[name] = {"type": "number"}
            multipart_props[name] = {"type": "number"}
        elif itype == "integer":
            json_props[name] = {"type": "integer"}
            multipart_props[name] = {"type": "integer"}
        elif itype in ("boolean", "bool"):
            json_props[name] = {"type": "boolean"}
            multipart_props[name] = {"type": "boolean"}
        elif itype in ("list", "array"):
            json_props[name] = {"type": "array", "items": {"type": "string"}}
            multipart_props[name] = {"type": "array", "items": {"type": "string"}}
        elif itype in ("json",):
            json_props[name] = {"type": "object"}
            multipart_props[name] = {"type": "object"}
        else:
            json_props[name] = {"type": "string"}
            multipart_props[name] = {"type": "string"}
        
        # Add default values if present
        if default is not None:
            try:
                json_props[name]["default"] = default
            except Exception:
                pass
        
        if item.get("required", False):
            required.append(name)
    
    return json_props, multipart_props, required


def map_output_props(output_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map output schema to OpenAPI properties.
    
    Args:
        output_schema: Output schema dictionary
        
    Returns:
        Dictionary of OpenAPI properties
    """
    props = {}
    for k, v in (output_schema or {}).items():
        t = v.get("type", "string")
        if t == "number":
            props[k] = {"type": "number", "description": v.get("description", "")}
        elif t == "integer":
            props[k] = {"type": "integer", "description": v.get("description", "")}
        elif t in ("boolean", "bool"):
            props[k] = {"type": "boolean", "description": v.get("description", "")}
        elif t in ("list", "array"):
            props[k] = {"type": "array", "items": {"type": "string"}, "description": v.get("description", "")}
        elif t in ("json", "object"):
            props[k] = {"type": "object", "description": v.get("description", "")}
        else:
            props[k] = {"type": "string", "description": v.get("description", "")}
    return props


def modify_openapi_schema(
    schema: Dict[str, Any],
    inputs_schema: List[Dict[str, Any]],
    output_schema_dict: Dict[str, Any],
    path: str = "/call",
) -> Dict[str, Any]:
    """
    Modify OpenAPI schema to include multipart/form-data and application/json.
    
    Args:
        schema: The OpenAPI schema dictionary
        inputs_schema: List of input schema items
        output_schema_dict: Output schema dictionary
        path: The endpoint path to modify
        
    Returns:
        Modified schema dictionary
    """
    paths = schema.get("paths", {})
    if path not in paths:
        return schema
    
    post_op = paths[path].get("post", {})
    
    # Map props
    json_props, multipart_props, required_fields = map_inputs_props(inputs_schema)
    
    # Build RequestModel for JSON
    request_model = {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": json_props
            }
        }
    }
    if required_fields:
        request_model["properties"]["data"]["required"] = required_fields
    
    # Add components/schemas
    components = schema.setdefault("components", {})
    comps_schemas = components.setdefault("schemas", {})
    comps_schemas["RequestModel"] = request_model
    comps_schemas["JobStatus"] = {
        "type": "object",
        "properties": map_output_props(output_schema_dict)
    }
    
    # Build content with multipart/form-data FIRST
    content = {}
    multipart_schema = {
        "type": "object",
        "properties": multipart_props,
    }
    if required_fields:
        multipart_schema["required"] = required_fields
    content["multipart/form-data"] = {"schema": multipart_schema}
    content["application/json"] = {"schema": {"$ref": "#/components/schemas/RequestModel"}}
    
    # Set responses
    responses = post_op.setdefault("responses", {})
    responses["200"] = {
        "description": "Successful Response",
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/JobStatus"}
            }
        }
    }
    
    # Replace requestBody
    post_op["requestBody"] = {
        "required": True,
        "content": content
    }
    
    # Ensure tags
    if not post_op.get("tags"):
        post_op["tags"] = ["jobs"]
    
    return schema

