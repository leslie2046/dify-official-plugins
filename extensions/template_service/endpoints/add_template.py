"""
Endpoint for adding a new app template.
POST /admin/templates

Supports auto-parsing fields from DSL YAML, so you only need to provide:
- export_data (required): Full DSL YAML content
- category (required): Category name
- language (required): Language code
- description (optional): Override description from DSL
- position (optional): Sort position
"""
import json
import uuid
import re
from typing import Mapping, Any
from werkzeug import Request, Response
from dify_plugin import Endpoint
from endpoints.template_storage import TemplateStorage


def parse_dsl(dsl_content: str) -> dict[str, Any]:
    """
    Parse DSL YAML content to extract app metadata.
    Uses simple regex parsing to avoid requiring PyYAML.
    
    Returns dict with: name, mode, icon, icon_background, icon_type, description
    """
    result = {
        "name": "",
        "mode": "chat",
        "icon": "🤖",
        "icon_background": "#FFEAD5",
        "icon_type": "emoji",
        "description": ""
    }
    
    # Try to find app section
    # Pattern: app:\n  key: value
    
    # Extract name
    name_match = re.search(r'^\s*name:\s*["\']?([^"\'\n]+)["\']?\s*$', dsl_content, re.MULTILINE)
    if name_match:
        result["name"] = name_match.group(1).strip()
    
    # Extract mode
    mode_match = re.search(r'^\s*mode:\s*["\']?([^"\'\n]+)["\']?\s*$', dsl_content, re.MULTILINE)
    if mode_match:
        result["mode"] = mode_match.group(1).strip()
    
    # Extract icon (handle unicode escapes like \U0001F916)
    icon_match = re.search(r'^\s*icon:\s*["\']?((?:\\U[0-9A-Fa-f]{8}|[^"\'\n])+)["\']?\s*$', dsl_content, re.MULTILINE)
    if icon_match:
        icon_str = icon_match.group(1).strip()
        # Decode unicode escapes
        try:
            icon_str = icon_str.encode('utf-8').decode('unicode_escape')
        except Exception:
            pass
        result["icon"] = icon_str
    
    # Extract icon_background
    bg_match = re.search(r'^\s*icon_background:\s*["\']?([#\w]+)["\']?\s*$', dsl_content, re.MULTILINE)
    if bg_match:
        result["icon_background"] = bg_match.group(1).strip()
    
    # Extract icon_type
    type_match = re.search(r'^\s*icon_type:\s*["\']?(emoji|image)["\']?\s*$', dsl_content, re.MULTILINE)
    if type_match:
        result["icon_type"] = type_match.group(1).strip()
    
    # Try to extract description from various places
    desc_match = re.search(r'^\s*description:\s*["\']?([^"\'\n]+)["\']?\s*$', dsl_content, re.MULTILINE)
    if desc_match:
        result["description"] = desc_match.group(1).strip()
    
    return result


class AddTemplateEndpoint(Endpoint):
    def _invoke(self, r: Request, values: Mapping, settings: Mapping) -> Response:
        """
        Add a new app template with auto-parsing from DSL.
        
        Headers:
            - Authorization: Bearer <admin_key>
        
        Body (JSON):
            Required:
            - export_data: Full DSL YAML content
            - category: Category name
            - language: Language code, e.g., 'zh-Hans'
            
            Optional (auto-parsed from DSL if not provided):
            - name: App name
            - mode: App mode - chat/agent-chat/workflow/advanced-chat/completion
            - icon: Emoji or image URL
            - icon_background: Background color
            - icon_type: 'emoji' or 'image'
            - description: App description
            
            Optional:
            - position: Sort position (default 0)
            - copyright: Copyright info
            - privacy_policy: Privacy policy URL
            - custom_disclaimer: Custom disclaimer
            - app_id: Custom app ID (auto-generated if not provided)
        
        Returns:
            JSON with created template info.
        """
        # Verify admin key
        admin_key = settings.get("admin_key", "")
        auth_header = r.headers.get("Authorization", "")
        
        if not auth_header.startswith("Bearer "):
            return Response(
                json.dumps({"error": "Authorization header required"}),
                status=401,
                content_type="application/json"
            )
        
        provided_key = auth_header[7:]  # Remove "Bearer " prefix
        if provided_key != admin_key:
            return Response(
                json.dumps({"error": "Invalid admin key"}),
                status=401,
                content_type="application/json"
            )
        
        # Parse request body
        try:
            data = r.get_json()
        except Exception as e:
            return Response(
                json.dumps({"error": f"Invalid JSON: {str(e)}"}),
                status=400,
                content_type="application/json"
            )
        
        if not data:
            return Response(
                json.dumps({"error": "Request body is required"}),
                status=400,
                content_type="application/json"
            )
        
        # Validate required fields (only export_data, category, language are truly required now)
        required_fields = ["export_data", "category", "language"]
        
        for field in required_fields:
            if not data.get(field):
                return Response(
                    json.dumps({"error": f"Field '{field}' is required"}),
                    status=400,
                    content_type="application/json"
                )
        
        # Parse DSL to extract app metadata
        dsl_content = data["export_data"]
        parsed = parse_dsl(dsl_content)
        
        # Use provided values or fall back to parsed values
        name = data.get("name") or parsed["name"] or "Untitled App"
        mode = data.get("mode") or parsed["mode"]
        icon = data.get("icon") or parsed["icon"]
        icon_background = data.get("icon_background") or parsed["icon_background"]
        icon_type = data.get("icon_type") or parsed["icon_type"]
        description = data.get("description") or parsed["description"] or ""
        
        # Validate mode
        valid_modes = ["chat", "agent-chat", "workflow", "advanced-chat", "completion"]
        if mode not in valid_modes:
            return Response(
                json.dumps({"error": f"Invalid mode '{mode}'. Must be one of: {valid_modes}"}),
                status=400,
                content_type="application/json"
            )
        
        # Generate or use provided app_id
        app_id = data.get("app_id") or str(uuid.uuid4())
        
        # Add template to storage
        storage = TemplateStorage(self.session)
        template = storage.add_template(
            app_id=app_id,
            name=name,
            mode=mode,
            icon=icon,
            icon_background=icon_background,
            description=description,
            category=data["category"],
            language=data["language"],
            position=data.get("position", 0),
            export_data=dsl_content,
            icon_type=icon_type,
            copyright=data.get("copyright", ""),
            privacy_policy=data.get("privacy_policy", ""),
            custom_disclaimer=data.get("custom_disclaimer", "")
        )
        
        return Response(
            json.dumps({
                "message": "Template created successfully",
                "template": {
                    "id": template["id"],
                    "name": template["name"],
                    "mode": template["mode"],
                    "icon": template["icon"],
                    "category": template["category"],
                    "language": template["language"],
                    "description": template["description"]
                },
                "parsed_from_dsl": {
                    "name": parsed["name"],
                    "mode": parsed["mode"],
                    "icon": parsed["icon"],
                    "icon_background": parsed["icon_background"]
                }
            }, ensure_ascii=False),
            status=201,
            content_type="application/json"
        )
