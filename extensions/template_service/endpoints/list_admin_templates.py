"""
Endpoint for listing all templates (admin view).
GET /admin/templates
"""
import json
from typing import Mapping
from werkzeug import Request, Response
from dify_plugin import Endpoint
from endpoints.template_storage import TemplateStorage


class ListAdminTemplatesEndpoint(Endpoint):
    def _invoke(self, r: Request, values: Mapping, settings: Mapping) -> Response:
        """
        List all templates with full details (admin view).
        
        Headers:
            - Authorization: Bearer <admin_key>
        
        Query params:
            - language: Filter by language (optional)
        
        Returns:
            JSON with all templates and statistics.
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
        
        # Get templates from storage
        storage = TemplateStorage(self.session)
        all_templates = storage._get_all_templates()
        all_categories = storage._get_categories()
        
        # Optional language filter
        language_filter = r.args.get("language")
        
        templates_list = []
        for app_id, template in all_templates.items():
            if language_filter and template.get("language") != language_filter:
                continue
            templates_list.append({
                "id": template["id"],
                "name": template["name"],
                "mode": template["mode"],
                "icon": template["icon"],
                "icon_background": template["icon_background"],
                "category": template["category"],
                "language": template["language"],
                "position": template.get("position", 0),
                "description": template.get("description", ""),
                "is_listed": template.get("is_listed", True)
            })
        
        # Sort by language, then by position
        templates_list.sort(key=lambda x: (x["language"], x["position"]))
        
        # Build language stats
        language_stats = {}
        for template in all_templates.values():
            lang = template.get("language", "unknown")
            language_stats[lang] = language_stats.get(lang, 0) + 1
        
        return Response(
            json.dumps({
                "total": len(templates_list),
                "language_stats": language_stats,
                "categories": {lang: sorted(list(cats)) for lang, cats in all_categories.items()},
                "templates": templates_list
            }, ensure_ascii=False),
            status=200,
            content_type="application/json"
        )
