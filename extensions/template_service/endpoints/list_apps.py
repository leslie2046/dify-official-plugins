"""
Endpoint for listing app templates.
GET /apps?language={language}
"""
import json
from typing import Mapping
from werkzeug import Request, Response
from dify_plugin import Endpoint
from endpoints.template_storage import TemplateStorage


class ListAppsEndpoint(Endpoint):
    def _invoke(self, r: Request, values: Mapping, settings: Mapping) -> Response:
        """
        List all app templates for the explore page.
        
        Query params:
            - language: Language code (e.g., 'zh-Hans', 'en-US'). Defaults to 'en-US'.
        
        Returns:
            JSON with categories and recommended_apps arrays.
        """
        # Get language from query params
        language = r.args.get("language", "en-US")
        
        # Get templates from storage
        storage = TemplateStorage(self.session)
        result = storage.list_templates(language)
        
        # If no results for requested language, try en-US as fallback
        if not result.get("recommended_apps") and language != "en-US":
            result = storage.list_templates("en-US")
        
        return Response(
            json.dumps(result, ensure_ascii=False),
            status=200,
            content_type="application/json"
        )
