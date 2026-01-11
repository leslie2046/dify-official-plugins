"""
Endpoint for getting app template detail.
GET /apps/<app_id>
"""
import json
from typing import Mapping
from werkzeug import Request, Response
from dify_plugin import Endpoint
from endpoints.template_storage import TemplateStorage


class GetAppEndpoint(Endpoint):
    def _invoke(self, r: Request, values: Mapping, settings: Mapping) -> Response:
        """
        Get app template detail for import.
        
        Path params:
            - app_id: The ID of the app template.
        
        Returns:
            JSON with app detail including export_data for importing.
        """
        app_id = values.get("app_id", "")
        
        if not app_id:
            return Response(
                json.dumps({"error": "app_id is required"}),
                status=400,
                content_type="application/json"
            )
        
        # Get template from storage
        storage = TemplateStorage(self.session)
        result = storage.get_app_detail(app_id)
        
        if not result:
            return Response(
                json.dumps({"error": "Template not found"}),
                status=404,
                content_type="application/json"
            )
        
        return Response(
            json.dumps(result, ensure_ascii=False),
            status=200,
            content_type="application/json"
        )
