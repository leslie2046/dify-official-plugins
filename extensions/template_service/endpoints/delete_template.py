"""
Endpoint for deleting an app template.
DELETE /admin/templates/<app_id>
"""
import json
from typing import Mapping
from werkzeug import Request, Response
from dify_plugin import Endpoint
from endpoints.template_storage import TemplateStorage


class DeleteTemplateEndpoint(Endpoint):
    def _invoke(self, r: Request, values: Mapping, settings: Mapping) -> Response:
        """
        Delete an app template.
        
        Headers:
            - Authorization: Bearer <admin_key>
        
        Path params:
            - app_id: The ID of the template to delete.
        
        Returns:
            JSON with success message.
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
        
        app_id = values.get("app_id", "")
        
        if not app_id:
            return Response(
                json.dumps({"error": "app_id is required"}),
                status=400,
                content_type="application/json"
            )
        
        # Delete template from storage
        storage = TemplateStorage(self.session)
        success = storage.delete_template(app_id)
        
        if not success:
            return Response(
                json.dumps({"error": "Template not found"}),
                status=404,
                content_type="application/json"
            )
        
        return Response(
            json.dumps({"message": "Template deleted successfully"}),
            status=200,
            content_type="application/json"
        )
