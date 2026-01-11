"""
Template storage manager for managing app templates using Dify plugin storage.
"""
import json
from typing import Any
from dify_plugin.core.runtime import Session


TEMPLATES_KEY = "templates"
CATEGORIES_KEY = "categories"


class TemplateStorage:
    """Manages template data using Dify plugin storage."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def _get_all_templates(self) -> dict[str, dict[str, Any]]:
        """Get all templates from storage."""
        try:
            data = self.session.storage.get(TEMPLATES_KEY)
            if data:
                return json.loads(data.decode('utf-8'))
        except Exception:
            pass
        return {}
    
    def _save_all_templates(self, templates: dict[str, dict[str, Any]]) -> None:
        """Save all templates to storage."""
        self.session.storage.set(TEMPLATES_KEY, json.dumps(templates).encode('utf-8'))
    
    def _get_categories(self) -> dict[str, set[str]]:
        """Get categories by language from storage."""
        try:
            data = self.session.storage.get(CATEGORIES_KEY)
            if data:
                raw = json.loads(data.decode('utf-8'))
                # Convert lists back to sets
                return {lang: set(cats) for lang, cats in raw.items()}
        except Exception:
            pass
        return {}
    
    def _save_categories(self, categories: dict[str, set[str]]) -> None:
        """Save categories to storage."""
        # Convert sets to lists for JSON serialization
        raw = {lang: list(cats) for lang, cats in categories.items()}
        self.session.storage.set(CATEGORIES_KEY, json.dumps(raw).encode('utf-8'))
    
    def add_template(
        self,
        app_id: str,
        name: str,
        mode: str,
        icon: str,
        icon_background: str,
        description: str,
        category: str,
        language: str,
        position: int,
        export_data: str,
        icon_type: str = "emoji",
        copyright: str = "",
        privacy_policy: str = "",
        custom_disclaimer: str = ""
    ) -> dict[str, Any]:
        """Add a new template."""
        templates = self._get_all_templates()
        categories = self._get_categories()
        
        template = {
            "id": app_id,
            "name": name,
            "mode": mode,
            "icon": icon,
            "icon_type": icon_type,
            "icon_background": icon_background,
            "description": description,
            "category": category,
            "language": language,
            "position": position,
            "export_data": export_data,
            "copyright": copyright,
            "privacy_policy": privacy_policy,
            "custom_disclaimer": custom_disclaimer,
            "is_listed": True
        }
        
        templates[app_id] = template
        
        # Update categories
        if language not in categories:
            categories[language] = set()
        categories[language].add(category)
        
        self._save_all_templates(templates)
        self._save_categories(categories)
        
        return template
    
    def delete_template(self, app_id: str) -> bool:
        """Delete a template by ID."""
        templates = self._get_all_templates()
        
        if app_id not in templates:
            return False
        
        del templates[app_id]
        self._save_all_templates(templates)
        
        # Rebuild categories
        self._rebuild_categories(templates)
        
        return True
    
    def _rebuild_categories(self, templates: dict[str, dict[str, Any]]) -> None:
        """Rebuild categories from templates."""
        categories: dict[str, set[str]] = {}
        
        for template in templates.values():
            lang = template.get("language", "en-US")
            cat = template.get("category", "")
            if cat:
                if lang not in categories:
                    categories[lang] = set()
                categories[lang].add(cat)
        
        self._save_categories(categories)
    
    def get_template(self, app_id: str) -> dict[str, Any] | None:
        """Get a template by ID."""
        templates = self._get_all_templates()
        return templates.get(app_id)
    
    def list_templates(self, language: str) -> dict[str, Any]:
        """
        List all templates for a given language.
        Returns format expected by Dify explore page.
        """
        templates = self._get_all_templates()
        categories = self._get_categories()
        
        recommended_apps = []
        
        for app_id, template in templates.items():
            if template.get("language") != language:
                continue
            if not template.get("is_listed", True):
                continue
            
            recommended_apps.append({
                "app": {
                    "id": template["id"],
                    "name": template["name"],
                    "mode": template["mode"],
                    "icon": template["icon"],
                    "icon_type": template.get("icon_type", "emoji"),
                    "icon_background": template["icon_background"]
                },
                "app_id": template["id"],
                "description": template.get("description", ""),
                "copyright": template.get("copyright", ""),
                "privacy_policy": template.get("privacy_policy", ""),
                "custom_disclaimer": template.get("custom_disclaimer", ""),
                "category": template["category"],
                "position": template.get("position", 0),
                "is_listed": True
            })
        
        # Sort by position
        recommended_apps.sort(key=lambda x: x.get("position", 0))
        
        return {
            "categories": sorted(list(categories.get(language, set()))),
            "recommended_apps": recommended_apps
        }
    
    def get_app_detail(self, app_id: str) -> dict[str, Any] | None:
        """
        Get app detail for template import.
        Returns format expected by Dify for importing apps.
        """
        template = self.get_template(app_id)
        if not template:
            return None
        
        return {
            "id": template["id"],
            "name": template["name"],
            "icon": template["icon"],
            "icon_background": template["icon_background"],
            "mode": template["mode"],
            "export_data": template.get("export_data", "")
        }
