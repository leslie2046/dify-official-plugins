# Template Service Plugin

This Dify extension plugin provides API endpoints to serve explore app templates. You can use it to replace the default Dify template service with your own custom templates.

## Features

- 📋 **List Templates**: Get all templates for the explore page
- 📄 **Get Template Detail**: Get template details with DSL for importing
- ➕ **Add Templates**: Add new app templates via API (auto-parses DSL!)
- 🗑️ **Delete Templates**: Remove templates via API
- 📊 **Admin Dashboard**: View all templates with statistics
- 🔐 **Admin Authentication**: Secure template management with API key

## Installation

1. Package the plugin:
   ```bash
   cd extensions/template_service
   dify-plugin package .
   ```

2. Install the plugin in Dify via the Plugin Management page

3. Configure the plugin with your Admin API Key

## Configuration

After installing the plugin, you need to configure:

| Setting | Description |
|---------|-------------|
| Admin API Key | Secret key for managing templates (add/delete) |

## API Endpoints

### 1. List Templates (Public)

For Dify explore page consumption.

```http
GET /apps?language=zh-Hans
```

### 2. Get Template Detail (Public)

For Dify app import.

```http
GET /apps/{app_id}
```

### 3. Add Template (Admin) ⭐ Smart DSL Parsing

**Only 3 required fields!** The plugin automatically extracts `name`, `mode`, `icon`, `icon_background` from the DSL.

```http
POST /admin/templates
Authorization: Bearer your-admin-key
Content-Type: application/json

{
  "export_data": "app:\n  name: My App\n  icon: 🤖\n  ...",
  "category": "推荐",
  "language": "zh-Hans",
  "description": "可选的自定义描述",
  "position": 1
}
```

**Required Fields:**
| Field | Description |
|-------|-------------|
| `export_data` | Full DSL YAML content |
| `category` | Category name |
| `language` | Language code (e.g., `zh-Hans`) |

**Optional Fields (auto-parsed from DSL if not provided):**
- `name`, `mode`, `icon`, `icon_background`, `icon_type`, `description`

**Other Optional Fields:**
- `app_id`: Custom ID (auto-generated if not provided)
- `position`: Sort order (default: 0)
- `copyright`, `privacy_policy`, `custom_disclaimer`

### 4. Delete Template (Admin)

```http
DELETE /admin/templates/{app_id}
Authorization: Bearer your-admin-key
```

### 5. List All Templates (Admin)

View all templates with statistics.

```http
GET /admin/templates
Authorization: Bearer your-admin-key
```

Optional query params:
- `language`: Filter by language

**Response:**
```json
{
  "total": 10,
  "language_stats": {"zh-Hans": 5, "en-US": 5},
  "categories": {"zh-Hans": ["推荐", "AI编程"], "en-US": ["Recommended"]},
  "templates": [...]
}
```

## Dify Configuration

To use this plugin as your template service, configure Dify's environment:

```bash
HOSTED_FETCH_APP_TEMPLATES_MODE=remote
HOSTED_FETCH_APP_TEMPLATES_REMOTE_DOMAIN=https://your-plugin-endpoint
```

## How to Get export_data (DSL)

### Option 1: Export from Dify Console

1. Go to your app in Dify
2. Click Settings → Export
3. Copy the YAML content

### Option 2: Use Dify API

```bash
curl "https://your-dify/console/api/apps/{app_id}/export" \
  -H "Authorization: Bearer your-token"
```

## Example: Adding a Template (Simplified!)

```bash
curl -X POST "https://your-plugin-endpoint/admin/templates" \
  -H "Authorization: Bearer your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "category": "客服",
    "language": "zh-Hans",
    "position": 1,
    "export_data": "app:\n  icon: 🤖\n  icon_background: \"#FFEAD5\"\n  mode: chat\n  name: 智能客服助手\nmodel_config:\n  model:\n    provider: openai\n    name: gpt-4\n  pre_prompt: 你是专业客服..."
  }'
```

The plugin will automatically extract `name`, `mode`, `icon`, `icon_background` from the DSL!

## Using with Dify Workflow

Create a workflow to manage templates:

```
开始 → 表单输入(category, language, position) → 代码节点(获取DSL) → HTTP Request(POST /admin/templates) → 结束
```

## License

MIT License
