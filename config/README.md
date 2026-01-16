# ⚙️ AtlasTrinity Configuration

## 📁 Runtime Location (Single Source of Truth)

**All runtime configs are in `~/.config/atlastrinity/`**

```
~/.config/atlastrinity/
├── .env                    # API keys (never overwritten)
├── config.yaml             # Main system config
├── mcp/
│   └── config.json         # MCP server definitions
└── models/, logs/, memory/
```

## 📁 Template Locations (in repository)

| Template               | Source Path                           | Deployed To                              |
| ---------------------- | ------------------------------------- | ---------------------------------------- |
| `config.yaml.template` | `config/config.yaml.template`         | `~/.config/atlastrinity/config.yaml`     |
| `config.json.template` | `src/mcp_server/config.json.template` | `~/.config/atlastrinity/mcp/config.json` |

Templates are copied to global folder on **first run only**.

## 🎯 Usage

### Edit Configurations

```bash
vim ~/.config/atlastrinity/config.yaml
vim ~/.config/atlastrinity/mcp/config.json
npm run dev  # restart app
```

### Update Templates (for developers)

Edit template files in the repository, they will be used for fresh installations.

## 🔗 Related Docs

- [CONFIG_ARCHITECTURE.md](../docs/CONFIG_ARCHITECTURE.md)
- [Setup Guide](../SETUP.md)
