# AtlasTrinity CI/CD System

Comprehensive continuous integration and deployment pipeline for the AtlasTrinity project.

## 📋 Overview

The CI/CD system consists of 6 GitHub Actions workflows that automate testing, building, and deploying the entire AtlasTrinity ecosystem:

### Workflows

1. **[ci-core.yml](.github/workflows/ci-core.yml)** - Main CI pipeline
   - Python linting (flake8, black, isort, mypy)
   - TypeScript linting (ESLint, Prettier)
   - Python tests with coverage
   - TypeScript/Electron build
   - Configuration validation

2. **[test-mcp-servers.yml](.github/workflows/test-mcp-servers.yml)** - MCP server validation
   - Tier 1-4 server connectivity tests
   - Individual tool invocation tests
   - Full stack integration tests
   - macOS and Ubuntu matrix testing

3. **[test-trinity.yml](.github/workflows/test-trinity.yml)** - Trinity agent tests
   - Atlas (planner) tests
   - Tetyana (executor) tests
   - Grisha (critic) tests
   - Multi-agent orchestration scenarios
   - Knowledge graph integration

4. **[build-macos.yml](.github/workflows/build-macos.yml)** - macOS app building
   - Swift binary compilation
   - Python venv packaging
   - Electron app bundling
   - DMG creation and validation
   - Code signing (optional)

5. **[release.yml](.github/workflows/release.yml)** - Automated releases
   - Changelog generation
   - GitHub release creation
   - Asset uploads
   - Notifications

6. **[schedule-health.yml](.github/workflows/schedule-health.yml)** - Daily health checks
   - MCP server health monitoring
   - Security vulnerability scanning
   - Performance benchmarks
   - Database integrity checks
   - API availability tests

## 🚀 Quick Start

### Running CI Locally

```bash
# Install dependencies
npm ci
pip install -r requirements.txt

# Run linting
npm run lint:all

# Run tests with coverage
npm run test:ci

# Build application
npm run build:ci
```

### Triggering Workflows

**Push to main branch:**
```bash
git push origin main
```
Triggers: `ci-core`, `test-mcp-servers`, `test-trinity`, `build-macos`

**Create a release:**
```bash
git tag v1.0.0
git push origin v1.0.0
```
Triggers: `release`

**Manual workflow dispatch:**
- Go to Actions tab in GitHub
- Select workflow
- Click "Run workflow"

## 🔧 Configuration

### Required Secrets

Configure these in GitHub repository settings → Secrets and variables → Actions:

#### Critical
- `GITHUB_TOKEN` - Automatically provided by GitHub Actions

#### API Keys
- `OPENAI_API_KEY` - For GPT models (Trinity agents)
- `ANTHROPIC_API_KEY` - For Claude models
- `COPILOT_API_KEY` - For GitHub Copilot integration

#### Code Signing (Optional, for releases)
- `APPLE_ID` - Apple developer account email
- `APPLE_APP_SPECIFIC_PASSWORD` - App-specific password
- `APPLE_TEAM_ID` - Apple team ID
- `CSC_LINK` - Base64-encoded certificate
- `CSC_KEY_PASSWORD` - Certificate password

#### Notifications (Optional)
- `SLACK_WEBHOOK_URL` - For Slack notifications
- `SLACK_BOT_TOKEN` - For Slack integration
- `SLACK_TEAM_ID` - Slack team ID

### Environment Variables

Set in workflow files or `.env`:
- `PYTHON_VERSION` - Default: `3.12`
- `NODE_VERSION` - Default: `22`
- `DATABASE_URL` - PostgreSQL connection string
 - `DATABASE_URL` - SQL connection string (default: local SQLite `sqlite+aiosqlite:///${CONFIG_ROOT}/atlastrinity.db`)
- `REDIS_URL` - Redis connection string

## 📊 Workflow Matrix

| Workflow | Triggers | Platforms | Duration | Cost |
|----------|----------|-----------|----------|------|
| CI Core | Push, PR | Ubuntu, macOS | ~8 min | Low |
| MCP Tests | Push, Schedule | Ubuntu, macOS | ~12 min | Medium |
| Trinity Tests | Push | Ubuntu + Services | ~15 min | Medium |
| macOS Build | Push (main), Tags | macOS | ~25 min | High |
| Release | Tags | macOS | ~30 min | High |
| Health Checks | Daily (6 AM UTC) | macOS, Ubuntu | ~10 min | Low |

## 🧪 Testing Strategy

### Unit Tests
```bash
pytest tests/ -v --cov=src
```

### Integration Tests
```bash
pytest tests/ -m integration
```

### MCP Server Tests
```bash
pytest tests/test_all_mcp_servers.py -v
```

### Trinity Agent Tests
```bash
pytest tests/test_trinity*.py -v
```

## 📦 Build Artifacts

### CI Artifacts (Temporary)
- Coverage reports (XML, HTML)
- Test results (JUnit XML)
- Build logs
- Retention: 7 days

### Release Artifacts (Permanent)
- `AtlasTrinity-vX.X.X-macOS-arm64.dmg` - macOS installer
- `checksums.txt` - SHA-256 checksums
- `CHANGELOG.md` - Release notes

## 🔍 Monitoring

### CI Status Badges

Add to README.md:
```markdown
![CI](https://github.com/olegkizima01/atlastrinity/workflows/CI%20Core%20Pipeline/badge.svg)
![MCP Tests](https://github.com/olegkizima01/atlastrinity/workflows/MCP%20Server%20Tests/badge.svg)
```

### Health Report

Daily health checks generate:
- `health-summary.md` - Overall system status
- `security-reports/` - Vulnerability scans
- `benchmarks/` - Performance metrics

## 🐛 Troubleshooting

### Common Issues

**1. MCP Server Connection Failures**
```bash
# Check configuration
python scripts/ci/validate_mcp_config.py

# Test individual server
pytest tests/test_all_mcp_servers.py -k <server_name> -v
```

**2. Build Failures on macOS**
```bash
# Check Swift version
swift --version

# Verify binary compilation
cd vendor/mcp-server-macos-use
swift build -c release
```

**3. Test Timeouts**
- Increase timeout in workflow: `timeout-minutes: 30`
- Or in pytest: `pytest --timeout=120`

**4. Coverage Drop**
- Check `coverage-report` artifact
- Review new code additions
- Add missing test cases

### Debug Mode

Enable debug logging in workflows:
```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

## 🤝 Contributing

### Pre-commit Checks

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

### Before Pushing

```bash
# Lint
npm run lint:all

# Format
npm run format:write

# Test
npm run test:ci

# Validate configs
python scripts/ci/validate_mcp_config.py
```

## 📝 Changelog

### Version 1.0 (2026-01-12)
- ✨ Initial CI/CD implementation
- ✅ 6 comprehensive workflows
- 🔧 Linting configurations
- 🧪 Full test coverage
- 📦 Automated releases
- 🔍 Daily health monitoring
- 🔒 Security scanning
- 📊 Performance benchmarks

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Electron Builder](https://www.electron.build/)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [AtlasTrinity Documentation](../docs/)

## 🔐 Security

- All secrets are stored in GitHub repository settings
- Never commit API keys or credentials to git
- Use environment variables for sensitive data
- Review Dependabot PRs for vulnerabilities
- Check security scan reports regularly

## 📧 Support

For CI/CD issues:
1. Check workflow logs in Actions tab
2. Review this documentation
3. Open an issue with workflow logs attached
4. Tag with `ci/cd` label

---

**AtlasTrinity CI/CD** 🔱 *Automated Excellence*
