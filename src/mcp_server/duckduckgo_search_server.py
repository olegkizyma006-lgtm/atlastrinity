import html
import json
import os
import re
from typing import Any

import requests
from mcp.server import FastMCP

server = FastMCP("search-server")

# Path to protocol file (centralized configuration)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROTOCOL_PATH = os.path.join(PROJECT_ROOT, "src", "brain", "data", "search_protocol.txt")


def _load_search_rules() -> dict[str, Any]:
    """Load search rules from the machine-readable section of the search protocol."""
    try:
        if not os.path.exists(PROTOCOL_PATH):
            return {}
        with open(PROTOCOL_PATH, encoding="utf-8") as f:
            content = f.read()

        # Find JSON block (starts after 5. MACHINE-READABLE CONFIGURATION)
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            config = json.loads(match.group(0))
            return config.get("search_rules", {})
    except Exception as e:
        print(f"[SEARCH-SERVER] Error loading protocol: {e}")
    return {}


def _execute_protocol_search(rule_name: str, query_val: str, provider_name: str) -> dict[str, Any]:
    """Execute search based on rules defined in the protocol."""
    rules = _load_search_rules()
    rule = rules.get(rule_name, {})

    operator = rule.get("operator", "{query}")
    priority_domains = rule.get("priority_domains", [])

    # Expand template
    expanded_query = operator.replace("{query}", query_val.strip())

    try:
        results = _search_ddg(query=expanded_query, max_results=10, timeout_s=15.0)

        prioritized = []
        others = []

        for r in results:
            if any(domain in r["url"] for domain in priority_domains):
                prioritized.append(r)
            else:
                others.append(r)

        return {
            "success": True,
            "query": expanded_query,
            "results": prioritized + others,
            "provider": provider_name,
        }
    except Exception as e:
        return {"error": str(e)}


def _search_ddg(query: str, max_results: int, timeout_s: float) -> list[dict[str, Any]]:
    url = "https://html.duckduckgo.com/html/"
    try:
        resp = requests.post(
            url,
            data={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": "https://duckduckgo.com/",
                "Origin": "https://duckduckgo.com",
            },
            timeout=timeout_s,
        )
        resp.raise_for_status()
        if not resp.text.strip():
            print("[SEARCH-SERVER] Empty response from DDG")
            return []

        if "Checking your browser" in resp.text or "Cloudflare" in resp.text:
            print("[SEARCH-SERVER] Blocked by CAPTCHA/Cloudflare")
            return []
    except Exception as e:
        print(f"[SEARCH-SERVER] Request error: {e}")
        return []

    # Minimal regex to find ALL links and their text
    pattern = re.compile(
        r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )

    results: list[dict[str, Any]] = []
    for match in pattern.finditer(resp.text):
        href = html.unescape(match.group(1)).strip()

        # Filter: Exclude internal DDG links and obvious junk
        if any(x in href for x in ["duckduckgo.com/", "ad_redirect", "javascript:", "#"]):
            if "duckduckgo.com/l/?" not in href:  # Keep result redirects if needed
                continue

        title_html = match.group(2)
        title = html.unescape(re.sub(r"<.*?>", "", title_html)).strip()

        if not href or not title or len(title) < 2:
            continue

        # Clean up DDG redirects if present
        if "uddg=" in href:
            # Extract real URL from the 'uddg' parameter
            match_real = re.search(r"uddg=([^&]+)", href)
            if match_real:
                from urllib.parse import unquote

                href = unquote(match_real.group(1))

        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = "https://duckduckgo.com" + href

        # Filter out links that are likely UI elements (too short or generic)
        if any(title.lower() == x for x in ["next", "previous", "images", "videos", "news"]):
            continue

        results.append({"title": title, "url": href})
        if len(results) >= max_results:
            break

    return results


@server.tool()
def duckduckgo_search(
    query: str, max_results: int = 5, timeout_s: float = 10.0, step_id: str | None = None
) -> dict[str, Any]:
    """
    Perform a web search using DuckDuckGo.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5, max: 20)
        timeout_s: Request timeout in seconds (default: 10.0)
    """
    if not query or not query.strip():
        return {"error": "query is required"}

    try:
        max_results_i = int(max_results)
        if max_results_i <= 0:
            return {"error": "max_results must be > 0"}
        max_results_i = min(max_results_i, 20)

        timeout_f = float(timeout_s)
        if timeout_f <= 0:
            return {"error": "timeout_s must be > 0"}

        results = _search_ddg(query=query.strip(), max_results=max_results_i, timeout_s=timeout_f)
        if not results:
            return {
                "success": False,
                "error": "Zero results found. DDG might be blocking or layout changed.",
                "query": query.strip(),
            }
        return {"success": True, "query": query.strip(), "results": results}
    except Exception as e:
        return {"error": str(e)}


@server.tool()
def business_registry_search(company_name: str, step_id: str | None = None) -> dict[str, Any]:
    """
    Perform a specialized search for Ukrainian company data in business registries.
    Rules and targets are defined in search_protocol.txt.

    Args:
        company_name: The name or EDRPOU code of the company to search for.
    """
    if not company_name or not company_name.strip():
        return {"error": "company_name is required"}

    return _execute_protocol_search(
        "business", company_name, "DuckDuckGo (Optimized Registry Search)"
    )


@server.tool()
def open_data_search(query: str, step_id: str | None = None) -> dict[str, Any]:
    """
    Search for datasets on the Ukrainian Open Data Portal (data.gov.ua).
    Rules and targets are defined in search_protocol.txt.

    Args:
        query: The search query for datasets.
    """
    if not query or not query.strip():
        return {"error": "query is required"}

    return _execute_protocol_search("open_data", query, "DuckDuckGo (Open Data Portal Search)")


if __name__ == "__main__":
    server.run()
