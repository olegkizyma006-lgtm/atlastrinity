import html
import re
from typing import Any, Dict, List, Optional

import requests
from mcp.server import FastMCP

server = FastMCP("search-server")


def _search_ddg(query: str, max_results: int, timeout_s: float) -> List[Dict[str, Any]]:
    url = "https://html.duckduckgo.com/html/"
    resp = requests.post(
        url,
        data={"q": query},
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://duckduckgo.com/",
            "Origin": "https://duckduckgo.com"
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()

    # Broaden regex to find any result link
    # Usually in the form: <a class="result__a" rel="nofollow" href="...">Title</a>
    pattern = re.compile(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )

    results: List[Dict[str, Any]] = []
    for match in pattern.finditer(resp.text):
        href = html.unescape(match.group(1)).strip()
        
        # Clean up DDG redirects if present in href
        if "duckduckgo.com/l/?" in href:
             # Extract real URL from the 'uddg' parameter or similar if needed
             pass
             
        title_html = match.group(2)
        title = html.unescape(re.sub(r"<.*?>", "", title_html)).strip()
        
        if not href or not title:
            continue
            
        # Ensure we're not just getting internal DDG links
        if href.startswith("//"):
             href = "https:" + href
        elif href.startswith("/"):
             href = "https://duckduckgo.com" + href
             
        results.append({"title": title, "url": href})
        if len(results) >= max_results:
            break

    return results


@server.tool()
def duckduckgo_search(query: str, max_results: int = 5, timeout_s: float = 10.0, step_id: Optional[str] = None) -> Dict[str, Any]:
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
        return {"success": True, "query": query.strip(), "results": results}
    except Exception as e:
        return {"error": str(e)}

@server.tool()
def business_registry_search(company_name: str, step_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform a specialized search for Ukrainian company data in business registries.
    Targets YouControl, Opendatabot, and EDRPOU.

    Args:
        company_name: The name or EDRPOU code of the company to search for.
    """
    if not company_name or not company_name.strip():
        return {"error": "company_name is required"}

    # Optimized operators for registries
    query = f"(site:youcontrol.com.ua OR site:opendatabot.ua OR site:clarity-project.info) \"{company_name.strip()}\""
    
    try:
        # We use a higher max_results for registries to capture various snippets
        results = _search_ddg(query=query, max_results=10, timeout_s=15.0)
        
        # Simple prioritization: move highly relevant domains to the top
        priority_domains = ["youcontrol.com.ua", "opendatabot.ua", "clarity-project.info"]
        
        prioritized = []
        others = []
        
        for r in results:
            if any(domain in r["url"] for domain in priority_domains):
                prioritized.append(r)
            else:
                others.append(r)
                
        return {
            "success": True, 
            "query": query, 
            "results": prioritized + others,
            "provider": "DuckDuckGo (Optimized Registry Search)"
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    server.run()
