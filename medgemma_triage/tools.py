import httpx

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(query: str, max_results: int = 3) -> str:
    """
    Search PubMed for articles matching the query.
    Returns a string summary of the top articles (Title + Abstract).
    """
    try:
        # Step 1: Search for article IDs
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        }
        
        with httpx.Client(timeout=30.0) as client:
            search_response = client.get(PUBMED_SEARCH_URL, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()
        
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            return "No articles found for this query."
        
        # Step 2: Fetch article details
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
            "rettype": "abstract",
        }
        
        with httpx.Client(timeout=30.0) as client:
            fetch_response = client.get(PUBMED_FETCH_URL, params=fetch_params)
            fetch_response.raise_for_status()
            xml_content = fetch_response.text
        
        # Step 3: Parse XML (simple extraction)
        articles = _parse_pubmed_xml(xml_content)
        
        if not articles:
            return "Could not parse article details."
        
        # Format output
        result_parts = []
        for i, article in enumerate(articles, 1):
            result_parts.append(f"**Article {i}:**")
            result_parts.append(f"Title: {article.get('title', 'N/A')}")
            result_parts.append(f"Abstract: {article.get('abstract', 'N/A')[:500]}...")
            result_parts.append("")
        
        return "\n".join(result_parts)
    
    except httpx.HTTPStatusError as e:
        return f"Search failed: HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        return f"Search failed: {str(e)}"
    except Exception as e:
        return f"Search failed: {str(e)}"


def _parse_pubmed_xml(xml_content: str) -> list:
    """
    Simple XML parsing for PubMed articles.
    Extracts title and abstract from each article.
    """
    import re
    
    articles = []
    
    # Find all PubmedArticle blocks
    article_blocks = re.findall(r'<PubmedArticle>(.*?)</PubmedArticle>', xml_content, re.DOTALL)
    
    for block in article_blocks:
        article = {}
        
        # Extract Title
        title_match = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', block, re.DOTALL)
        if title_match:
            article['title'] = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
        
        # Extract Abstract
        abstract_match = re.search(r'<AbstractText[^>]*>(.*?)</AbstractText>', block, re.DOTALL)
        if abstract_match:
            article['abstract'] = re.sub(r'<[^>]+>', '', abstract_match.group(1)).strip()
        else:
            article['abstract'] = "No abstract available."
        
        articles.append(article)
    
    return articles
