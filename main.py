import requests

# GET /query -- Search for papers on arXiv based on specific search criteria
url = "https://export.arxiv.org/query"
params = {"q": "machine learning", "max_results": 10}
search_criteria = requests.get(url, params=params)

# GET /abs/:id  -- Retrieve detailed information on a specific paper by its arXiv identifier.
url = "https://export.arxiv.org/abs/2003.10810"
specific_paper = requests.get(url)

# GET /list/:set -- Retrieve a list of papers based on specific criteria.
url = "https://export.arxiv.org/api/query"
params = {"search_query": "machine learning", "max_results": 10}
list_paper = requests.get(url, params=params)

print(f"{search_criteria.text}\n\n")
#print(f"{specific_paper.text}\n\n")
#print(f"{list_paper.text}\n\n")
