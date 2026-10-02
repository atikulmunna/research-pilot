from research_pilot.services.literature import (
    ArxivProvider,
    LiteratureSearch,
    MockLiteratureProvider,
    OpenAlexProvider,
    SemanticScholarProvider,
    dedupe,
    reconstruct_abstract,
)

OPENALEX_ROW = {
    "id": "https://openalex.org/W4389984066",
    "doi": "https://doi.org/10.48550/arxiv.2312.10997",
    "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
    "publication_year": 2023,
    "authorships": [{"author": {"display_name": "Yunfan Gao"}}, {"author": {"display_name": "Yun Xiong"}}],
    "abstract_inverted_index": {"Large": [0], "models": [2], "language": [1]},
    "cited_by_count": 749,
    "referenced_works": ["https://openalex.org/W1"],
    "primary_location": {"landing_page_url": "http://arxiv.org/abs/2312.10997", "source": {"display_name": "arXiv (Cornell University)"}},
    "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/2312.10997"},
}

ARXIV_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2506.06962v3</id>
    <title>AR-RAG: Autoregressive Retrieval
      Augmentation for Image Generation</title>
    <published>2025-06-08T00:00:00Z</published>
    <summary>We introduce AR-RAG.</summary>
    <author><name>Jingyuan Qi</name></author>
    <link href="https://arxiv.org/abs/2506.06962v3" rel="alternate" type="text/html"/>
    <link href="https://arxiv.org/pdf/2506.06962v3" rel="related" type="application/pdf" title="pdf"/>
  </entry>
</feed>"""


def test_reconstruct_abstract():
    assert reconstruct_abstract(OPENALEX_ROW["abstract_inverted_index"]) == "Large language models"
    assert reconstruct_abstract(None) == ""


def test_openalex_parse():
    paper = OpenAlexProvider.parse(OPENALEX_ROW)
    assert paper.title.startswith("Retrieval-Augmented") and paper.year == 2023
    assert paper.doi == "10.48550/arxiv.2312.10997" and paper.arxiv_id == "2312.10997"
    assert paper.authors == ["Yunfan Gao", "Yun Xiong"]
    assert paper.pdf_url.endswith("2312.10997") and paper.cited_by_count == 749
    assert paper.references == ["https://openalex.org/W1"] and paper.source == "openalex"


def test_semantic_scholar_parse():
    paper = SemanticScholarProvider.parse(
        {"paperId": "abc", "title": "T", "year": 2020, "authors": [{"name": "A"}], "externalIds": {"DOI": "10.1/x", "ArXiv": "2001.1"}, "citationCount": 5, "openAccessPdf": {"url": "u"}}
    )
    assert (paper.s2_id, paper.doi, paper.arxiv_id, paper.pdf_url, paper.source) == ("abc", "10.1/x", "2001.1", "u", "semantic_scholar")


def test_arxiv_feed_parse():
    papers = ArxivProvider.parse_feed(ARXIV_FEED)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.title == "AR-RAG: Autoregressive Retrieval Augmentation for Image Generation"
    assert paper.arxiv_id == "2506.06962v3" and paper.year == 2025 and paper.pdf_url.endswith("v3")


def test_dedupe_merges_across_providers():
    openalex = OpenAlexProvider.parse(OPENALEX_ROW)
    arxiv = ArxivProvider.parse_feed(ARXIV_FEED)[0]
    same = arxiv.model_copy(update={"arxiv_id": "2312.10997v2", "title": "different title", "doi": ""})
    merged = dedupe([openalex, arxiv, same])
    assert len(merged) == 2


class Boom:
    name = "boom"

    def search(self, query, limit):
        raise RuntimeError("down")


def test_provider_failures_become_warnings():
    search = LiteratureSearch([Boom(), MockLiteratureProvider()])
    hits = search.search("graph learning", 3)
    assert len(hits) == 3 and all(p.synthetic for p in hits)
    assert "boom search failed" in search.drain_warnings()[0]


def test_mock_expand_gives_citation_neighbours():
    search = LiteratureSearch([MockLiteratureProvider()])
    paper = search.search("x", 1)[0]
    links = search.expand(paper, 5)
    assert len(links["references"]) == 2 and len(links["citing"]) == 2
