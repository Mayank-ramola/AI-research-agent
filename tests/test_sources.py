from agent.sources import SourceStore, normalize_url


def test_normalize_url_ignores_tracking_www_and_slash():
    a = normalize_url("https://www.Example.com/post/?utm_source=x&id=1#top")
    b = normalize_url("http://example.com/post?id=1")
    assert a == b


def test_store_merges_duplicate_urls():
    store = SourceStore()
    first = store.add("Title", "https://example.com/a", "short")
    second = store.add("Title copy", "https://www.example.com/a/?utm_campaign=z", "a longer snippet")
    assert first.id == second.id
    assert len(store.all()) == 1
    assert store.duplicates_skipped == 1
    assert store.get(first.id).snippet == "a longer snippet"


def test_store_merges_same_title_on_same_domain():
    store = SourceStore()
    store.add("Same Title", "https://example.com/a")
    store.add("Same title", "https://example.com/amp/a")
    assert len(store.all()) == 1


def test_different_sources_get_new_ids():
    store = SourceStore()
    assert store.add("A", "https://a.com/1").id == 1
    assert store.add("B", "https://b.com/1").id == 2
