from agent.verify import extract_citation_ids, find_uncited_sentences, renumber, strip_invalid


def test_extract_in_order_with_grouped_ids():
    assert extract_citation_ids("A [3]. B [1, 3]. C [2].") == [3, 1, 2]


def test_strip_invalid_removes_unknown_ids():
    text, removed = strip_invalid("Fact [1][9]. Other [2, 7].", {1, 2})
    assert text == "Fact [1]. Other [2]."
    assert removed == 2


def test_renumber_follows_order_of_appearance():
    text, mapping = renumber("First [5]. Second [2][5]. Third [9].")
    assert text == "First [1]. Second [2][1]. Third [3]."
    assert mapping == {5: 1, 2: 2, 9: 3}


def test_uncited_sentences_ignore_headings_and_short_lines():
    report = (
        "# Title\n"
        "This is a long factual sentence without any citation attached to it at all. "
        "This one is supported by a source [1].\n"
        "Short."
    )
    flagged = find_uncited_sentences(report)
    assert len(flagged) == 1
    assert flagged[0].startswith("This is a long factual sentence")
