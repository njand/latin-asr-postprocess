import pytest
from latin_itn_training.cltk_tokenizer import (
    CLTKLegacyLatinTokenizer,
    rejoin_cltk_enclitics_and_format,
)


@pytest.fixture
def tokenizer():
    return CLTKLegacyLatinTokenizer()


def test_cltk_tokenizer_replacements(tokenizer):
    """Test standard Latin ablative/cum and contraction replacements."""
    tokens = tokenizer.tokenize("mecum tecum sodes")
    words = [t[0] for t in tokens]
    assert words == ["cum", "me", "cum", "te", "si", "audes"]


def test_cltk_tokenizer_enclitics(tokenizer):
    """Test splitting of -que, -ne, and -ve enclitics when not exception words."""
    # 'armaque' should split into ('arma', False) and ('-que', True)
    tokens = tokenizer.tokenize("armaque")
    assert [(t.token_str, t.is_enclitic) for t in tokens] == [("arma", False), ("-que", True)]


@pytest.mark.parametrize("word", ["atque", "bene", "potest", "neve", "atque", "nomine"])
def test_cltk_exception_words_do_not_split(tokenizer, word):
    """Verify exception list words remain single tokens."""
    tokens = tokenizer.tokenize(word)
    assert len(tokens) == 1
    assert tokens[0].token_str == word
    assert not tokens[0].is_enclitic

def test_cltk_tokenizer_st_enclitic(tokenizer):
    """Test handling of -st (est) enclitic contractions."""
    tokens = tokenizer.tokenize("qualist")
    words = [t[0] for t in tokens]
    # 'qualist' replacement rule produces 'qualis est'
    assert words == ["qualis", "est"]


def test_rejoin_cltk_enclitics_and_format():
    """Test rejoining hyphenated enclitics and formatting casing and punctuation."""
    tokens_meta = [("arma", False), ("virum", False), ("-que", True), ("cano", False)]
    tags = ["TITLE_NONE", "LOWER_NONE", "LOWER_COMMA", "LOWER_PERIOD"]

    formatted = rejoin_cltk_enclitics_and_format(tokens_meta, tags)
    assert formatted == "Arma virumque, cano."


def test_cltk_enclitic_at_index_zero():
    """Ensure enclitic marked at index 0 (no host word) handles formatting gracefully without crashing."""
    tokens_meta = [("-que", True), ("virum", False)]
    tags = ["LOWER_NONE", "LOWER_PERIOD"]

    # Should not crash on boundary; falls back to appending string cleanly
    formatted = rejoin_cltk_enclitics_and_format(tokens_meta, tags)
    assert formatted == "que virum."


def test_cltk_consecutive_enclitics():
    """Test rejoining multiple enclitics in sequence (e.g., host-que-ne)."""
    tokens_meta = [("arma", False), ("-que", True), ("-ne", True)]
    tags = ["TITLE_NONE", "LOWER_NONE", "LOWER_QUESTION"]

    formatted = rejoin_cltk_enclitics_and_format(tokens_meta, tags)
    assert formatted == "Armaquene?"


def test_cltk_mismatched_tags_and_tokens():
    """Ensure mismatched tag/token lengths raise a ValueError."""
    tokens_meta = [("arma", False), ("-que", True)]
    tags = ["TITLE_NONE"]  # Mismatched length

    with pytest.raises(ValueError):
        rejoin_cltk_enclitics_and_format(tokens_meta, tags)


def test_cltk_st_contraction_formatting(tokenizer):
    """Verify how '-st' (est) contractions rejoin during post-processing."""
    raw_tokens = tokenizer.tokenize("similist")
    tags = ["TITLE_NONE", "LOWER_PERIOD"]
    formatted = rejoin_cltk_enclitics_and_format(raw_tokens, tags)
    
    assert formatted == "Similist.", f"Unexpected formatting: {formatted}"


def test_cltk_tokenizer_special_enclitic_splits(tokenizer):
    """Test -n -> -ne and -st -> est special enclitic splitting."""
    # Test -n expansion (e.g. 'viden' -> 'vide' + '-ne')
    tokens_n = tokenizer.tokenize("viden")
    assert [(t.token_str, t.is_enclitic) for t in tokens_n] == [("vide", False), ("-ne", True)]

    # Test -st expansion with non-'ust' stem (e.g. 'homost' -> 'homo' + 'est')
    tokens_st = tokenizer.tokenize("homost")
    assert [(t.token_str, t.is_enclitic) for t in tokens_st] == [("homo", False), ("est", True)]

    # Test -st expansion with 'ust' stem (e.g. 'venust' -> 'venus' + 'est')
    tokens_ust = tokenizer.tokenize("venust")
    assert [(t.token_str, t.is_enclitic) for t in tokens_ust] == [("venus", False), ("est", True)]


def test_cltk_empty_inputs(tokenizer):
    """Verify empty strings and empty lists format cleanly."""
    assert tokenizer.tokenize("") == []
    assert rejoin_cltk_enclitics_and_format([], []) == ""