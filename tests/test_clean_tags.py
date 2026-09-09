from support_agent.data.clean import clean_text, content_tokens, detect_lang, normalize_for_dedup
from support_agent.data.tags import primary_reply_type, tag_reply


def test_clean_text_handles_mentions_urls_and_entities():
    raw = "@115712 @SpotifyCares it&amp;#39;s broken again https://t.co/xyz123   help"
    assert clean_text(raw) == "it&#39;s broken again <url> help".replace("&#39;", "'") or clean_text(raw) == "it&#39;s broken again <url> help"
    assert clean_text(raw, brand="SpotifyCares") == clean_text(raw, brand="spotifycares")
    assert "<brand>" in clean_text(raw, brand="SpotifyCares")
    # the brand's anonymised main account maps to <brand> too; a genuinely other handle is dropped
    assert clean_text("@115888 @999 app broken", brand="SpotifyCares", aliases=["115888"]) == "<brand> app broken"


def test_content_tokens_ignore_placeholders():
    assert content_tokens("@SpotifyCares https://t.co/abc") == []
    assert content_tokens("@SpotifyCares app keeps crashing https://t.co/abc") == ["app", "keeps", "crashing"]


def test_strip_signoffs():
    from support_agent.data.clean import strip_signoffs

    assert strip_signoffs("Let us know how it goes /JR") == "Let us know how it goes"
    assert strip_signoffs("We'll look backstage /LO https://t.co/x") == "We'll look backstage https://t.co/x"
    assert strip_signoffs("Try the A/B toggle ^GT. Thanks!") == "Try the A/B toggle . Thanks!"
    assert strip_signoffs("no initials here") == "no initials here"


def test_normalize_for_dedup_collapses_variants():
    a = "@SpotifyCares Spotify is DOWN!!! https://t.co/a"
    b = "@SpotifyCares spotify is down... https://t.co/b"
    assert normalize_for_dedup(a) == normalize_for_dedup(b) == "spotify is down"


def test_detect_lang_short_text_is_unknown():
    assert detect_lang("@SpotifyCares 👀") == "unk"
    assert detect_lang("@SpotifyCares my app keeps crashing every time I open a playlist") == "en"


def test_reply_type_priority():
    assert primary_reply_type("@1 Sorry! Please DM us your account email and we'll take a look. /JR") == "dm_redirect"
    assert primary_reply_type("@1 Try a clean reinstall of the app: https://t.co/abc") == "steps"
    assert primary_reply_type("@1 More info here: https://t.co/abc") == "link"
    assert primary_reply_type("@1 Which device and OS version are you on?") == "clarifying_question"
    assert primary_reply_type("@1 So sorry about that!") == "apology_only"
    assert primary_reply_type("@1 Thanks for the love!") == "other"
    assert {"dm_redirect", "apology"} <= tag_reply("@1 Sorry! DM us please")
